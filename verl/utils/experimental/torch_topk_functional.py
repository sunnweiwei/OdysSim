# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TODO@Weiwei[OPD-TopK] Fused kernel for top-k OPD (On-Policy Distillation).

Extends FusedLinearForPPO with top-k token support:
- extract_topk: teacher mode, extracts top-k token ids and log probs (no grad)
- forward: student mode, computes token_log_probs + entropy (same as FusedLinearForPPO)
           AND gathers log probs at given top-k positions (with grad for KL loss)

The logits [T, V] are computed chunk-by-chunk and discarded immediately,
so full vocab-size tensors are never held in memory across the full sequence.
"""

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Chunk-level forward / backward helpers
# ---------------------------------------------------------------------------


def _topk_fwd(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    gather_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Single-chunk forward: token log prob + entropy + gathered log probs.

    Args:
        hidden_states: [chunk, D]
        vocab_weights: [V, D]
        input_ids:     [chunk]      — sampled token ids
        gather_ids:    [chunk, k]   — top-k positions to gather
        temperature:   softmax temperature

    Returns:
        token_log_probs:   [chunk]
        entropy:           [chunk]
        gathered_log_probs: [chunk, k]
    """
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)

    # chunk-level cast to int64 for gather (input may be int32 for memory efficiency)
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1).to(torch.int64)).squeeze(-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    gathered_log_probs = log_probs.gather(-1, gather_ids.to(torch.int64))  # [chunk, k]

    return (
        token_log_probs.to(orig_dtype),
        entropy.to(orig_dtype),
        gathered_log_probs.to(orig_dtype),
    )


def _topk_bwd(
    dlog_probs: Optional[torch.FloatTensor],
    dentropy: Optional[torch.FloatTensor],
    d_gathered: Optional[torch.FloatTensor],
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    gather_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """Single-chunk backward for token_log_probs, entropy, and gathered_log_probs.

    For log_softmax output y_i gathered at position j:
        dy_j / d(logit_i) = delta_{ij} - softmax_i

    For k gathered positions the total gradient is:
        dL/d(logit_i) = sum_j [ d_gathered_j * (delta_{gather_j, i} - probs_i) ]
                      = scatter(d_gathered) - probs * sum(d_gathered)
    This is combined with the single-token and entropy gradients from the original kernel.
    """
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)

    dlogits = 0

    # chunk-level cast to int64 for scatter/gather (input may be int32)
    input_ids_i64 = input_ids.to(torch.int64)
    gather_ids_i64 = gather_ids.to(torch.int64)

    # Gradient from single-token log_probs (same as original kernel)
    if dlog_probs is not None:
        one_hot = torch.zeros_like(logits).scatter_(-1, input_ids_i64.unsqueeze(-1), 1)
        dlogits = dlogits + dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot - probs)

    # Gradient from entropy (same as original kernel)
    if dentropy is not None:
        log_probs = logits.log_softmax(dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        dlogits = dlogits + probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))

    # Gradient from gathered log_probs at k positions
    if d_gathered is not None:
        d_gathered = d_gathered.to(torch.float32)
        # scatter per-position gradients into vocab dimension
        d_from_gather = torch.zeros_like(logits)
        d_from_gather.scatter_add_(-1, gather_ids_i64, d_gathered)
        # subtract probs * sum(upstream) — the softmax Jacobian term
        d_from_gather -= probs * d_gathered.sum(-1, keepdim=True)
        dlogits = dlogits + d_from_gather

    dlogits = dlogits.to(orig_dtype) / temperature

    dhidden_states = dlogits @ vocab_weights
    dvocab_weights = dlogits.t() @ hidden_states

    return dhidden_states, dvocab_weights


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class FusedLinearForPPOTopKFunction(torch.autograd.Function):
    """Fused lm_head + softmax + gather for PPO with top-k OPD support.

    Returns (token_log_probs, entropy, gathered_log_probs) in a single pass
    through the vocabulary projection, chunked to bound peak memory.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        gather_ids: torch.LongTensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        ctx.set_materialize_grads(False)

        orig_ndim = hidden_states.ndim
        assert orig_ndim in (2, 3), f"Invalid hidden_states shape: {hidden_states.shape}"

        orig_batch_size = -1
        if orig_ndim == 3:
            orig_batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.flatten(0, 1)
            input_ids = input_ids.flatten(0, 1)
            gather_ids = gather_ids.flatten(0, 1)

        T = hidden_states.shape[0]
        k = gather_ids.shape[1]

        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad
        log_probs = hidden_states.new_zeros(T, requires_grad=output_requires_grad)
        entropy = hidden_states.new_zeros(T, requires_grad=output_requires_grad)
        gathered = hidden_states.new_zeros(T, k, requires_grad=output_requires_grad)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            c_log_probs, c_entropy, c_gathered = _topk_fwd(
                hidden_states=hidden_states[start:end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[start:end],
                gather_ids=gather_ids[start:end],
                temperature=temperature,
            )
            log_probs[start:end] = c_log_probs
            entropy[start:end] = c_entropy
            gathered[start:end] = c_gathered

        if orig_ndim == 3:
            log_probs = log_probs.view(orig_batch_size, -1)
            entropy = entropy.view(orig_batch_size, -1)
            gathered = gathered.view(orig_batch_size, -1, k)

        ctx.save_for_backward(hidden_states, vocab_weights, input_ids, gather_ids)
        ctx.orig_batch_size = orig_batch_size
        ctx.orig_ndim = orig_ndim
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size

        return log_probs, entropy, gathered

    @staticmethod
    def backward(
        ctx,
        dlog_probs: Optional[torch.FloatTensor],
        dentropy: Optional[torch.FloatTensor],
        d_gathered: Optional[torch.FloatTensor],
    ):
        if dlog_probs is None and dentropy is None and d_gathered is None:
            return None, None, None, None, None, None

        hidden_states, vocab_weights, input_ids, gather_ids = ctx.saved_tensors
        orig_ndim = ctx.orig_ndim
        orig_batch_size = ctx.orig_batch_size
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size

        if orig_ndim == 3:
            if dlog_probs is not None:
                dlog_probs = dlog_probs.flatten()
            if dentropy is not None:
                dentropy = dentropy.flatten()
            if d_gathered is not None:
                d_gathered = d_gathered.flatten(0, 1)

        T = hidden_states.shape[0]

        dhidden_states = None
        if hidden_states.requires_grad:
            dhidden_states = torch.zeros_like(hidden_states)
        dvocab_weights = None
        if vocab_weights.requires_grad:
            dvocab_weights = torch.zeros_like(vocab_weights)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)

            c_dlog_probs = dlog_probs[start:end] if dlog_probs is not None else None
            c_dentropy = dentropy[start:end] if dentropy is not None else None
            c_d_gathered = d_gathered[start:end] if d_gathered is not None else None

            h, v = _topk_bwd(
                dlog_probs=c_dlog_probs,
                dentropy=c_dentropy,
                d_gathered=c_d_gathered,
                hidden_states=hidden_states[start:end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[start:end],
                gather_ids=gather_ids[start:end],
                temperature=temperature,
            )

            if hidden_states.requires_grad:
                dhidden_states[start:end] += h
            if vocab_weights.requires_grad:
                dvocab_weights += v

        if orig_ndim == 3 and hidden_states.requires_grad:
            hidden_size = hidden_states.shape[-1]
            dhidden_states = dhidden_states.view(orig_batch_size, -1, hidden_size)

        return (
            dhidden_states,   # hidden_states
            dvocab_weights,   # vocab_weights
            None,             # input_ids
            None,             # gather_ids
            None,             # temperature
            None,             # chunk_size
        )


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------


class FusedLinearForPPOTopK(torch.nn.Module):
    """Fused kernel for top-k OPD distillation.

    Two operations, both chunk logits to bound memory:

    1. extract_topk (teacher, no grad):
       hidden_states, vocab_weights → topk_ids [T, k], topk_log_probs [T, k]

    2. forward (student, with grad):
       hidden_states, vocab_weights, input_ids, gather_ids
       → token_log_probs [T], entropy [T], gathered_log_probs [T, k]
    """

    def __init__(self, chunk_size: int = 512):
        super().__init__()
        self.chunk_size = chunk_size

    @torch.no_grad()
    def extract_topk(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        topk: int,
        temperature: float = 1.0,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """Extract top-k token ids and their log probs. No gradient.

        Used in teacher forward under torch.no_grad().

        Args:
            hidden_states: [T, D] or [B, T, D]
            vocab_weights: [V, D]
            topk:          number of top tokens to extract
            temperature:   softmax temperature

        Returns:
            topk_ids:       [T, k] or [B, T, k]  (int32)
            topk_log_probs: [T, k] or [B, T, k]
        """
        orig_ndim = hidden_states.ndim
        orig_batch_size = -1
        if orig_ndim == 3:
            orig_batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.flatten(0, 1)

        T = hidden_states.shape[0]
        topk_ids = torch.zeros(T, topk, dtype=torch.int32, device=hidden_states.device)
        topk_log_probs = torch.zeros(T, topk, dtype=hidden_states.dtype, device=hidden_states.device)

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            logits = (hidden_states[start:end] @ vocab_weights.t()) / temperature
            logits = logits.to(torch.float32)
            log_probs = logits.log_softmax(dim=-1)
            c_topk_log_probs, c_topk_ids = log_probs.topk(topk, dim=-1)
            topk_ids[start:end] = c_topk_ids
            topk_log_probs[start:end] = c_topk_log_probs.to(hidden_states.dtype)

        if orig_ndim == 3:
            topk_ids = topk_ids.view(orig_batch_size, -1, topk)
            topk_log_probs = topk_log_probs.view(orig_batch_size, -1, topk)

        return topk_ids, topk_log_probs

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        gather_ids: torch.LongTensor,
        temperature: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Fused forward: token log prob + entropy + gathered log probs at top-k positions.

        Used in student forward during update_policy. Gradients flow through all
        three outputs for policy loss (via token_log_probs) and KL loss (via gathered).

        Args:
            hidden_states: [T, D] or [B, T, D]
            vocab_weights: [V, D]
            input_ids:     [T] or [B, T]      — sampled token ids
            gather_ids:    [T, k] or [B, T, k] — teacher's top-k positions
            temperature:   softmax temperature

        Returns:
            token_log_probs:    [T] or [B, T]
            entropy:            [T] or [B, T]
            gathered_log_probs: [T, k] or [B, T, k]
        """
        input_ids = input_ids.to(torch.int64)
        # gather_ids stays int32 for memory efficiency; cast to int64 chunk-by-chunk inside kernel
        return FusedLinearForPPOTopKFunction.apply(
            hidden_states,
            vocab_weights,
            input_ids,
            gather_ids,
            temperature,
            self.chunk_size,
        )