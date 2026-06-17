# Copyright 2025 Individual Contributor: OdysSim Authors
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

import os

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

# Avoid deadlocks when tokenizer is used inside DataLoader worker processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Tokenizer wrapper ──────────────────────────────────────────────────────────


def _wrap_tokenizer(tokenizer):
    """
    Return tokenizer with apply_chat_template that disables thinking for all models.
    Controlled by TURNOFF_THINK env var (default: '1' = True).
    Tries thinking_budget=0 (Seed/Gemma style) then enable_thinking=False (Qwen3 style),
    falling back to a plain call if neither is supported.
    """
    turnoff_think = os.getenv("TURNOFF_THINK", "1").lower() not in ("0", "false", "no")
    _orig = tokenizer.apply_chat_template

    def _patched(*args, **kwargs):
        if not turnoff_think:
            return _orig(*args, **kwargs)
        for extra in [{"thinking_budget": 0}, {"enable_thinking": False}, {}]:
            try:
                return _orig(*args, **{**extra, **kwargs})
            except TypeError:
                continue
        return _orig(*args, **kwargs)

    tokenizer.apply_chat_template = _patched
    return tokenizer


# ── File-list parsing ──────────────────────────────────────────────────────────


def parse_files(files_arg) -> list[tuple[str, float]]:
    """
    Parse train_files / val_files arg into list of (path, ratio).

    Accepts:
      - Hydra list:  [/data/a.parquet, /data/b.parquet:1.5]
      - String:      "/data/a.parquet /data/b.parquet:1.5"
      - Single str:  "/data/a.parquet"

    Ratio semantics:
      ratio=1   → use natural dataset size (default)
      ratio=1.5 → oversample 1.5×
      ratio=2   → oversample 2×
      ratio=0.5 → subsample 50%
    """
    import glob as _glob

    if isinstance(files_arg, str):
        tokens = files_arg.strip().split()
    else:
        tokens = list(files_arg)

    result = []
    for token in tokens:
        # Split on last ":" to get optional ratio.
        # Safe for paths like s3://bucket/file.parquet (no trailing colon).
        parts = token.rsplit(":", 1)
        try:
            ratio = float(parts[-1])
            path = parts[0]
        except (ValueError, IndexError):
            ratio = 1.0
            path = token
        # Glob expansion for paths containing * or ?. Same ratio applies to each
        # matched shard — net effective rows = sum(shard_rows) * ratio, which
        # matches the intent of "oversample this dataset by ratio" regardless of
        # how many shards it's split into.
        if "*" in path or "?" in path:
            matched = sorted(_glob.glob(path))
            if not matched:
                raise FileNotFoundError(f"Glob matched no parquet files: {path}")
            for p in matched:
                result.append((p, ratio))
        else:
            result.append((path, ratio))
    return result


# ── Parquet batch loader (avoids pyarrow nested-chunked read bug) ─────────────


def _load_parquet_batches(path, batch_size=10000):
    """Return (batches, offsets) for random access via binary search.
    Uses pq.ParquetFile.iter_batches instead of pq.read_table to sidestep
    'Nested data conversions not implemented for chunked array outputs'
    on list<struct> columns with large per-row payloads."""
    import bisect as _bisect  # noqa: F401  (ensured imported; used in _slice_to_row)

    pf = pq.ParquetFile(path, memory_map=True)
    batches = list(pf.iter_batches(batch_size=batch_size))
    offsets = [0]
    for b in batches:
        offsets.append(offsets[-1] + b.num_rows)
    return batches, offsets


def _slice_to_row(batches, offsets, row_idx):
    import bisect

    bi = bisect.bisect_right(offsets, row_idx) - 1
    local_idx = row_idx - offsets[bi]
    raw = batches[bi].slice(local_idx, 1).to_pydict()
    return {k: v[0] for k, v in raw.items()}


# ── Collate ────────────────────────────────────────────────────────────────────


def sft_collate_fn(samples: list[dict], max_prompt_length: int, max_response_length: int) -> dict:
    """Pad sequences to [left_prompt_pad | prompt | response | right_resp_pad].

    Prompts are left-padded to max_prompt_length (global) so that
    _forward_micro_batch's [:, -response_length-1:-1] slice correctly
    extracts response log-probs regardless of variable prompt lengths.
    Responses are right-padded to the batch max (not global max) to keep
    tensor sizes small and avoid exhausting shared memory in DataLoader workers.
    The left-pad zeros have attention_mask=0 and are removed by unpad_input,
    so they cost no compute.
    """
    max_resp = min(max(s["responses"].shape[0] for s in samples), max_response_length)

    input_ids_list, attn_list, pos_list = [], [], []
    for s in samples:
        resp_len = s["responses"].shape[0]
        prompt_len = s["input_ids"].shape[0] - resp_len
        left_pad = max_prompt_length - prompt_len
        right_pad = max_resp - resp_len
        assert left_pad >= 0, f"prompt length {prompt_len} exceeds max_prompt_length {max_prompt_length}"
        assert right_pad >= 0, f"response length {resp_len} exceeds max_response_length {max_response_length}"

        input_ids_list.append(pad(s["input_ids"], (left_pad, right_pad), value=0))
        attn_list.append(pad(s["attention_mask"], (left_pad, right_pad), value=0))
        pos_list.append(pad(s["position_ids"], (left_pad, right_pad), value=0))

    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "position_ids": torch.stack(pos_list),
    }

    # Response-length tensors: right-pad to batch max response length.
    for key in ["responses", "response_mask", "old_log_probs", "advantages"]:
        result[key] = torch.stack([pad(s[key], (0, max_resp - s[key].shape[0]), value=0) for s in samples])

    # Non-tensor: per-row data_source strings (routed to DataProto.non_tensor_batch).
    result["data_source"] = np.array(
        [s.get("data_source", "unknown") for s in samples],
        dtype=object,
    )

    return result


# ── Dataset ────────────────────────────────────────────────────────────────────


class SFTDataset(Dataset):
    """
    Multi-source SFT dataset.

    Each item is processed by agents/sft_data.py::process(row, tokenizer),
    which the user implements to return:
        input_ids      (full_seq_len,)   long
        attention_mask (full_seq_len,)   long
        position_ids   (full_seq_len,)   long
        responses      (resp_len,)       long   — response token ids only
        response_mask  (resp_len,)       float  — 1 where loss is computed

    Adds dummy old_log_probs / advantages required by verl's dp_actor.

    Args:
        files_arg: train_files / val_files value from config.
                   e.g. [/data/a.parquet, /data/b.parquet:1.5]
        tokenizer:  HuggingFace tokenizer.
        lazy:       If False (default), load all parquet data into RAM at init.
                    If True, read only row-counts at init; load files on first
                    access per DataLoader worker (good for very large datasets).
        seed:       RNG seed for sampling / shuffling.
    """

    def __init__(self, files_arg, tokenizer, config=None, lazy: bool = False, seed: int = 42):
        from sft.sft_data import process

        self._process = process
        self.tokenizer = _wrap_tokenizer(tokenizer)
        self.config = config

        file_ratios = parse_files(files_arg)
        rng = np.random.default_rng(seed)

        # Both paths use pq.ParquetFile.iter_batches(...) instead of
        # pq.read_table(...). The latter hits "Nested data conversions not
        # implemented for chunked array outputs" on pandas-written parquets
        # with list<struct> columns when per-row payloads are large
        # (e.g. multi-turn wildchat convs ≥10kB/row). RecordBatch.slice().to_pydict()
        # works because batches are single-chunk.

        if not lazy:
            self._lazy = False
            shards = []
            for path, ratio in file_ratios:
                batches, offsets = _load_parquet_batches(path)
                shards.append((batches, offsets, ratio))

            index_map = []
            for src_idx, (_, offsets, ratio) in enumerate(shards):
                n_rows = offsets[-1]
                n = int(n_rows * ratio)
                replace = ratio > 1.0
                idxs = rng.choice(n_rows, size=n, replace=replace)
                index_map.extend((src_idx, int(i)) for i in idxs)

            perm = rng.permutation(len(index_map))
            self._index_map = [index_map[i] for i in perm]
            self._shards = [(b, o) for b, o, _ in shards]

        else:
            self._lazy = True
            sources = []
            for path, ratio in file_ratios:
                n_rows = pq.read_metadata(path).num_rows
                sources.append({"path": path, "n_rows": n_rows, "ratio": ratio})

            index_map = []
            for src_idx, src in enumerate(sources):
                n = int(src["n_rows"] * src["ratio"])
                replace = src["ratio"] > 1.0
                idxs = rng.choice(src["n_rows"], size=n, replace=replace)
                index_map.extend((src_idx, int(i)) for i in idxs)

            perm = rng.permutation(len(index_map))
            self._index_map = [index_map[i] for i in perm]
            self._sources = sources
            self._shards = {}  # src_idx → (batches, offsets), populated per worker on first access

    def __len__(self):
        return len(self._index_map)

    def _get_row(self, idx: int) -> dict:
        src_idx, row_idx = self._index_map[idx]
        if not self._lazy:
            batches, offsets = self._shards[src_idx]
        else:
            if src_idx not in self._shards:
                self._shards[src_idx] = _load_parquet_batches(self._sources[src_idx]["path"])
            batches, offsets = self._shards[src_idx]
        return _slice_to_row(batches, offsets, row_idx)

    def __getitem__(self, idx: int) -> dict:
        row = self._get_row(idx)
        d = self._process(row, self.tokenizer, config=self.config)

        seq_len = d["input_ids"].shape[0]
        if "attention_mask" not in d:
            d["attention_mask"] = torch.ones(seq_len, dtype=torch.long)
        if "position_ids" not in d:
            d["position_ids"] = torch.arange(seq_len, dtype=torch.long)

        # Dummy tensors required by verl's dp_actor select_keys.
        # SFT loss (loss_mode=sft) ignores them.
        resp_len = d["responses"].shape[0]
        d["old_log_probs"] = torch.zeros(resp_len, dtype=torch.float32)
        d["advantages"] = torch.ones(resp_len, dtype=torch.float32)

        # Non-tensor field for per-source val loss breakdown.
        d["data_source"] = str(row.get("data_source", "unknown"))
        return d
