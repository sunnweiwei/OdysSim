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

"""
HugAgent agent for Harmony evaluation.

Individual-level belief reasoning on the HugAgent benchmark
(Li et al., "HugAgent: A Human Simulation Benchmark for Individual-Level
Reasoning", EMNLP 2026, arXiv:2510.15144). Data: Hugging Face dataset
`social-atoms/hugagent` (gated, CC BY-NC 4.0), 54 participants, 1,742 items,
three domains (healthcare, surveillance, zoning).

Two single-turn tasks, routed by `task_type` on the row:
  belief_attribution (belief state inference, data_source `hugagent_bsi`):
    given a participant's demographics and their own interview answers, pick
    which of two options states the belief the participant holds. Gold is a
    letter; a few items accept either letter ("A/B").
    reward = exact match (0 / 1)
  belief_update (belief dynamics update, data_source `hugagent_bdu`):
    predict the participant's own rating after a scenario, on a 1-10 stance
    scale or a 1-5 reason-weight scale.
    reward = tolerance accuracy as in the paper: 1 if |pred - gold| <= 1 on a
    5-point scale or <= 2 on a 10-point scale, else 0. The normalized error
    1 - |pred - gold| / span is logged as hugagent/norm_acc.

The released items are evaluation only (every row carries split="test"); the
agent raises if asked to run such a row in training mode.

Expected fields in data["extra_info"] (one HugAgent item, see the dataset card):
    task_type, topic, demographics (dict), context_qas (list of {question, answer}),
    task_question, answer_options (dict, BSI), answer (BSI),
    question_type, scale ([lo, hi]), reason_text, user_answer (BDU).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agents.utils import Agent, process_post_chat, remove_think

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
STRICT_INT_RE = re.compile(r"^-?\d+$")

BSI_SYSTEM_PROMPT = (
    "You are an expert psychologist specializing in Theory of Mind and belief attribution. "
    "Your task: analyze conversation transcripts to infer what the participant believes about causal relationships. "
    "Focus on understanding their mental model - what they think causes what, not what is objectively true. "
    "Consider their background, conversation patterns, and implicit beliefs expressed through their responses. "
    "Base your inference strictly on evidence from their statements, not general assumptions. "
    "You may reason briefly, then output the final option letter in <answer>X</answer> tags."
)
BDU_SYSTEM_PROMPT = (
    "You are an expert in survey research and human psychology. "
    "Your task: predict how this person would respond to a specific survey question based on their background and conversation. "
    "Consider their demographics, expressed opinions, and conversation patterns. "
    "Focus on understanding their likely response pattern, not what would be objectively correct. "
    "Base your prediction on evidence from their profile and statements. "
    "You may reason briefly, then output the final number in <answer>X</answer> tags."
)


def _demographics_block(demo: Any) -> str:
    if isinstance(demo, dict) and demo:
        lines = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in demo.items()]
        return "Person's Background:\n" + "\n".join(lines)
    if isinstance(demo, str) and demo.strip():
        return "Person's Background:\n" + demo.strip()
    return ""


def _context_block(qas: Any, title: str) -> str:
    if not qas:
        return ""
    parts = [f"Q{i}: {qa.get('question', '')}\nA{i}: {qa.get('answer', '')}" for i, qa in enumerate(qas, 1)]
    return title + "\n" + "\n\n".join(parts)


def build_prompt(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return (task, system_prompt, user_prompt). Prompt text follows the paper (Appendix K.1)."""
    task = str(row.get("task_type") or "belief_attribution")
    parts = [p for p in (_demographics_block(row.get("demographics")),) if p]
    if task == "belief_attribution":
        ctx = _context_block(row.get("context_qas"), "Conversation History:")
        if ctx:
            parts.append(ctx)
        parts.append(f"Task: {row.get('task_question', '')}")
        options = row.get("answer_options") or {}
        keys = list(options.keys())
        parts.append("Answer options:\n" + "\n".join(f"{k}) {options[k]}" for k in keys))
        parts.append(
            f"Based on the evidence above, choose the single letter ({', '.join(keys)}) that best represents "
            "this person's belief, and output it as <answer>LETTER</answer>."
        )
        return task, BSI_SYSTEM_PROMPT, "\n\n".join(parts)

    ctx = _context_block(row.get("context_qas"), "Previous Conversation:")
    if ctx:
        parts.append(ctx)
    lo, hi = _scale(row)
    parts.append(f"Survey Question: {row.get('task_question', '')}")
    if str(row.get("question_type") or "") == "reason_evaluation":
        parts.append(f"Context: This asks about the influence of: {row.get('reason_text', '')}")
        parts.append(f"Scale: {lo} to {hi} (1=no influence, {hi}=very strong influence)")
        parts.append(
            "Based on this person's profile and conversation, what rating would they likely give? "
            "Output only the number as <answer>N</answer>."
        )
    else:
        parts.append(f"Scale: {lo} to {hi}")
        parts.append(
            "Based on this person's profile and conversation, what number would they likely choose? "
            "Output only the number as <answer>N</answer>."
        )
    return task, BDU_SYSTEM_PROMPT, "\n\n".join(parts)


def _scale(row: dict[str, Any]) -> tuple[int, int]:
    scale = row.get("scale") or [1, 10]
    try:
        return int(scale[0]), int(scale[1])
    except (TypeError, ValueError, IndexError):
        return 1, 10


def extract_answer(text: str, task: str, option_keys: Optional[list[str]] = None) -> Optional[Any]:
    """Strict extraction: only the contents of <answer>...</answer> count."""
    if not text:
        return None
    m = ANSWER_TAG_RE.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    if task == "belief_attribution":
        letter = inner.strip(" .)").upper()
        keys = [k.upper() for k in (option_keys or ["A", "B"])]
        return letter if letter in keys else None
    if STRICT_INT_RE.match(inner):
        return int(inner)
    return None


def compute_reward(pred: Optional[Any], row: dict[str, Any], task: str) -> tuple[float, float]:
    """Return (reward, norm_acc). reward is the paper's metric; norm_acc is 1 - |err| / span (BDU) or reward (BSI)."""
    if pred is None:
        return 0.0, 0.0
    if task == "belief_attribution":
        accepted = [a.strip().upper() for a in str(row.get("answer", "")).split("/")]
        hit = 1.0 if str(pred).upper() in accepted else 0.0
        return hit, hit
    gold = row.get("user_answer")
    if gold is None:
        return 0.0, 0.0
    lo, hi = _scale(row)
    span = max(1, hi - lo)
    tol = 1 if span <= 5 else 2
    err = abs(int(pred) - int(gold))
    reward = 1.0 if err <= tol else 0.0
    norm_acc = max(0.0, min(1.0, 1.0 - err / span))
    return reward, norm_acc


async def agent_loop(data: dict, context):
    """HugAgent single-turn prediction (belief state inference or belief dynamics update)."""
    row = data["extra_info"]
    if getattr(context, "is_train", False) and str(row.get("split", "")).lower() == "test":
        # The released HugAgent items are a test set only. Refuse to use them as RL training data.
        raise ValueError("HugAgent rows with split='test' are evaluation only and must not be used for training.")
    if row.get("index") is None:
        # The published parquets carry item_id, but no index. Also support
        # callers that invoke the agent directly without RLHFDataset.
        row = {**row, "index": row["item_id"]}
        data = {**data, "extra_info": row}
    task, system_prompt, user_prompt = build_prompt(row)
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    agent = Agent(
        context.llm_client,
        chat,
        context.tokenizer,
        context.config,
        prompt_turn=2,
        enable_think=True,
    )
    response = await agent.step()
    content = remove_think(response)
    option_keys = list((row.get("answer_options") or {}).keys()) or None
    pred = extract_answer(content, task, option_keys)
    reward, norm_acc = compute_reward(pred, row, task)
    short = "bsi" if task == "belief_attribution" else "bdu"
    output = await agent.get_agent_output(
        reward,
        extra_info={
            "all/score": reward,
            f"hugagent_{short}/reward": reward,
            f"hugagent_{short}/norm_acc": norm_acc,
            f"hugagent_{short}/parsed": 1.0 if pred is not None else 0.0,
            "hugagent/response_length": len(response.split()) if response else 0,
        },
    )
    await process_post_chat(data, context, agent.chat, output, extra=None)
    return output
