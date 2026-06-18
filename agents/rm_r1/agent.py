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

"""Harmony agent loop for RM-R1 pairwise reward-model training and eval.

Prompts and reward semantics mirror https://github.com/RM-R1-UIUC/RM-R1.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.rm_r1.prompts import (
    RM_R1_INSTRUCT_SYSTEM_PROMPT,
    RM_R1_MULTI_TURN_INSTRUCT_USER_PROMPT,
    RM_R1_MULTI_TURN_REASONING_USER_PROMPT,
    RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT,
    RM_R1_SINGLE_TURN_REASONING_USER_PROMPT,
)
from agents.utils import Agent, process_post_chat

MODE_INSTRUCT = "instruct"
MODE_REASONING = "reasoning"
RLVR_SUITES = {"rlvr", "train", "rm_r1_rlvr", "reasoning_rlvr", "after_distill_rlvr"}

_MODE_ALIASES = {
    "reasoning": MODE_REASONING,
    "rm-r1-reasoning": MODE_REASONING,
    "deepseek": MODE_REASONING,
    "deepseek-distilled": MODE_REASONING,
    "instruct": MODE_INSTRUCT,
    "rm-r1-instruct": MODE_INSTRUCT,
    "after-distill": MODE_INSTRUCT,
    "after-distill-rlvr": MODE_INSTRUCT,
}

_WINNER_A = {"a", "model_a", "model-a", "answer_a", "response_a", "chatbot_a"}
_WINNER_B = {"b", "model_b", "model-b", "answer_b", "response_b", "chatbot_b"}


def normalize_mode(mode: Any) -> str:
    key = str(mode or "").strip().lower().replace("_", "-")
    return _MODE_ALIASES.get(key, MODE_INSTRUCT)


def _first(row: dict, *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def winner_to_letter(winner: Any) -> Optional[str]:
    t = str(winner or "").strip().lower()
    if t in _WINNER_A:
        return "A"
    if t in _WINNER_B:
        return "B"
    return None


def letter_to_winner(letter: Any) -> str:
    t = str(letter or "").strip().upper()
    if t == "A":
        return "model_a"
    if t == "B":
        return "model_b"
    raise ValueError(f"Unknown RM-R1 answer letter: {letter!r}")


def build_pairwise_messages(row: dict, mode: str) -> list[dict[str, str]]:
    ctx = row.get("context_messages")
    if isinstance(ctx, list) and ctx and not row.get("force_rebuild_prompt"):
        return [
            {"role": str(m["role"]), "content": str(m.get("content", ""))}
            for m in ctx
            if isinstance(m, dict) and m.get("role")
        ]

    if row.get("multi_turn") or row.get("conversation_1"):
        c1 = _first(row, "conversation_1", "conversation1", "conversation_a")
        c2 = _first(row, "conversation_2", "conversation2", "conversation_b")
        if not c1 or not c2:
            raise ValueError("RM-R1 multi-turn row missing conversation_1/conversation_2.")
        if mode == MODE_REASONING:
            user = RM_R1_MULTI_TURN_REASONING_USER_PROMPT.format(conversation_1=c1, conversation_2=c2)
            return [{"role": "user", "content": user}]
        user = RM_R1_MULTI_TURN_INSTRUCT_USER_PROMPT.format(conversation_1=c1, conversation_2=c2)
        return [
            {"role": "system", "content": RM_R1_INSTRUCT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    question = _first(row, "question", "prompt_text", "prompt", "instruction")
    answer_a = _first(row, "answer_a", "response_a", "output_a", "assistant_response_a")
    answer_b = _first(row, "answer_b", "response_b", "output_b", "assistant_response_b")
    if not question or not answer_a or not answer_b:
        raise ValueError("RM-R1 row missing question/answer_a/answer_b.")
    if mode == MODE_REASONING:
        user = RM_R1_SINGLE_TURN_REASONING_USER_PROMPT.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        return [{"role": "user", "content": user}]
    user = RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    return [
        {"role": "system", "content": RM_R1_INSTRUCT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def process_judgement(judgment: str, *, strong_error: bool = False) -> str:
    has_a = "[[A]]" in judgment
    has_b = "[[B]]" in judgment
    if has_a and has_b:
        return "strong_error" if strong_error else "error"
    if has_a:
        return "A"
    if has_b:
        return "B"
    return "error"


def answer_reward(solution_str: str, answer: str) -> float:
    """RM-R1 RLVR reward: inspect only the last 80 chars."""
    pred = (solution_str or "")[-80:]
    if answer not in {"model_a", "model_b"}:
        raise NotImplementedError("Check your dataset label!")
    has_a = "<answer>[[A]]</answer>" in pred
    has_b = "<answer>[[B]]</answer>" in pred
    if answer == "model_a" and has_a and not has_b:
        return 1.0
    if answer == "model_b" and has_b and not has_a:
        return 1.0
    return -1.0


def evaluation_reward(judgment: str, winner: Any, *, strong_error: bool = False) -> float:
    predicted = process_judgement(judgment, strong_error=strong_error)
    expected = winner_to_letter(winner)
    if predicted in {"A", "B"} and expected and predicted == expected:
        return 1.0
    return 0.0


async def agent_loop(data: dict, context):
    row = dict(data["extra_info"])
    mode = normalize_mode(row.get("rm_r1_mode"))
    chat = build_pairwise_messages(row, mode)

    enable_think = row.get("enable_think")
    if enable_think is None:
        enable_think = mode == MODE_REASONING
    agent = Agent(
        context.llm_client,
        chat,
        context.tokenizer,
        context.config,
        prompt_turn=len(chat),
        enable_think=bool(enable_think),
    )
    response = await agent.step() or ""

    eval_suite = str(row.get("eval_suite") or "rlvr").lower()
    expected_winner = _first(row, "winner", "answer", "ground_truth")
    if not expected_winner:
        letter = _first(row, "correct_letter", "gold_letter")
        if letter:
            expected_winner = letter_to_winner(letter)
    if not expected_winner:
        raise ValueError("RM-R1 row missing winner/answer/ground_truth.")

    if eval_suite in RLVR_SUITES or row.get("reward_mode") == "rlvr":
        letter = winner_to_letter(expected_winner)
        normalized = letter_to_winner(letter) if letter else expected_winner
        reward = answer_reward(response, normalized)
    else:
        reward = evaluation_reward(response, expected_winner, strong_error=eval_suite.startswith("rmb"))

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "rm_r1/reward": reward,
            "rm_r1/response_length": len(response.split()),
        },
    )
    await process_post_chat(data, context, agent.chat, output)
    return output
