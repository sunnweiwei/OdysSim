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

from __future__ import annotations

import asyncio
from typing import Any, Literal, Optional

from pydantic import BaseModel

from agents.sim_arena.math_prompts import (
    ASSISTANT_FIRST_TURN_USER_TEMPLATE,
    ASSISTANT_SYSTEM_PROMPT,
    EVAL_SIMULATOR_ALL_ATTRIBUTES_FULFILLMENT_PROMPT_TEMPLATE,
    EVAL_SIMULATOR_INTERACTION_STYLE_LIKERT_PROMPT_TEMPLATE,
    EVAL_SIMULATOR_WRITING_STYLE_LIKERT_PROMPT_TEMPLATE,
    MATH_SIMULATOR_INITIAL_USER_MESSAGE_TEMPLATE,
    MATH_SIMULATOR_INTERACTION_STYLE_FEATURES_TEXT,
    MATH_SIMULATOR_SYSTEM_PROMPT,
    MATH_SIMULATOR_WRITING_STYLE_FEATURES_TEXT,
)
from agents.utils import Agent, call_openai, call_openai_parse, get_judge_model, get_judge_reasoning, process_post_chat

# --- Structured output schemas ---


class _LikertResult(BaseModel):
    key_differences: list[str]
    similarity_score: float


class _FeatureResult(BaseModel):
    feature_name: str
    analysis: str
    classification: Literal["Match", "No Match"]


class _FulfillmentResult(BaseModel):
    results: list[_FeatureResult]


def _unwrap_raw_data(raw_data: dict[str, Any]) -> dict[str, Any]:
    import json as _json

    if isinstance(raw_data, dict) and isinstance(raw_data.get("extra_info"), dict):
        raw_data = raw_data["extra_info"]
    if isinstance(raw_data.get("raw"), str):
        raw_data = {**raw_data, **_json.loads(raw_data["raw"])}
    return raw_data


def _public_chat(assistant_chat: list[dict], first_query: Optional[str] = None) -> list[dict]:
    turn, out = 1, []
    for msg in assistant_chat:
        role = msg.get("role")
        if role == "system":
            continue
        content = first_query if role == "user" and turn == 1 and first_query else msg.get("content", "")
        out.append({"role": role, "content": content})
        if role == "assistant":
            turn += 1
    return out


def _fmt_conversation(messages: list[dict]) -> str:
    return "\n".join(f"- {'Student' if m['role'] == 'user' else 'AI Tutor'}: {m['content']}" for m in messages)


def _mean(values: list[Optional[float]]) -> float:
    return sum(v or 0.0 for v in values) / len(values) if values else 0.0


def _norm(x: Optional[float], max_v: float) -> Optional[float]:
    return max(0.0, min(1.0, x / max_v)) if x is not None else None


async def _none():
    return None


async def agent_loop(raw_data, context):
    data = _unwrap_raw_data(raw_data)

    assert "problem" in data, f"Missing 'problem' in data keys: {list(data.keys())}"
    assert "human_user_queries" in data, f"Missing 'human_user_queries' in data keys: {list(data.keys())}"
    assert "human_public_conversation" in data, f"Missing 'human_public_conversation' in data keys: {list(data.keys())}"

    problem = data["problem"]
    human_user_queries = data.get("human_user_queries")
    human_public_conversation = data.get("human_public_conversation")
    target_interaction_style_features = data.get("target_interaction_style_features", [])

    # --- Rollout ---
    sim_system = MATH_SIMULATOR_SYSTEM_PROMPT.format(user_profile=data.get("user_profile_text", ""))
    sim_init = MATH_SIMULATOR_INITIAL_USER_MESSAGE_TEMPLATE.format(math_problem=problem).strip()

    assistant_chat = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    user_agent = Agent(
        context.llm_client,
        [{"role": "system", "content": sim_system}, {"role": "user", "content": sim_init}],
        context.tokenizer,
        context.config,
        prompt_turn=2,
    )
    first_message = None
    termination_reason = "max_turns_reached"

    for turn in range(8):
        user_raw = await user_agent.step()
        if not user_raw:
            termination_reason = "simulator_no_output"
            break
        query = next(
            (user_raw.split(tag, 1)[1].strip() for tag in ("Response:", "Query:") if tag in user_raw),
            user_raw,
        )
        if not query or "terminate conversation" in user_raw.lower():
            termination_reason = "simulator_terminated"
            break

        if first_message is None:
            first_message = query
            assistant_chat.append(
                {
                    "role": "user",
                    "content": ASSISTANT_FIRST_TURN_USER_TEMPLATE.format(
                        problem=problem.strip(),
                        query=query,
                    ),
                }
            )
        else:
            assistant_chat.append({"role": "user", "content": query})

        response = await call_openai(assistant_chat, model="gpt-5-nano", reasoning_effort="minimal")
        if not response:
            termination_reason = "assistant_no_output"  # noqa: F841
            break
        assistant_chat.append({"role": "assistant", "content": response})
        user_agent.append({"role": "user", "content": response})

    num_turns = sum(1 for m in assistant_chat if m["role"] == "assistant")
    public_conversation = _public_chat(assistant_chat, first_message)
    conversation_text = _fmt_conversation(public_conversation)
    # --- Eval (parallel) ---

    real_user_queries_text = (
        "\n".join(human_user_queries) if isinstance(human_user_queries, list) else human_user_queries
    )
    real_conversation_text = (
        _fmt_conversation(human_public_conversation)
        if isinstance(human_public_conversation, list)
        else human_public_conversation
    )
    sim_user_queries_text = "\n".join(m["content"] for m in public_conversation if m["role"] == "user")

    all_features = [
        {"name": f.get("Feature Name", ""), "desc": f.get("Feature Question Answer") or f.get("Feature Question", "")}
        for f in target_interaction_style_features
        if f.get("Feature Name") and (f.get("Feature Question Answer") or f.get("Feature Question"))
    ]
    features_text = "\n".join(f"{i + 1}. {f['name']}: {f['desc']}" for i, f in enumerate(all_features))

    writing_result, interaction_likert_result, fulfillment_result = await asyncio.gather(
        call_openai_parse(
            [
                {
                    "role": "user",
                    "content": EVAL_SIMULATOR_WRITING_STYLE_LIKERT_PROMPT_TEMPLATE.format(
                        real_user_queries=real_user_queries_text,
                        simulated_queries=sim_user_queries_text,
                        features=MATH_SIMULATOR_WRITING_STYLE_FEATURES_TEXT,
                    ),
                }
            ],
            _LikertResult,
            model=get_judge_model("gpt-5-nano"),
            reasoning_effort=get_judge_reasoning("minimal"),
        ),
        call_openai_parse(
            [
                {
                    "role": "user",
                    "content": EVAL_SIMULATOR_INTERACTION_STYLE_LIKERT_PROMPT_TEMPLATE.format(
                        real_conversation=real_conversation_text,
                        simulated_conversation=conversation_text,
                        features=MATH_SIMULATOR_INTERACTION_STYLE_FEATURES_TEXT,
                    ),
                }
            ],
            _LikertResult,
            model=get_judge_model("gpt-5-nano"),
            reasoning_effort=get_judge_reasoning("minimal"),
        ),
        call_openai_parse(
            [
                {
                    "role": "user",
                    "content": EVAL_SIMULATOR_ALL_ATTRIBUTES_FULFILLMENT_PROMPT_TEMPLATE.format(
                        conversation_text=conversation_text,
                        features_text=features_text,
                    ),
                }
            ],
            _FulfillmentResult,
            model=get_judge_model("gpt-5-nano"),
            reasoning_effort=get_judge_reasoning("minimal"),
        )
        if all_features
        else _none(),
    )

    # --- Parse results ---
    writing_style_score = writing_result["similarity_score"] if writing_result else None
    interaction_style_score = interaction_likert_result["similarity_score"] if interaction_likert_result else None

    fulfillment_scores = []
    for res in fulfillment_result["results"] if fulfillment_result else []:
        fulfillment_scores.append(1 if res["classification"] == "Match" else 0)
    fulfillment_rate = _mean(fulfillment_scores) if fulfillment_scores else None

    # --- Reward ---
    reward = (
        _mean(
            [
                _norm(writing_style_score, 5),
                _norm(interaction_style_score, 5),
                fulfillment_rate,
            ]
        )
        or 0.0
    )

    extra_info = {
        "sim_arena_math/reward": reward,
        "sim_arena_math/num_turn": num_turns,
        "sim_arena_math/writing_style_likert": writing_style_score,
        "sim_arena_math/interaction_style_likert": interaction_style_score,
        "sim_arena_math/fulfillment_rate": fulfillment_rate,
        "all/score": reward,
    }
    output = await user_agent.get_agent_output(reward, extra_info=extra_info)
    await process_post_chat(raw_data, context, user_agent.chat, output, extra=extra_info)
    return output
