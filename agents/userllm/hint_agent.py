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
userLLM hint agent — re-runs a rollout with a hint-augmented teacher prompt
and evaluates the result.

For each task this agent:
  1. Reads the prior rollout reward/sub_scores from data['extra_info']['rollout']
  2. Uses a pre-generated hint from data['extra_info']['hint']
  3. Builds a teacher prompt (base prompt + optional reference + hint coaching)
  4. Does a fresh rollout with the teacher prompt
  5. Evaluates the new output with the same metric as the main agent
  6. Returns the new reward as the training signal

This lets the training signal reflect improvement from the hint before using it
for context distillation (KL training).
"""

import logging
from typing import Any

from agents.userllm.agent import (
    _as_test_case,
    _extract_choice_texts,
    _intent_1gram_overlap_compatible,
    _maybe_ai_detector_score,
    _maybe_intent_adherence,
    _normalize_for_choice_match,
    _to_optional_bool,
)
from agents.userllm.helpers import _extract_intent
from agents.userllm.hint import generate_hint, get_teacher_prompt
from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)


async def agent_loop(data, context):
    """
    Hint agent: rollout with hint-augmented teacher prompt, then evaluate.

    Expected extra_info fields (same as userllm agent, plus):
      - hint:    str  — coaching brief from hint.generate_hint()
      - rollout: dict — extra_info dict from the prior main-agent rollout,
                        containing 'userllm/reward' and sub-score keys
    """
    row = data["extra_info"]
    source = str(row.get("source") or "").strip().lower()

    if source == "commonsense_qa":
        reward_metric = "role_adherence"
    elif source == "natural_questions":
        reward_metric = "intent_adherence"
    elif "prism" in source:
        reward_metric = "prism"
    else:
        reward_metric = None

    tc = _as_test_case(row)
    intent = _extract_intent(tc)

    hint = row.get("hint", "")
    rollout = row.get("rollout", {})
    old_reward = float(rollout.get("userllm/reward", 0.0))

    # If hint was not pre-generated (e.g. standalone use), generate it now
    if not hint:
        old_sub_scores = {
            "userllm/intent_decomposition": rollout.get("userllm/intent_decomposition"),
            "userllm/termination_f1": rollout.get("userllm/termination_f1"),
            "userllm/ai_detector_score": rollout.get("userllm/ai_detector_score"),
            "userllm/role_adherence": rollout.get("userllm/role_adherence"),
            "userllm/intent_adherence": rollout.get("userllm/intent_adherence"),
        }
        old_output = rollout.get("output", "")
        hint = await generate_hint(
            row=row,
            output=old_output,
            reward=old_reward,
            sub_scores=old_sub_scores,
        )

    # ------------------------------------------------------------------
    # Teacher rollout — same task but with hint-augmented prompt
    # ------------------------------------------------------------------
    prompt = get_teacher_prompt(row, hint)
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    actor_agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)

    response = await actor_agent.step()
    output_text = remove_think(response)

    pred_endconversation = "<|endconversation|>" in output_text
    if pred_endconversation:
        output_text = output_text.split("<|endconversation|>")[0].strip() + "<|endconversation|>"

    # ------------------------------------------------------------------
    # Evaluate new output with the same metric
    # ------------------------------------------------------------------
    sub_scores: dict[str, Any] = {
        "userllm-hint/intent_decomposition": None,
        "userllm-hint/termination_f1": None,
        "userllm-hint/ai_detector_score": None,
        "userllm-hint/role_adherence": None,
        "userllm-hint/intent_adherence": None,
    }

    if reward_metric == "prism":
        intent_decomp = _intent_1gram_overlap_compatible(intent, output_text) if intent else 0.0
        true_end = _to_optional_bool(row.get("is_last_turn"))
        term_score = 1.0 if (true_end is not None and pred_endconversation == true_end) else 0.0
        ai_score = await _maybe_ai_detector_score(row, output_text)
        ai_score = ai_score if ai_score is not None else 0.0
        new_reward = (intent_decomp + term_score + ai_score) / 3.0
        sub_scores.update(
            {
                "userllm-hint/intent_decomposition": intent_decomp,
                "userllm-hint/termination_f1": term_score,
                "userllm-hint/ai_detector_score": ai_score,
            }
        )

    elif reward_metric == "role_adherence":
        choices = _extract_choice_texts(row)
        role_reward = None
        if choices:
            out_norm = _normalize_for_choice_match(output_text)
            mentioned = sum(1 for c in choices if c and _normalize_for_choice_match(c) in out_norm)
            if mentioned != len(choices):
                attempt = 1 if mentioned in (1, 2) else 0
                role_reward = float(1 - attempt)
        new_reward = role_reward if role_reward is not None else 0.0
        sub_scores["userllm-hint/role_adherence"] = new_reward

    elif reward_metric == "intent_adherence":
        ia_reward = await _maybe_intent_adherence(row, output_text)
        new_reward = ia_reward if ia_reward is not None else 0.0
        sub_scores["userllm-hint/intent_adherence"] = new_reward

    else:
        new_reward = 0.0

    reward_delta = new_reward - old_reward

    # Per-dimension deltas for all reward types
    dim_deltas: dict[str, Any] = {}
    if reward_metric == "prism":
        old_intent_decomp = rollout.get("userllm/intent_decomposition")
        old_term = rollout.get("userllm/termination_f1")
        old_ai = rollout.get("userllm/ai_detector_score")
        if old_intent_decomp is not None:
            dim_deltas["userllm-hint/intent_decomposition_delta"] = (
                sub_scores["userllm-hint/intent_decomposition"] or 0.0
            ) - float(old_intent_decomp)
            dim_deltas["userllm-hint/termination_f1_delta"] = (
                sub_scores["userllm-hint/termination_f1"] or 0.0
            ) - float(old_term or 0.0)
            dim_deltas["userllm-hint/ai_detector_score_delta"] = (
                sub_scores["userllm-hint/ai_detector_score"] or 0.0
            ) - float(old_ai or 0.0)
    elif reward_metric == "role_adherence":
        old_role = rollout.get("userllm/role_adherence")
        if old_role is not None:
            dim_deltas["userllm-hint/role_adherence_delta"] = new_reward - float(old_role)
    elif reward_metric == "intent_adherence":
        old_ia = rollout.get("userllm/intent_adherence")
        if old_ia is not None:
            dim_deltas["userllm-hint/intent_adherence_delta"] = new_reward - float(old_ia)

    extra_info = {
        "userllm-hint/reward": new_reward,
        "userllm-hint/reward_delta": reward_delta,
        "userllm-hint/delta_positive": int(reward_delta > 0),
        **sub_scores,
        **dim_deltas,
    }

    output_obj = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    await process_post_chat(data, context, actor_agent.chat, output_obj)
    return output_obj
