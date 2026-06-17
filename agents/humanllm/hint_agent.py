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
HumanLLM hint agent — measures improvement from reflection hints.

Runs a fresh attempt using the hint-augmented prompt and returns the new reward,
with reward_delta tracked in extra_info. Mirrors social_r1 / socsci210 pattern.
"""

import logging

from agents.humanllm.agent import (
    SYSTEM_PROMPT,
    _predict_index,
)
from agents.humanllm.hint import get_teacher_prompt
from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    HumanLLM hint agent: re-attempt with a hint-augmented prompt.

    Expected extra_info fields (same as humanllm agent, plus):
      - hint: str — coaching hint generated from the prior rollout
      - old_reward: float — reward from the prior rollout
    """
    row = data["extra_info"]
    hint = row.get("hint", "")
    old_reward = float(row.get("old_reward", 0.0))

    candidates = list(row.get("candidates") or [])
    if not candidates:
        raise ValueError("HumanLLM hint row is missing 'candidates'.")

    gold_index = int(row.get("answer_index", -1))
    if gold_index < 0 or gold_index >= len(candidates):
        raise ValueError(f"HumanLLM hint row has invalid answer_index={gold_index}.")

    teacher_prompt = get_teacher_prompt(row, hint)
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": teacher_prompt},
    ]

    actor_agent = Agent(
        context.llm_client,
        chat,
        context.tokenizer,
        context.config,
        prompt_turn=2,
        enable_think=True,
    )
    response = await actor_agent.step()

    content = remove_think(response or "")
    predicted = _predict_index(content, candidates)
    has_prediction = predicted is not None
    new_reward = 1.0 if (has_prediction and predicted == gold_index) else 0.0
    reward_delta = new_reward - old_reward

    extra_info = {
        "humanllm-hint/reward": new_reward,
        "humanllm-hint/reward_delta": reward_delta,
        "humanllm-hint/delta_positive": int(reward_delta > 0),
        "humanllm-hint/has_prediction": int(has_prediction),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    await process_post_chat(data, context, actor_agent.chat, output)
    return output
