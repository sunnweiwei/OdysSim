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
Search-R1 hint agent — measures improvement from reflection hints.

Runs a fresh attempt using the hint-augmented prompt and returns the new reward,
with reward_delta tracked in extra_info.
"""

import logging

from agents.social_r1.agent import SYSTEM_PROMPT, _extract_gold_letter, _extract_option_letter
from agents.social_r1.hint import get_teacher_prompt
from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    Search-R1 hint agent: re-attempt with a hint-augmented prompt.

    Expected extra_info fields (same as search-r1 agent, plus):
      - hint: str — coaching hint generated from the prior rollout
      - old_reward: float — reward from the prior rollout
    """
    row = data["extra_info"]
    hint = row.get("hint", "")
    old_reward = float(row.get("old_reward", 0.0))

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

    predicted = remove_think(response)
    predicted = _extract_option_letter(predicted)

    correct = str(row.get("answer_letter") or "").strip().upper()
    if not correct:
        correct = _extract_gold_letter(str(row.get("answer_text") or ""))

    new_reward = 1.0 if (predicted is not None and predicted == correct) else 0.0
    reward_delta = new_reward - old_reward

    extra_info = {
        "social_r1-hint/reward": new_reward,
        "social_r1-hint/reward_delta": reward_delta,
        "social_r1-hint/delta_positive": int(reward_delta > 0),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    await process_post_chat(data, context, actor_agent.chat, output)
    return output
