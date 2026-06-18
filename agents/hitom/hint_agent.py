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
HiToM hint agent — measures improvement from reflection hints.

Runs a fresh attempt using the hint-augmented prompt and returns the new reward,
with reward_delta tracked in extra_info.
"""

import logging

from agents.hitom.agent import evaluate_answer, extract_answer
from agents.hitom.hint import get_teacher_prompt
from agents.utils import Agent, remove_think

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    HiToM hint agent: re-attempt the question with a hint-augmented prompt.

    Expected extra_info fields (same as hitom agent, plus):
      - hint: str — coaching hint generated from the prior rollout
      - old_reward: float — reward from the prior rollout
    """
    row = data["extra_info"]
    hint = row.get("hint", "")
    old_reward = float(row.get("old_reward", 0.0))

    teacher_prompt = get_teacher_prompt(row, hint)
    chat = [
        {"role": "system", "content": ""},
        {"role": "user", "content": teacher_prompt},
    ]

    actor_agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await actor_agent.step()

    predicted = remove_think(response)
    predicted = extract_answer(predicted)

    correct_answer = str(row.get("correct_answer", ""))
    is_correct = evaluate_answer(predicted, row.get("choices", ""), correct_answer)
    new_reward = 1.0 if is_correct else 0.0
    reward_delta = new_reward - old_reward

    extra_info = {
        "hitom-hint/reward": new_reward,
        "hitom-hint/reward_delta": reward_delta,
        "hitom-hint/delta_positive": int(reward_delta > 0),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    return output
