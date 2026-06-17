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
FanToM hint agent — measures improvement from reflection hints.

Runs a fresh attempt using the hint-augmented prompt and returns the new reward,
with reward_delta tracked in extra_info.
"""

import logging

from agents.fantom.agent import extract_answer, score_response
from agents.fantom.hint import get_teacher_prompt
from agents.utils import Agent, remove_think

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    FanToM hint agent: re-attempt the question with a hint-augmented prompt.

    Expected extra_info fields (same as fantom agent, plus):
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

    question_type = str(row.get("question_type", ""))
    correct_answer = row.get("correct_answer", "")
    wrong_answer = row.get("wrong_answer", "")

    new_reward = score_response(question_type, correct_answer, wrong_answer, predicted)
    reward_delta = new_reward - old_reward

    extra_info = {
        "fantom-hint/reward": new_reward,
        "fantom-hint/reward_delta": reward_delta,
        "fantom-hint/delta_positive": int(reward_delta > 0),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    return output
