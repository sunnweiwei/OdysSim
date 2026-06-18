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
HUMANUAL hint agent — measures improvement from judge-derived hints.

Runs a fresh attempt using the hint-augmented prompt (with reference included)
and returns the new alignment score as the reward.
"""

import logging

from agents.humanual.agent import (
    RESPONSE_JUDGE_PROMPT,
    SIMULATION_USER_PROMPT,
    STATE_DESCRIPTIONS,
    JudgeOutput,
    _extract_field,
)
from agents.humanual.hint import get_teacher_system_prompt
from agents.utils import Agent, call_openai_parse

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    HUMANUAL hint agent: re-attempt with teacher prompt (judge coaching + reference).

    Expected extra_info fields (same as humanual agent, plus):
      - hint: str — coaching note from the prior judge output
      - old_reward: float — reward from the prior rollout
    """
    import json

    row = data.get("extra_info", data.get("row", {}))
    hint = row.get("hint", "")
    old_reward = float(row.get("old_reward", 0.0))

    prompt_field = row.get("prompt", "")
    if isinstance(prompt_field, str):
        try:
            prompt_parsed = json.loads(prompt_field)
            if isinstance(prompt_parsed, list):
                prompt_text = "\n".join(m.get("content", "") for m in prompt_parsed if isinstance(m, dict))
            else:
                prompt_text = prompt_field
        except (json.JSONDecodeError, TypeError):
            prompt_text = prompt_field
    else:
        prompt_text = str(prompt_field)

    persona = str(row.get("persona", ""))
    completion = str(row.get("completion", ""))
    dimension = "response"

    teacher_system = get_teacher_system_prompt(
        persona=persona,
        dimension=dimension,
        hint=hint,
        reference=completion,
    )
    from agents.utils import truncate_text_left

    user_prompt = SIMULATION_USER_PROMPT.format(prompt=truncate_text_left(prompt_text, 2000))
    chat = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": user_prompt},
    ]

    actor_agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)
    raw_generated = await actor_agent.step()
    generated = _extract_field(raw_generated, dimension) if raw_generated else ""

    judge_prompt_text = RESPONSE_JUDGE_PROMPT.format(
        state_desc=STATE_DESCRIPTIONS["response"],
        context=prompt_text,
        ground_truth=completion,
        generated=generated,
    )
    result = await call_openai_parse(
        [{"role": "user", "content": judge_prompt_text}],
        text_format=JudgeOutput,
        reasoning={"effort": "medium"},
    )

    if result is not None:
        new_reward = max(0.0, min(1.0, float(result["score"])))
    else:
        logger.warning("HUMANUAL hint judge failed, defaulting to 0.0")
        new_reward = 0.0

    reward_delta = new_reward - old_reward
    extra_info = {
        "humanual-hint/reward": new_reward,
        "humanual-hint/reward_delta": reward_delta,
        "humanual-hint/delta_positive": int(reward_delta > 0),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    return output
