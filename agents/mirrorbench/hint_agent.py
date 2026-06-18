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
MirrorBench hint agent — measures improvement from judge-derived hints.

Runs a fresh simulation using the hint-augmented user-proxy prompt
(with real conversation as style reference) and returns the new reward.
"""

import logging

from agents.mirrorbench.agent import (
    build_assistant_mirror_system_prompt,
    format_conversation,
    run_gteval,
)
from agents.mirrorbench.hint import get_teacher_system_prompt
from agents.utils import Agent, call_openai, split_think, truncate_turns_for_reference

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    MirrorBench hint agent: re-run simulation with hint-augmented user-proxy prompt.

    Expected extra_info fields (same as mirrorbench agent, plus):
      - hint: str — coaching note from the prior GTEval judge output
      - old_reward: float — reward from the prior rollout
    """
    info = data["extra_info"]
    hint = info.get("hint", "")
    old_reward = float(info.get("old_reward", 0.0))

    task_description = info.get("task_description", "")
    real_turns = info.get("turns", [])
    few_shot_examples = info.get("few_shot_user_examples") or None
    metadata = {"few_shot_user_examples": few_shot_examples}
    max_turns = None

    # Only user turns — the model is simulating the user, not the assistant
    user_turns_only = [t for t in real_turns if t.get("role") == "user"]
    real_conversation_str = format_conversation(truncate_turns_for_reference(user_turns_only))

    teacher_system = get_teacher_system_prompt(
        task_description=task_description,
        domain=metadata.get("domain"),
        persona=metadata.get("persona"),
        hint=hint,
        real_conversation_str=real_conversation_str,
    )
    assistant_system_prompt = build_assistant_mirror_system_prompt(
        real_conversation=real_conversation_str,
    )

    chat = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": "Generate the next user message."},
    ]
    user_agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)

    effective_max_turns = min(max_turns or len(real_turns), 10)
    turns_to_simulate = real_turns[:effective_max_turns]
    if turns_to_simulate and turns_to_simulate[-1].get("role") == "assistant":
        turns_to_simulate = turns_to_simulate[:-1]

    simulation_turns = []
    for turn in turns_to_simulate:
        role = turn.get("role")
        if role == "user":
            response = await user_agent.step()
            if response:
                response = response.strip()
                if response.lower().startswith("user:"):
                    response = response[5:].strip()
                simulation_turns.append({"role": "user", "content": split_think(response)[1]})
            if not response:
                break
        elif role == "assistant":
            assistant_messages = [{"role": "system", "content": assistant_system_prompt}] + simulation_turns
            response = await call_openai(assistant_messages, model="gpt-5.4-nano", reasoning_effort="none")
            if response:
                simulation_turns.append({"role": "assistant", "content": response})

    real_conv_str = format_conversation(turns_to_simulate)
    proxy_conv_str = format_conversation(simulation_turns)
    new_reward, _ = await run_gteval(real_conv_str, proxy_conv_str, client=None)
    reward_delta = new_reward - old_reward

    extra_info = {
        "mirrorbench-hint/reward": new_reward,
        "mirrorbench-hint/reward_delta": reward_delta,
        "mirrorbench-hint/delta_positive": int(reward_delta > 0),
    }

    output = await user_agent.get_agent_output(new_reward, extra_info=extra_info)
    return output
