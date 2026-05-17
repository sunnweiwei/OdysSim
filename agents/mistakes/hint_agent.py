"""
Mistakes hint agent — measures improvement from reflection hints.

Runs a fresh attempt using the hint-augmented prompt and returns the new reward,
with reward_delta tracked in extra_info.
"""

import logging

from agents.utils import Agent, process_post_chat, remove_think
from agents.mistakes.hint import get_teacher_prompt
from agents.mistakes.agent import _extract_option_letter, STUDENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    Mistakes hint agent: re-attempt with a hint-augmented prompt.

    Expected extra_info fields (same as mistakes agent, plus):
      - hint: str — coaching hint generated from the prior rollout
      - old_reward: float — reward from the prior rollout
    """
    row = data["extra_info"]
    hint = row.get("hint", "")
    old_reward = float(row.get("old_reward", 0.0))

    teacher_prompt = get_teacher_prompt(row, hint)
    chat = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": teacher_prompt},
    ]

    actor_agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await actor_agent.step()

    predicted = remove_think(response)
    predicted = _extract_option_letter(predicted)

    correct = str(row.get("TargetOption") or row.get("target_option") or "").strip().upper()
    new_reward = 1.0 if (predicted is not None and predicted == correct) else 0.0
    reward_delta = new_reward - old_reward

    extra_info = {
        "mistakes-hint/reward": new_reward,
        "mistakes-hint/reward_delta": reward_delta,
        "mistakes-hint/delta_positive": int(reward_delta > 0),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    return output