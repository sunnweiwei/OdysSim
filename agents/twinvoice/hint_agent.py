"""
TwinVoice hint agent — measures improvement from persona-style hints.

Runs a fresh attempt using the hint-augmented prompt and returns the new reward.
"""

import logging

from agents.utils import Agent, remove_think
from agents.twinvoice.hint import get_teacher_prompt
from agents.twinvoice.agent import extract_choice

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    TwinVoice hint agent: re-attempt the MCQ with a hint-augmented prompt.

    Expected extra_info fields (same as twinvoice agent, plus):
      - hint: str — coaching hint generated from the prior rollout
      - old_reward: float — reward from the prior rollout (0.0 or 1.0)
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

    content = remove_think(response)
    predicted = extract_choice(content)

    correct_idx = row["answer_idx"]
    correct_letter = "ABCD"[correct_idx]
    is_correct = predicted == correct_letter if predicted else False
    new_reward = 1.0 if is_correct else 0.0
    reward_delta = new_reward - old_reward

    extra_info = {
        "twinvoice-hint/reward": new_reward,
        "twinvoice-hint/reward_delta": reward_delta,
        "twinvoice-hint/delta_positive": int(reward_delta > 0),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    return output