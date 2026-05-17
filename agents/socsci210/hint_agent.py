"""
SocSci210 hint agent — measures improvement from reflection hints.

Runs a fresh attempt using the hint-augmented prompt and returns the new reward,
with reward_delta tracked in extra_info.
"""

import logging

from agents.utils import Agent, process_post_chat, remove_think
from agents.socsci210.hint import get_teacher_prompt
from agents.socsci210.agent import (
    SYSTEM_PROMPT,
    _compute_reward,
    _extract_prediction,
)

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    SocSci210 hint agent: re-attempt with a hint-augmented prompt.

    Expected extra_info fields (same as socsci210 agent, plus):
      - hint: str — coaching hint generated from the prior rollout
      - old_reward: float — reward from the prior rollout
    """
    row = data["extra_info"]
    hint = row.get("hint", "")
    old_reward = float(row.get("old_reward", 0.0))

    response_type = str(row.get("response_type") or "ordinal").strip().lower()
    r_min = float(row.get("r_min", 0))
    r_max = float(row.get("r_max", 1 if response_type == "binary" else 7))
    try:
        gold = float(row["response"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("SocSci210 hint row is missing a numeric 'response' field.")

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

    content = remove_think(response)
    pred = _extract_prediction(content, response_type)
    new_reward, is_exact = _compute_reward(pred, gold, r_min, r_max, response_type)
    reward_delta = new_reward - old_reward

    extra_info = {
        "socsci210-hint/reward": new_reward,
        "socsci210-hint/reward_delta": reward_delta,
        "socsci210-hint/delta_positive": int(reward_delta > 0),
        "socsci210-hint/is_exact_match": float(is_exact),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    await process_post_chat(data, context, actor_agent.chat, output)
    return output
