"""
LifeChoices hint agent — measures improvement from reflection hints.

For each task, this agent:
  1. Receives a hint generated from the prior (incorrect) rollout
  2. Runs a fresh attempt using the hint-augmented prompt
  3. Evaluates the new answer against the ground truth
  4. Returns reward = new score (1.0/0.0), with reward_delta tracked in extra_info
"""

import logging

from agents.utils import Agent, process_post_chat, remove_think
from agents.lifechoices.hint import get_teacher_prompt
from agents.lifechoices.agent import extract_choice

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    LifeChoices hint agent: re-attempt the MCQ with a hint-augmented prompt.

    Expected extra_info fields (same as lifechoices agent, plus):
      - hint: str — coaching hint generated from the prior rollout
      - old_reward: float — reward from the prior rollout (0.0 or 1.0)
    """
    character_data = data["extra_info"]
    hint = character_data.get("hint", "")
    old_reward = float(character_data.get("old_reward", 0.0))

    teacher_prompt = get_teacher_prompt(character_data, hint)
    chat = [
        {"role": "system", "content": ""},
        {"role": "user", "content": teacher_prompt},
    ]

    actor_agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await actor_agent.step()

    predicted = remove_think(response)
    predicted = extract_choice(predicted)

    correct_answer = character_data.get("Multiple Choice Question", {}).get("Correct Answer", "")
    if correct_answer:
        correct_answer = correct_answer.strip().upper()
        if len(correct_answer) > 1:
            correct_answer = correct_answer[0]

    is_correct = predicted == correct_answer if predicted else False
    new_reward = 1.0 if is_correct else 0.0
    reward_delta = new_reward - old_reward

    extra_info = {
        "lifechoices-hint/reward": new_reward,
        "lifechoices-hint/reward_delta": reward_delta,
        "lifechoices-hint/delta_positive": int(reward_delta > 0),
    }

    output = await actor_agent.get_agent_output(new_reward, extra_info=extra_info)
    return output