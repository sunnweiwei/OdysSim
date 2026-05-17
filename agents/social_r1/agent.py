"""
Search-R1 agent for Harmony evaluation.

Single-turn benchmark:
- Input row contains one social-reasoning multiple-choice question
- Agent answers with an option letter
- Reward is 1.0 when the predicted option matches the gold answer
"""

from __future__ import annotations

import copy
import re
import uuid
from typing import Any, Dict, Optional

from agents.utils import Agent, process_post_chat, remove_think

ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE)
OPTION_LINE_RE = re.compile(r"^\s*([A-Z])[\.\)]\s+", re.MULTILINE)

SYSTEM_PROMPT = (
    "You are solving a social reasoning multiple-choice question.\n"
    "Read the story, question, and options carefully.\n"
    "Think concisely, then output your final answer in the format:\n"
    "<answer>X</answer>\n"
    "where X is exactly one option letter from the provided choices."
)


def _extract_option_letter(text: str) -> Optional[str]:
    """Extract the final option letter from a model response."""
    if not text:
        return None

    match = ANSWER_TAG_RE.search(text)
    if match:
        return match.group(1).upper()

    return None


def _extract_gold_letter(answer_text: str) -> str:
    """Extract the leading answer letter from a gold answer string like 'B. In the box'."""
    if not answer_text:
        return ""
    match = re.match(r"^\s*([A-Z])[\.\)]", answer_text)
    if match:
        return match.group(1).upper()
    return answer_text.strip()[:1].upper()


def _build_prompt_from_row(row: Dict[str, Any]) -> str:
    """Build the user prompt from normalized or raw human-sim formatted rows."""
    prompt_text = str(row.get("prompt_text") or "").strip()
    if prompt_text:
        return prompt_text

    conversations = row.get("conversations") or []
    if conversations:
        messages = conversations[0].get("messages") or []
        if messages:
            return str(messages[0].get("content") or "").strip()

    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        return prompt

    return ""


async def agent_loop(data: dict, context):
    """
    Search-R1: answer one social-reasoning multiple-choice question.

    Input:
      data["extra_info"]: normalized row from prepare_dataset.py
    """
    row = data["extra_info"]
    prompt = _build_prompt_from_row(row)
    if not prompt:
        raise ValueError("Search-R1 row is missing prompt text.")

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    content = remove_think(response)
    predicted = _extract_option_letter(content)

    correct = str(row.get("answer_letter") or "").strip().upper()
    if not correct:
        correct = _extract_gold_letter(str(row.get("answer_text") or ""))

    reward = 1.0 if (predicted is not None and predicted == correct) else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "social_r1/reward": reward,
            "social_r1/response_length": len(response.split()) if response else 0,
            "all/score": reward
        },
    )

    # ===========================================================================
    # Hint + second attempt
    # ===========================================================================
    extra = {}
    hint = None
    if (getattr(context.config.algorithm, "agent_version", None) == "copy"
            and reward < 1.0):
        from agents.social_r1.hint import generate_hint
        hint = await generate_hint(row, content)
        if hint:
            extra["hint"] = hint

    if (getattr(context.config.algorithm, "agent_version", None) == "copy"
            and context.is_train
            and hint):
        from agents.social_r1.hint_agent import agent_loop as hint_agent_loop
        data["extra_info"]["hint"] = hint
        data["extra_info"]["old_reward"] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = "hint_agent"
        output = [output, copy_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output
