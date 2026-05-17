"""
Hint generation for the HumanLLM Item Selection agent.

Generates a concise reference response that explains why the simulated user
would pick the gold candidate, drawing on their persona and purchase history.
The hint is then used as a teacher example in the hint agent's second attempt.

Mirrors the social_r1 / socsci210 hint pattern: the hint LLM is shown the gold
answer up front and emits a 1-2 sentence rationale ending with
`<answer>{letter}</answer>` for format consistency.
"""

from __future__ import annotations

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(row: dict, reference_response: str) -> str:
    """Build a teacher prompt that appends the hint as a reference response."""
    from agents.humanllm.agent import _build_prompt_from_row

    base_prompt = _build_prompt_from_row(row)
    reminder = (
        "\nHere is a reference response showing a concise and correct way to pick the item:\n"
        f"{reference_response}\n\n"
        "Now pick the item in the same style.\n"
    )
    return base_prompt + reminder


def _build_hint_prompt(row: dict) -> str:
    """Build the prompt sent to the hint-generation LLM."""
    from agents.humanllm.agent import _build_prompt_from_row

    base_prompt = _build_prompt_from_row(row)
    answer_letter = str(row.get("answer_letter") or "").strip().upper()
    answer_index = int(row.get("answer_index", -1))
    answer_text = str(row.get("answer_text") or "").strip()
    candidate_num = answer_index + 1  # 1-based to match "Candidate N" in prompt

    return f"""You are simulating a user choosing items they would purchase next, in a 20-way multiple-choice setup.

{base_prompt}

The correct answer is Candidate {candidate_num} (letter {answer_letter}):
{answer_text}

Write a short reference response (1-2 concise sentences) that explains — from this user's perspective — why they picked this item, drawing on their persona and purchase history. Then end with exactly:
<answer>{answer_letter}</answer>"""


async def generate_hint(row: dict, content: str) -> str:
    """Generate a concise reference response. Returns empty string on failure."""
    prompt = _build_hint_prompt(row)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]

    hint_text = None
    try:
        async with asyncio.timeout(100):
            hint_text = await call_openai(messages, model="gpt-5.4-mini", reasoning_effort="low")
            if hint_text:
                hint_text = remove_think(hint_text).strip()
    except asyncio.TimeoutError:
        logger.warning("HumanLLM hint generation timed out")
    except Exception as e:
        logger.warning(f"HumanLLM hint generation failed: {e}")

    return hint_text or ""
