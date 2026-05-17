"""
Hint generation for the Mistakes agent.

Generates a concise hint that concretizes the misconception as a specific wrong
computation on this particular problem. The hint must NOT reveal the target option
letter or directly alter the model's behavior — it only makes the misconception
more vivid and actionable so the model can simulate it faithfully.
"""

import asyncio
import logging
from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(row: dict, reference_response: str) -> str:
    """Build the teacher version of the prompt showing a reference response to repeat."""
    from agents.mistakes.agent import _build_prompt_from_row

    base_prompt = _build_prompt_from_row(row)

    reminder = (
        f"\nHere is a reference response showing how a student with this misconception may reason:\n"
        f"{reference_response}\n\n"
    )
    return base_prompt + reminder


def _build_hint_prompt(row: dict) -> str:
    from agents.mistakes.agent import OPTION_LETTERS

    options = []
    for letter in OPTION_LETTERS:
        text = str(row.get(f"Answer{letter}Text") or "").strip()
        if text:
            options.append(f"{letter}) {text}")
    options_text = "\n".join(options)

    misconception = str(row.get("MisconceptionName") or row.get("misconception_text") or "").strip()
    question = str(row.get("QuestionText") or row.get("question") or "").strip()
    target_option = str(row.get("TargetOption") or row.get("target_option") or "").strip().upper()
    target_text = str(row.get("TargetAnswer") or row.get("target_text") or "").strip()

    return f"""You are simulating a student who has a specific misconception in math.

Math Problem:
Question: {question}
Answer Choices:
{options_text}

Student Misconception:
{misconception}

The student with this misconception picks option {target_option}: {target_text}

Write a short response as this student would — show their (wrong) reasoning in 1-2 concise sentences."""


async def generate_hint(row: dict, content: str) -> str:
    """Generate a concise misconception-concretizing hint. Returns empty string on failure."""
    prompt = _build_hint_prompt(row)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    hint_text = None
    try:
        async with asyncio.timeout(100):
            hint_text = await call_openai(messages, model='gpt-5.4-nano', reasoning_effort='medium')
            if hint_text:
                hint_text = remove_think(hint_text).strip()
    except asyncio.TimeoutError:
        logger.warning("Mistakes hint generation timed out")
    except Exception as e:
        logger.warning(f"Mistakes hint generation failed: {e}")

    return hint_text or ""