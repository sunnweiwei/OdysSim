"""
Hint generation for the ParaToMI agent.

Generates a concise, 1-sentence coaching hint after the model answers incorrectly.
ParaToMI is a location-tracking ToM task: hints focus on object movement chains
and character presence without revealing the correct location.
"""

import asyncio
import logging
from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(row: dict, hint: str) -> str:
    """Build the teacher version of the prompt with the hint as a natural reminder."""
    from agents.paratomi.agent import build_prompt

    base_prompt = build_prompt(row)

    reminder = f"\nAs you reason through this, keep in mind: {hint}\n"
    if "## Question" in base_prompt:
        base_prompt = base_prompt.replace("## Question", reminder + "## Question", 1)
    else:
        base_prompt = base_prompt + reminder

    return base_prompt


def _build_hint_prompt(row: dict, content: str) -> str:
    from agents.paratomi.agent import parse_story

    story = parse_story(row.get("story", ""))
    question = row.get("question", "")
    correct_answer = str(row.get("correct_answer", ""))
    question_type = str(row.get("qType", ""))

    return f"""You are a coaching assistant helping a model improve at Theory-of-Mind location-tracking reasoning.

The model answered a ParaToMI question incorrectly (question type: {question_type}).
ParaToMI key rules:
- Characters only perceive scenes in their current location.
- When an object is moved, it is no longer in its original location.
- Answer with the most detailed position possible (if object is in A and A is in B, answer 'A').

## Story
{story}

## Question
{question}

## Model's reasoning and answer
{content if content else "(no response)"}

## Correct answer
{correct_answer}

---

Write a 1-sentence hint to guide the model's reasoning on its next attempt.

**Strict rules:**
- Do NOT directly state or imply which option letter is correct.
- Use the correct answer to identify what the model missed (e.g., a movement step, a character's absence, belief vs. reality), then hint at that reasoning gap.
- One sentence only."""


async def generate_hint(row: dict, content: str) -> str:
    """Generate a concise location-tracking reasoning hint. Returns empty string on failure."""
    prompt = _build_hint_prompt(row, content)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    hint_text = None
    try:
        async with asyncio.timeout(60):
            hint_text = await call_openai(messages, model='gpt-5.4-nano', reasoning_effort='low')
            if hint_text:
                hint_text = remove_think(hint_text).strip()
    except asyncio.TimeoutError:
        logger.warning("ParaToMI hint generation timed out")
    except Exception as e:
        logger.warning(f"ParaToMI hint generation failed: {e}")

    return hint_text or ""