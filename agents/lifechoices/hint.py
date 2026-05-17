"""
Hint generation for the LifeChoices agent.

Generates a concise, high-level coaching hint after the model answers incorrectly.
The hint must NOT reveal the correct answer — it only guides the model to reason
more deeply in-character, pointing to relevant profile/memory aspects and the
right reasoning approach.
"""

import asyncio
import logging
from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(character_data: dict, hint: str) -> str:
    """Build the teacher version of the prompt with coaching notes appended.

    The coaching note appears before the task so the model internalizes it
    before reading the options.  The correct answer is never included.
    """
    from agents.lifechoices.agent import create_prompt
    from agents.utils import truncate_text
    character_data = {
        **character_data,
        "input_text": truncate_text(str(character_data.get("input_text", "")), 2000),
    }
    base_prompt = create_prompt(character_data)

    # Insert the hint as a natural thinking reminder just before the Outputs section,
    # so the model absorbs it as part of the task rather than referencing it explicitly.
    reminder = f"\nAs you reason through this, keep in mind: {hint}\n"
    if "# Outputs:" in base_prompt:
        base_prompt = base_prompt.replace("# Outputs:", reminder + "# Outputs:", 1)
    else:
        base_prompt = base_prompt + reminder

    return base_prompt


def _build_hint_prompt(
        character_data: dict,
        content: str | None,
) -> str:
    """Build the prompt sent to the hint-generation LLM.

    The hint LLM sees the full character data and the model's wrong answer,
    but is instructed never to reveal or imply the correct option letter.
    """
    mcq = character_data.get("Multiple Choice Question", {})
    scenario = mcq.get("Scenario", "")
    question = mcq.get("Question", "")
    options = mcq.get("Options", [])
    character_name = character_data.get("character_name", "the character")
    description = character_data.get("character_name", "")
    memory = character_data.get("input_text", "")

    options_text = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )

    correct_answer = mcq.get("Correct Answer", "")
    predicted_str = content if content else "(no response)"

    return f"""You are a coaching assistant helping a language model improve at persona-driven roleplay decisions.

A model was asked to choose what the literary character **{character_name}** would do in a scenario, and it answered incorrectly.

## Character Profile
**Name:** {character_name}
**Description / Background:**
{description}

**Memory (key experiences):**
{memory}

## The Question
**Scenario:** {scenario}
**Question:** {question}
**Options:**
{options_text}

**Model's reasoning and answer:** {predicted_str}

## Correct answer (for your reference only — do NOT reveal this)
{correct_answer}

---

Write a short coaching hint (1 sentence) to help the model do better on its next attempt.

**Strict rules:**
- Do NOT state or imply which option letter is correct.
- Use the correct answer to identify which character trait, value, or memory the model overlooked, then hint at that without naming the option.
- Be concrete and reference the character and scenario directly.
- One sentence only.

Output only the hint text, no preamble."""


async def generate_hint(
        character_data: dict,
        content: str | None,
) -> str:
    """Generate a concise in-character reasoning hint.

    Returns an empty string on failure so callers can treat it as optional.
    """
    prompt = _build_hint_prompt(character_data, content)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    hint_text = None
    try:
        async with asyncio.timeout(120):
            hint_text = await call_openai(messages, model='gpt-5.4-nano', reasoning_effort='low')
            if hint_text:
                hint_text = remove_think(hint_text).strip()
    except asyncio.TimeoutError:
        logger.warning("LifeChoices hint generation timed out")
    except Exception as e:
        logger.warning(f"LifeChoices hint generation failed: {e}")

    return hint_text or ""