"""
Hint generation for the FanToM agent.

Generates a concise, type-aware coaching hint after the model answers incorrectly
or partially. The hint must NOT reveal the correct answer — it guides the model
toward the right reasoning approach for the specific question type.
"""

import asyncio
import logging
from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)

# High-level reasoning guidance per question type, used to build targeted hints.
_QT_GUIDANCE = {
    "multiple-choice": (
        "Focus on tracking exactly what information each character has directly witnessed "
        "or been told — not what they could infer or what the reader knows."
    ),
    "list": (
        "Carefully trace which characters were present or informed at each point in the "
        "conversation — include everyone who qualifies and exclude everyone who does not."
    ),
    "binary": (
        "Determine strictly whether the named character had direct access to the relevant "
        "information through their own presence or explicit communication, not through inference."
    ),
    "fact": (
        "Answer based only on what is explicitly stated in the conversation context; "
        "aim for a complete and precise response that covers all key details."
    ),
}


def _get_qt_guidance(question_type: str) -> str:
    if question_type.endswith(":multiple-choice"):
        return _QT_GUIDANCE["multiple-choice"]
    elif question_type.endswith(":list"):
        return _QT_GUIDANCE["list"]
    elif question_type.endswith(":binary"):
        return _QT_GUIDANCE["binary"]
    elif question_type.startswith("fact"):
        return _QT_GUIDANCE["fact"]
    return "Think carefully about what each character knows based solely on what they witnessed or were told."


def get_teacher_prompt(row: dict, hint: str) -> str:
    """Build the teacher version of the prompt with a coaching note inserted."""
    from agents.fantom.agent import PROMPT_TEMPLATE

    context_text = row.get("context", "")
    complete_question = row.get("complete_question", row.get("question", ""))
    extra_info = row.get("extra_info", "")

    base_prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        extra_info=extra_info,
        question=complete_question,
    )

    # Insert the hint as a natural thinking reminder just before the Question section,
    # so the model absorbs it as part of the task rather than referencing it explicitly.
    reminder = f"\nAs you reason through this, keep in mind: {hint}\n"
    if "## Question" in base_prompt:
        base_prompt = base_prompt.replace("## Question", reminder + "## Question", 1)
    else:
        base_prompt = base_prompt + reminder

    return base_prompt


def _build_hint_prompt(row: dict, content: str, reward: float) -> str:
    question_type = str(row.get("question_type", ""))
    context_text = row.get("context", "")
    complete_question = row.get("complete_question", row.get("question", ""))
    correct_answer = str(row.get("correct_answer", ""))
    qt_guidance = _get_qt_guidance(question_type)

    return f"""You are a coaching assistant helping a model improve at Theory-of-Mind reasoning.

The model answered a FanToM question incorrectly (question type: {question_type}).

## Conversation Context
{context_text}

## Question
{complete_question}

## Model's reasoning and answer
{content if content else "(no response)"}

## Correct answer
{correct_answer}

---

Write a 1-sentence coaching hint to help the model reason better on its next attempt.

**Strict rules:**
- Do NOT directly state or imply which option letter is correct.
- Use the correct answer to identify what the model missed, then hint at the reasoning approach: {qt_guidance}
- Reference the specific conversation and question directly.
- One sentence only."""


async def generate_hint(row: dict, content: str, reward: float) -> str:
    """Generate a concise ToM reasoning hint. Returns empty string on failure."""
    prompt = _build_hint_prompt(row, content, reward)
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
        logger.warning("FanToM hint generation timed out")
    except Exception as e:
        logger.warning(f"FanToM hint generation failed: {e}")

    return hint_text or ""