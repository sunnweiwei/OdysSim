# Copyright 2025 Individual Contributor: OdysSim Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hint generation for the TwinVoice agent.

Generates a concise coaching hint after the model picks the wrong persona response.
The hint must NOT reveal the correct answer — it only draws attention to specific
stylistic patterns, tone, or behavioral traits in the persona's history that the
model likely overlooked.
"""

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(row: dict, hint: str) -> str:
    """Build the teacher version of the TwinVoice prompt with the hint injected.

    The hint appears as a natural thinking reminder just before the response
    choices, so the model absorbs it before evaluating options.
    """
    from agents.twinvoice.agent import build_prompt

    base_prompt = build_prompt(row)

    reminder = f"\nAs you reason through this, keep in mind: {hint}\n"
    # Insert before the "# Response Choices:" section
    if "# Response Choices:" in base_prompt:
        base_prompt = base_prompt.replace("# Response Choices:", reminder + "# Response Choices:", 1)
    else:
        base_prompt = base_prompt + reminder

    return base_prompt


def _build_hint_prompt(row: dict, content: str | None) -> str:
    """Build the prompt sent to the hint-generation LLM.

    The hint LLM sees the full persona history and the correct answer privately,
    but is instructed never to reveal or imply the correct option letter.
    """
    history = row.get("conversation_history") or row.get("history") or []
    if isinstance(history, str):
        import json

        history = json.loads(history)
    history_text = "\n".join(f"- {h}" for h in history)

    anchor_post = row.get("anchor_post") or ""
    choices = row.get("answer_choices") or row.get("choices") or []
    if isinstance(choices, str):
        import json

        choices = json.loads(choices)
    correct_idx = row.get("answer_idx") or 0
    correct_text = choices[correct_idx] if 0 <= correct_idx < len(choices) else ""

    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(choices))
    predicted_str = content if content else "(no response)"

    return f"""You are a coaching assistant helping a language model improve at persona-driven response selection.

A model was asked to identify which response a specific persona would most likely write, and it chose incorrectly.

## Persona Conversation History
{history_text}

## Anchor Post (Stimulus)
{anchor_post}

## Response Choices
{options_text}

## Model's reasoning and answer
{predicted_str}

## Correct answer (for your reference only — do NOT reveal this)
{correct_text}

---

Write a short coaching hint (1 sentence) to help the model do better on its next attempt.

**Strict rules:**
- Do NOT state or imply which option letter is correct.
- Use the correct answer to identify which specific stylistic pattern, tone, vocabulary, or behavioral trait in the persona's history the model overlooked, then hint at that without naming the option.
- Be concrete — reference observable patterns from the persona's history.
- One sentence only.

Output only the hint text, no preamble."""


async def generate_hint(row: dict, content: str | None) -> str:
    """Generate a concise persona-style hint.

    Returns an empty string on failure so callers can treat it as optional.
    """
    prompt = _build_hint_prompt(row, content)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    hint_text = None
    try:
        async with asyncio.timeout(120):
            hint_text = await call_openai(messages, model="gpt-5.4-nano", reasoning_effort="low")
            if hint_text:
                hint_text = remove_think(hint_text).strip()
    except asyncio.TimeoutError:
        logger.warning("TwinVoice hint generation timed out")
    except Exception as e:
        logger.warning(f"TwinVoice hint generation failed: {e}")

    return hint_text or ""
