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
Hint generation for the HiToM agent.

Uses an LLM to produce a single concrete sentence that encodes the key reasoning
step — specific to the story's events and agents — so the teacher model can
trivially infer the correct answer without the answer being stated explicitly.
"""

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(row: dict, hint: str) -> str:
    """Build the teacher version of the prompt with the hint as a natural reminder."""
    from agents.hitom.agent import PROMPT_TEMPLATE

    story = row["story"]
    question_with_choices = row["question"] + "\n" + row["choices"]
    extra_info = row.get("extra_info", "")

    base_prompt = PROMPT_TEMPLATE.format(
        story=story,
        extra_info=extra_info,
        question=question_with_choices,
    )

    reminder = f"\nAs you reason through this, keep in mind: {hint}\n"
    if "## Question" in base_prompt:
        base_prompt = base_prompt.replace("## Question", reminder + "## Question", 1)
    else:
        base_prompt = base_prompt + reminder

    return base_prompt


def _build_hint_prompt(row: dict) -> str:
    story = row.get("story", "")
    question = row.get("question", "")
    choices = row.get("choices", "")
    correct_answer = str(row.get("correct_answer", ""))
    deception = str(row.get("deception", "False")).lower() == "true"

    deception_note = (
        "NOTE: This story involves deception — verbal statements do NOT change actual beliefs."
        if deception
        else "NOTE: This story does not involve deception."
    )

    return f"""You are helping a model reason about a Theory-of-Mind question. The model answered incorrectly.

{deception_note}

HiToM rules:
- An agent knows only what they directly witnessed before exiting a location.
- Spoken statements do NOT update an agent's actual belief.
- An agent trusts whoever exited the room later than themselves.
- Private communications are not heard by others.

## Story
{story}

## Question
{question}
{choices}

## Correct answer
{correct_answer}

---

Write exactly ONE very concise sentence that states the critical reasoning step — specific to this story's agents and events — that makes the correct answer obvious. Do NOT directly state the answer. Just give the key fact."""


async def generate_hint(row: dict, content: str) -> str:
    """Generate a concrete one-sentence HiToM reasoning hint via LLM."""
    prompt = _build_hint_prompt(row)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    try:
        async with asyncio.timeout(60):
            hint_text = await call_openai(messages, model="gpt-5.4-nano", reasoning_effort="low")
            if hint_text:
                return remove_think(hint_text).strip()
    except asyncio.TimeoutError:
        logger.warning("HiToM hint generation timed out")
    except Exception as e:
        logger.warning(f"HiToM hint generation failed: {e}")
    return ""
