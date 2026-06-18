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
Hint generation for the Search-R1 agent.

Generates a concise reference response that explains the correct answer for a
social-reasoning multiple-choice question. The hint is then used as a teacher
example in the hint agent's second attempt.
"""

from __future__ import annotations

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(row: dict, reference_response: str) -> str:
    """Build a teacher prompt showing a reference response to imitate."""
    from agents.social_r1.agent import _build_prompt_from_row

    base_prompt = _build_prompt_from_row(row)
    reminder = (
        "\nHere is a reference response showing a concise and correct way to solve this question:\n"
        f"{reference_response}\n\n"
        "Now answer the same question in the same style.\n"
    )
    return base_prompt + reminder


def _build_hint_prompt(row: dict) -> str:
    from agents.social_r1.agent import _build_prompt_from_row, _extract_gold_letter

    prompt = _build_prompt_from_row(row)
    answer_letter = str(row.get("answer_letter") or "").strip().upper()
    answer_text = str(row.get("answer_text") or "").strip()
    if not answer_letter:
        answer_letter = _extract_gold_letter(answer_text)

    return f"""You are solving a social reasoning multiple-choice question.

Question:
{prompt}

The correct answer is {answer_letter}: {answer_text}

Write a short reference response that explains why this answer is correct in 1-2 concise sentences, then end with:
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
        logger.warning("Search-R1 hint generation timed out")
    except Exception as e:
        logger.warning(f"Search-R1 hint generation failed: {e}")

    return hint_text or ""
