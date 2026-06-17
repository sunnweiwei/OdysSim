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
Hint generation for the SocSci210 agent.

Generates a concise reference response that explains why a survey respondent with
this persona would give the ground-truth answer to this stimulus. The hint is
then used as a teacher example in the hint agent's second attempt (mirrors the
"oracle reasoning trace" used by Kolluri et al., 2025).
"""

from __future__ import annotations

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def get_teacher_prompt(row: dict, reference_response: str) -> str:
    """Build a teacher prompt showing a reference response to imitate."""
    from agents.socsci210.agent import _build_prompt_from_row

    base_prompt = _build_prompt_from_row(row)
    reminder = (
        "\nHere is a reference response showing a concise and correct way to answer this question:\n"
        f"{reference_response}\n\n"
        "Now answer the same question in the same style.\n"
    )
    return base_prompt + reminder


def _format_gold(gold, response_type: str) -> str:
    """Render the ground-truth response as the string a respondent would emit."""
    if gold is None:
        return ""
    try:
        gold_int = int(float(gold))
    except (TypeError, ValueError):
        return str(gold).strip()
    if response_type == "binary":
        return "yes" if gold_int == 1 else "no"
    return str(gold_int)


def _build_hint_prompt(row: dict) -> str:
    from agents.socsci210.agent import _build_prompt_from_row

    prompt = _build_prompt_from_row(row)
    response_type = str(row.get("response_type") or "ordinal").strip().lower()
    gold_str = _format_gold(row.get("response"), response_type)
    r_min = row.get("r_min")
    r_max = row.get("r_max")

    if response_type == "binary":
        scale_desc = "yes/no"
    elif r_min is not None and r_max is not None:
        scale_desc = f"integer from {int(r_min)} to {int(r_max)}"
    else:
        scale_desc = "numeric"

    return f"""You are simulating a survey respondent in a social science experiment. A respondent with a specific demographic profile answered the following prompt:

{prompt}

The response they gave was: {gold_str} (on a {scale_desc} scale)

Write a short reference response (1-2 concise sentences) that plausibly explains — from this respondent's perspective — why they would give that answer. Draw on the demographic cues in the prompt. Then end with exactly:
<answer>{gold_str}</answer>"""


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
        logger.warning("SocSci210 hint generation timed out")
    except Exception as e:
        logger.warning(f"SocSci210 hint generation failed: {e}")

    return hint_text or ""
