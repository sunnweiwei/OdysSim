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
Hint generation for the HUMANUAL agent.

Uses the judge output (key_points, thought, score) to generate a concise
1-sentence coaching hint via an LLM call.
"""

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


def _build_hint_prompt(
    judge_result: dict,
    generated: str,
    completion: str,
    dimension: str,
    dimension_desc: str,
) -> str:
    key_points = str(judge_result.get("key_points", "")).strip()
    thought = str(judge_result.get("thought", "")).strip()
    score = judge_result.get("score", None)

    score_str = f"{float(score):.2f}/1.0" if score is not None else "unknown"
    predicted_str = generated if generated else "(no response)"

    return f"""You are a coaching assistant helping a language model improve at persona-driven user simulation.

The model was asked to generate a **{dimension}** for a persona. The evaluation criteria:
{dimension_desc}

The model's previous attempt scored {score_str}.

## Reference response (ground truth)
{completion}

## Model's previous response
{predicted_str}

## Judge's key points from the reference
{key_points}

## Judge's assessment of what was matched or missed
{thought}

---

Write a short coaching hint (1 sentence) to help the model do better on its next attempt.

**Rules:**
- Focus on the most important gap between the model's response and the reference — tone, style, content, or length.
- Do NOT copy or quote the reference verbatim.
- One sentence only.

Output only the hint text, no preamble."""


async def generate_hint(
    judge_result: dict | None,
    generated: str,
    completion: str = "",
    dimension: str = "response",
    dimension_desc: str = "",
) -> str:
    """Generate a concise 1-sentence coaching hint from judge output and reference.

    Returns an empty string on failure so callers can treat it as optional.
    """
    if not judge_result:
        return ""

    key_points = str(judge_result.get("key_points", "")).strip()
    thought = str(judge_result.get("thought", "")).strip()
    if not key_points and not thought:
        return ""

    prompt = _build_hint_prompt(judge_result, generated, completion, dimension, dimension_desc)
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
        logger.warning("Humanual hint generation timed out")
    except Exception as e:
        logger.warning(f"Humanual hint generation failed: {e}")

    return hint_text or ""


def get_teacher_system_prompt(persona: str, dimension: str, hint: str, reference: str) -> str:
    """Build an augmented system prompt with judge-derived coaching and the reference response.

    The reference is framed as inspiration rather than a template to copy verbatim.
    """
    from agents.humanual.agent import _SYSTEM_PROMPT_TEMPLATE, STATE_DESCRIPTIONS
    from agents.utils import truncate_text

    reference = truncate_text(reference, 2000)

    base_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        persona=persona,
        dimension=dimension,
        dimension_desc=STATE_DESCRIPTIONS[dimension],
    )

    guidance_section = (
        "\n## Guidance for this response\n"
        f"{hint}\n\n"
        "## Reference response\n"
        f"(Use this as inspiration for the right tone, content, and style)\n"
        f"{reference}\n"
    )

    # Insert guidance just before ## Task and Output format
    if "## Task and Output format:" in base_prompt:
        base_prompt = base_prompt.replace(
            "## Task and Output format:",
            guidance_section + "## Task and Output format:",
            1,
        )
    else:
        base_prompt = base_prompt + "\n" + guidance_section

    return base_prompt
