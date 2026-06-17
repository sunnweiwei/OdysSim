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
Hint generation for the MirrorBench agent.

Uses the GTEval judge output (reasoning + score) to generate a focused
1-sentence hint via an LLM call.
"""

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)


async def generate_hint(gteval_result: dict | None) -> str:
    """Generate a concise 1-sentence hint from the GTEval judge output.

    Returns an empty string if the judge result is unavailable or on failure.
    """
    if not gteval_result:
        return ""

    reasoning = str(gteval_result.get("reasoning", "")).strip()
    score = gteval_result.get("score", None)

    if not reasoning:
        return ""

    score_str = f"{float(score):.2f}/1.0" if score is not None else "unknown"

    prompt = f"""You are a coaching assistant helping a language model improve at simulating a human user in conversation.

The model's user-proxy simulation scored {score_str} on style and realism.

## Judge's assessment
{reasoning}

---

Write a short coaching hint (1 sentence) focused on the most important stylistic or behavioral aspect the model should improve to sound more like a real human user.

**Rules:**
- Focus on tone, naturalness, length, or specificity — whichever the judge flagged most.
- One sentence only.

Output only the hint text, no preamble."""

    hint_text = None
    try:
        async with asyncio.timeout(120):
            hint_text = await call_openai(
                [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                model="gpt-5.4-nano",
                reasoning_effort="low",
            )
            if hint_text:
                hint_text = remove_think(hint_text).strip()
    except asyncio.TimeoutError:
        logger.warning("MirrorBench hint generation timed out")
    except Exception as e:
        logger.warning(f"MirrorBench hint generation failed: {e}")

    return hint_text or ""


def get_teacher_system_prompt(
    task_description: str,
    domain: str | None,
    persona: str | None,
    hint: str,
    real_conversation_str: str,
) -> str:
    """Build an augmented user-proxy system prompt with judge coaching and the real conversation.

    The real conversation is framed as a style reference — not a script to copy.
    """
    from agents.mirrorbench.agent import build_user_proxy_system_prompt
    from agents.utils import truncate_text

    base_prompt = build_user_proxy_system_prompt(
        task_description=task_description,
        domain=domain,
        persona=persona,
    )

    real_conversation_str = truncate_text(real_conversation_str, 2000)

    guidance_section = (
        "\n\n## Guidance for this simulation\n"
        f"{hint}\n\n"
        "## Real user messages (reference)\n"
        "Use these as reference for how the real user communicates in this conversation.\n"
        f"{real_conversation_str}"
    )

    return base_prompt + guidance_section
