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
AlignX agent_loop — personalized 2-way preference selection.

Source: https://github.com/JinaLeejnl/AlignX (HF: JinaLeejnl/AlignX,
JinaLeejnl/AlignX-test).

Each example is a (prompt, chosen, rejected) triple plus optional
personalization context: a natural-language "Demographic Information"
description of the user, prior pairwise preferences, and a 90-dim preference
direction vector. The trained model must predict which of two presented
responses (A vs B, randomly assigned per example) the user would prefer.

Reward = 1 if model picks the chosen response, 0 otherwise. The randomization
defeats position bias — without it, a model that always says 'A' would score
~50% reward "for free".

Input row (from prepare_dataset.process_alignx*):
  extra_info = {
    "prompt":    str,        # the post
    "chosen":    str,        # user's preferred response
    "rejected":  str,        # the other response
    "demographic":  str,                       # natural-language persona (optional)
    "profile":   str,                          # synthesized user profile (test-only fallback)
    "icl_pairs": list[{prompt,chosen,rejected}],  # prior preferences (optional)
    "ugc":       list[{prompt,comment}],       # prior user comments (optional)
    "n_icl":     int,        # how many icl_pairs to include in-context (default 2)
    "n_ugc":     int,        # how many ugc comments to include (default 2)
    "index":     int,
  }

Persona priority: demographic > profile (latter is the AlignX-test field that
synthesizes a profile from whichever conditioning signal a given test split
carries; e.g. Reddit_PAIR has empty `demographic` but a populated `profile`).
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from agents.utils import Agent, process_post_chat, remove_think

SYSTEM_PROMPT = (
    "You are predicting which response a specific user would prefer. "
    "Read the user description and any prior preference examples carefully, "
    "then choose the response (A or B) that this user would prefer. "
    "You may think step by step concisely before answering. "
    "Output your final choice as <answer>A</answer> or <answer>B</answer>."
)

ANSWER_PATTERN = re.compile(r"<answer>\s*([AB])\s*</answer>")


def _predict_choice(model_output: str) -> Optional[str]:
    """Return 'A' or 'B' only if wrapped in <answer>...</answer>, else None."""
    if not model_output:
        return None
    m = ANSWER_PATTERN.search(model_output)
    return m.group(1) if m else None


def _format_persona_context(row: dict[str, Any]) -> str:
    """Build the personalization preamble.

    Prefers `demographic`; falls back to `profile` when demographic is empty
    (AlignX-test/Reddit_PAIR and Reddit_UGC strip demographic and only carry
    a `profile` summary synthesized from the surviving conditioning signal).
    Adds in-context examples from prior preference pairs and prior comments.
    """
    parts = []

    demographic = (row.get("demographic") or row.get("Demographic Information") or "").strip()
    profile = (row.get("profile") or "").strip()
    persona_text = demographic or profile
    if persona_text:
        parts.append(f"### About this user\n{persona_text}")

    icl_pairs = row.get("icl_pairs") or row.get("Pair-wise Comparative Feedback") or []
    n_icl = int(row.get("n_icl", 2))
    if icl_pairs and n_icl > 0:
        examples = []
        for i, pair in enumerate(icl_pairs[:n_icl]):
            p = (pair.get("prompt") or "").strip()
            c = (pair.get("chosen") or "").strip()
            r = (pair.get("rejected") or "").strip()
            if p and c and r:
                examples.append(f"Example {i + 1}:\nPost: {p}\nThis user preferred: {c}\nOver: {r}")
        if examples:
            parts.append("### Prior preference examples for this user\n\n" + "\n\n".join(examples))

    ugc = row.get("ugc") or row.get("User-Generated Content") or []
    n_ugc = int(row.get("n_ugc", 2))
    if ugc and n_ugc > 0:
        comments = []
        for i, item in enumerate(ugc[:n_ugc]):
            p = (item.get("prompt") or "").strip()
            c = (item.get("comment") or "").strip()
            if p and c:
                comments.append(f"On post: {p}\nUser said: {c}")
        if comments:
            parts.append("### Prior comments by this user\n\n" + "\n\n".join(comments))

    return "\n\n".join(parts)


def _build_prompt(row: dict[str, Any], a_text: str, b_text: str) -> str:
    persona = _format_persona_context(row)
    post = (row.get("raw_prompt") or "").strip()
    parts = []
    if persona:
        parts.append(persona)
    parts.append(f"### Post\n{post}")
    parts.append(f"### Response A\n{a_text}")
    parts.append(f"### Response B\n{b_text}")
    parts.append("Which response would this user prefer? Reply with <answer>A</answer> or <answer>B</answer>.")
    return "\n\n".join(parts)


async def agent_loop(data: dict[str, Any], context):
    row = data.get("extra_info") or {}
    prompt = (row.get("raw_prompt") or "").strip()
    chosen = (row.get("chosen") or "").strip()
    rejected = (row.get("rejected") or "").strip()
    if not prompt or not chosen or not rejected:
        raise ValueError("AlignX row missing prompt/chosen/rejected")

    # Deterministic per-row A/B assignment so reruns are reproducible but the
    # model can't position-bias its way to a free 50%.
    rng = random.Random(int(row.get("index", 0)))
    flip = rng.random() < 0.5
    if flip:
        a_text, b_text, gold_letter = rejected, chosen, "B"
    else:
        a_text, b_text, gold_letter = chosen, rejected, "A"

    user_prompt = _build_prompt(row, a_text, b_text)
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    agent = Agent(
        context.llm_client,
        chat,
        context.tokenizer,
        context.config,
        prompt_turn=2,
        enable_think=True,
    )
    response = await agent.step()
    content = remove_think(response or "")

    predicted = _predict_choice(content)
    has_prediction = predicted is not None
    correct = (predicted == gold_letter) if has_prediction else False
    reward = 1.0 if correct else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "alignx/reward": reward,
            "alignx/response_length": len(response.split()) if response else 0,
            "all/score": reward,
        },
    )
    await process_post_chat(data, context, agent.chat, output)
    return output
