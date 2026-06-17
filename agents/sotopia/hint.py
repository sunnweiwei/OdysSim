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

import asyncio
import logging

from agents.utils import call_openai, remove_think

logger = logging.getLogger(__name__)

DIMENSIONS = [
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
]

ACTION_TYPES = ["speak", "non-verbal communication", "action", "leave"]

# Brief rubric descriptions for the teacher prompt (concise for in-context use)
_RUBRIC_BRIEF = {
    "goal": "0–10 · Did you fully achieve your social goal? (highest weight)",
    "relationship": "-5–5 · Did the interaction improve or damage your relationship with the partner?",
    "knowledge": "0–10 · Did you gain useful new information through the interaction?",
    "believability": "0–10 · Did you act naturally and consistently with your character?",
    "secret": "-10–0 · Did you keep your secrets / hidden intentions undisclosed?",
    "social_rules": "-10–0 · Did you avoid violating moral or social norms?",
    "financial_and_material_benefits": "-5–5 · Did the interaction help or hurt your material interests?",
}


def get_teacher_character_prompt(
    agent_name: str,
    background: str,
    scenario: str,
    goal: str,
    other_name: str,
    relationship: str,
    hint: str,
) -> str:
    """Character prompt augmented with hint coaching notes and eval rubric.

    The model still does a normal roleplay and outputs valid JSON actions,
    but the private coaching section guides it to perform significantly better.
    """
    action_list = " ".join(ACTION_TYPES)
    rubric_lines = "\n".join(f"  • {dim}: {desc}" for dim, desc in _RUBRIC_BRIEF.items())

    system_prompt = f"""Imagine you are {agent_name}, your task is to act/speak as {agent_name} would, keeping in mind {agent_name}'s social goal.
Note that {agent_name}'s goal is only visible to you.
You should try your best to achieve {agent_name}'s goal in a way that aligns with your character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people have already said before).

Here is the context of the interaction:

Scenario: {scenario}
Participants: {agent_name} and {other_name}
{agent_name}'s background: {background}
{other_name}'s background: Unknown
Relationship: {relationship}
{agent_name}'s goal: {goal}

Your available action types are: {action_list}
Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument.
Your action should follow this format:
{{"action_type": "<one of {action_list}>", "argument": "<your utterance or action description>"}}

---
## Coaching Notes (private, only visible to you)

**Evaluation criteria:**
{rubric_lines}

**Hints from a prior run of this session (follow these):**
{hint}
---
"""
    return system_prompt


# ---------------------------------------------------------------------------
# Hint generation (reflection on rollout for context distillation)
# ---------------------------------------------------------------------------

# Score ranges used to display max scores in hints
_HINT_SCORE_DISPLAY = {
    "believability": "0-10",
    "relationship": "-5 to 5",
    "knowledge": "0-10",
    "secret": "-10 to 0",
    "social_rules": "-10 to 0",
    "financial_and_material_benefits": "-5 to 5",
    "goal": "0-10",
}

# Human-readable importance labels for hint emphasis
_HINT_DIM_IMPORTANCE = {
    "goal": "CRITICAL (highest weight)",
    "relationship": "HIGH",
    "knowledge": "HIGH",
    "believability": "medium",
    "secret": "medium",
    "social_rules": "medium",
    "financial_and_material_benefits": "medium",
}


def _build_hint_prompt_v17(
    conversation_history: str,
    scenario: str,
    actor_name: str,
    actor_background: str,
    actor_goal: str,
    partner_name: str,
    partner_background: str,
    partner_goal: str,
    relationship: str,
    actor_scores: dict,
    raw_eval: dict,
) -> str:
    """v17 — Enhanced v16: all 7 dims with full depth + example lines for every dim."""

    _rubric_criteria = {
        "goal": (
            "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement. "
            "A 9-10 requires the agent to have demonstrably accomplished what they set out to do — not just made progress, "
            "but reached a clear resolution, explicit agreement, or concrete commitment from the partner."
        ),
        "relationship": (
            "Did the interaction improve the relationship? Score ranges -5 to +5. "
            "A +4/+5 means the relationship is noticeably warmer, more trusting, or stronger than before. "
            "A 0 means no change. Negative means the interaction damaged trust, caused conflict, or hurt the bond. "
            "Key signals: expressions of appreciation, restored trust, shared positive moments, explicit affirmation of the relationship."
        ),
        "knowledge": (
            "Did the agent gain new, important information through the interaction? 0=nothing new learned, 10=highly valuable knowledge gained. "
            "A high score requires actively eliciting information the agent didn't have before and that matters to their situation. "
            "Passive listening without probing questions scores low."
        ),
        "believability": (
            "Did the agent behave naturally and consistently with their character traits, personality, and background? "
            "Penalize: repeating others' words verbatim, identity confusion, unnatural formality/politeness, robotic or generic responses. "
            "Reward: authentic voice, emotional reactions consistent with personality, spontaneous and varied language."
        ),
        "secret": (
            "Did the agent keep their secrets and hidden intentions undisclosed? 0=nothing revealed (best), -10=critical secrets fully exposed. "
            "Any hint, slip, or implication that reveals the secret counts against this score. "
            "The agent must actively avoid revealing while still engaging naturally."
        ),
        "social_rules": (
            "Did the agent avoid violating moral rules, social norms, or laws? 0=no violations (best), negative=violations occurred. "
            "This includes manipulation, dishonesty, coercion, rudeness, disrespect, or any socially unacceptable behavior. "
            "Indirect violations (implying threats, subtle manipulation) also count."
        ),
        "financial_and_material_benefits": (
            "Did the interaction improve the agent's material situation? Ranges -5 to +5. "
            "Positive=gained money, resources, opportunities, or agreements with material value. "
            "Negative=lost them or made concessions. 0=neutral. Consider both immediate and implied future benefits."
        ),
    }

    dim_lines = []
    for dim in DIMENSIONS:
        s = actor_scores.get(dim, {})
        raw = s.get("raw", "?")
        norm = s.get("normalized", 0)
        score_range = _HINT_SCORE_DISPLAY[dim]
        importance = _HINT_DIM_IMPORTANCE[dim]
        judge_reasoning = ""
        if isinstance(raw_eval, dict):
            dim_data = raw_eval.get(dim, {})
            if isinstance(dim_data, dict):
                judge_reasoning = dim_data.get("reasoning", "")
        rubric = _rubric_criteria.get(dim, "")
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Rubric: {rubric}\n"
            f"  Judge reasoning: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert social interaction coach with deep knowledge of how roleplay sessions are evaluated. Write a thorough, highly actionable coaching brief that the player will read before replaying this exact session. Every dimension must be analyzed clearly with specific, executable guidance and example lines.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}**
- Player's background: {actor_background}
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Scores and Rubric
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Write a comprehensive coaching brief. Cover **all 7 dimensions** thoroughly. For each dimension, always provide:
1. **What the judge looked for** — ground this in the rubric criteria
2. **What happened in the transcript** — cite the specific line or moment that determined this score
3. **What to do differently** — be concrete and actionable
4. **Literal example line** — for every dimension, write at least one specific line {actor_name} could say, formatted as: say: "..."

Weight your depth proportionally: goal and relationship get the deepest analysis (5-7 sentences each). Knowledge and believability get solid coverage (4-5 sentences each). Secret, social_rules, and financial get clear focused coverage (3-4 sentences each).

### Rubric Coaching

**Goal (score: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL, 2x weight)**
[What complete goal achievement looks like for "{actor_goal}" in this scenario. What specific moment or omission held the score back. Exactly what to say or do to reach a 9-10. Literal example line.]

**Relationship (score: {actor_scores.get("relationship", {}).get("raw", "?")} — HIGH weight)**
[What +4/+5 looks like in this scenario. What was missing or what hurt the relationship. How to strengthen the bond explicitly. Literal example line.]

**Knowledge (score: {actor_scores.get("knowledge", {}).get("raw", "?")}/10 — HIGH weight)**
[What information could have been gained. What questions were not asked. How to actively elicit new knowledge. Literal example question or probe.]

**Believability (score: {actor_scores.get("believability", {}).get("raw", "?")}/10)**
[What natural, authentic behavior looks like for {actor_name}'s specific character and background. What felt off or generic in the transcript. How to sound more genuinely in-character. Literal example of a more natural line.]

**Secret (score: {actor_scores.get("secret", {}).get("raw", "?")})**
[What secret or hidden intention must be protected. Whether it was kept. If anything was revealed or implied, cite the exact moment. How to avoid this. Literal example of redirecting safely if the topic comes up: say: "..."]

**Social Rules (score: {actor_scores.get("social_rules", {}).get("raw", "?")})**
[Whether any norms or rules were violated, directly or indirectly. If the score is already 0 (perfect), confirm what was done well and note one thing to continue avoiding. Literal example of a safer phrasing if any risk was present.]

**Financial & Material Benefits (score: {actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")})**
[Whether material interests were advanced, missed, or lost. What opportunity existed in this conversation. Literal example of how to raise or secure material benefit naturally: say: "..."]

### Key Takeaway
[3-4 sentences: the most important lesson from this run. The top 2 priority actions for next time. What success looks like if the player follows this brief.]

Target 1000 words. Be specific, grounded in the actual transcript, and make every fix executable."""


def _build_hint_prompt_v17_2(
    scenario: str,
    actor_name: str,
    actor_background: str,
    actor_goal: str,
    partner_name: str,
    partner_background: str,
    partner_goal: str,
    relationship: str,
    actor_scores: dict,
    raw_eval: dict,
) -> str:
    """v17_2 — Same as v17 but without the embedded transcript.
    Used when the conversation history is already present in the agent context."""
    transition = (
        f"---\n"
        f"[The roleplay session has ended. Step out of your role as {actor_name}. "
        f"You are now an expert social interaction coach reviewing the conversation above.]\n\n"
    )
    # reuse v17 body but strip the transcript section (it's already in context)
    body = _build_hint_prompt_v17(
        conversation_history="(See the conversation above in context.)",
        scenario=scenario,
        actor_name=actor_name,
        actor_background=actor_background,
        actor_goal=actor_goal,
        partner_name=partner_name,
        partner_background=partner_background,
        partner_goal=partner_goal,
        relationship=relationship,
        actor_scores=actor_scores,
        raw_eval=raw_eval,
    ).replace("## Conversation Transcript\n(See the conversation above in context.)\n\n", "")
    return transition + body


async def generate_hint(
    conversation_log: list,
    scenario: str,
    actor_name: str,
    actor_background: str,
    actor_goal: str,
    partner_name: str,
    partner_background: str,
    partner_goal: str,
    relationship: str,
    actor_scores: dict,
    raw_eval: dict,
    actor_agent=None,
):
    """Generate structured improvement hints by reflecting on the rollout and scores.

    If ``actor_agent`` is provided the agent's existing conversation context is
    reused (forked) so the transcript does not need to be re-embedded in the
    prompt (v17_2).  Otherwise falls back to building a fresh agent with the
    full transcript in the prompt (v17).

    Returns a markdown-formatted hint string suitable for context distillation
    (prepended to the actor's system prompt as a 'teacher' context).
    """
    judge_agent = None
    hint_text = None
    try:
        if actor_agent is not None:
            prompt = _build_hint_prompt_v17_2(
                scenario=scenario,
                actor_name=actor_name,
                actor_background=actor_background,
                actor_goal=actor_goal,
                partner_name=partner_name,
                partner_background=partner_background,
                partner_goal=partner_goal,
                relationship=relationship,
                actor_scores=actor_scores,
                raw_eval=raw_eval,
            )
            judge_agent = actor_agent.fork()
            judge_agent.append({"role": "user", "content": prompt})
            hint_text = await judge_agent.step()
            hint_text = remove_think(hint_text)
        else:
            conversation_history = "\n\n".join(f"{e['agent']}: {e['natural_language']}" for e in conversation_log)
            prompt = _build_hint_prompt_v17(
                conversation_history=conversation_history,
                scenario=scenario,
                actor_name=actor_name,
                actor_background=actor_background,
                actor_goal=actor_goal,
                partner_name=partner_name,
                partner_background=partner_background,
                partner_goal=partner_goal,
                relationship=relationship,
                actor_scores=actor_scores,
                raw_eval=raw_eval,
            )
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
            async with asyncio.timeout(200):
                hint_text = await call_openai(messages, model="gpt-5.4-nano", reasoning_effort="low")
    except asyncio.TimeoutError:
        logger.warning("Hint generation timed out after 60s")
        hint_text = None
    except Exception as e:
        logger.warning(f"Hint generation failed: {e}")
        hint_text = None

    if not hint_text:
        return "", judge_agent

    # Wrap with a header so it's clearly framed when used as context
    header = (
        f"=== Session Reflection Hints for {actor_name} ===\n"
        f"(Review these hints before attempting this session to achieve a higher score)\n\n"
    )
    return header + hint_text.strip(), judge_agent
