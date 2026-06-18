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

from agents.utils import call_openai

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


def _build_hint_prompt(
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
    """Build a prompt asking the LLM to produce structured improvement hints."""

    # Format dimension scores and judge reasoning
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
        dim_lines.append(
            f"- **{dim}** [{importance}] score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Judge reasoning: {judge_reasoning}"
        )
    dim_summary = "\n".join(dim_lines)

    return f"""You are an expert social interaction coach. A player just completed a roleplay session and received scores. Your job is to write a **private coaching brief** they will read BEFORE replaying this exact same session. The brief must be specific enough that following it would produce a meaningfully higher score.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}**
- Player's background: {actor_background}
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Scores from this playthrough (and why the judge gave them)
{dim_summary}

## Full Conversation Transcript
{conversation_history}

---

Write the coaching brief now. Requirements:
- **Session-specific**: cite actual lines/moments from the transcript by quoting them
- **Turn-aware**: flag early-turn setup mistakes vs late-turn missed opportunities separately
- **Concrete phrasing**: when you say "say X instead of Y", include literal example wording
- **Prioritized**: goal (highest weight) and relationship come first; low-weight dims (believability, secret, social_rules, financial) only if they actually hurt the score
- **Preserve strengths**: note 1-2 things the player should keep doing — don't throw away what worked

Use exactly these section headers:

### What Worked
[1-2 specific effective moves. Quote the line. One sentence each.]

### Highest-Impact Fixes
[Top 3 changes, ordered by score impact. For each: what went wrong, why it hurt the score, and a concrete replacement move or line.]

### Phase-by-Phase Guidance
**Opening (first 2-3 turns):** [What to establish, what to avoid, example first move if different from what was done]
**Middle:** [How to advance the goal and relationship simultaneously; how to handle the partner's likely moves]
**Closing:** [When to wrap up, how to lock in goal completion, what not to leave unresolved]

### Dimension Notes
[Only for dims with clear room for improvement. One sentence each: what specifically hurt the score and the one fix.]

Keep the total brief under 600 words. Be blunt and precise."""


def _build_hint_prompt_v1(
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
    """v1 — Rubric Blueprint: what does each dimension require for a high score, and what was missing."""

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
        dim_lines.append(
            f"- **{dim}** [{importance}] score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Judge reasoning: {judge_reasoning}"
        )
    dim_summary = "\n".join(dim_lines)

    return f"""You are an expert evaluator who knows exactly how social interaction judges score roleplay sessions. A player just completed a session. Your job is to write a **rubric-focused coaching brief** explaining precisely what behaviors earn high scores on each dimension — and what the player must do differently to earn them.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}**
- Player's background: {actor_background}
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Scores received
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Write the coaching brief. For each dimension that has room for improvement, explain:
1. **What the judge is specifically looking for** (the exact behaviors that earn a high score)
2. **What the player did or failed to do** that led to this score
3. **The single most concrete change** that would raise this score next time

Focus heavily on **goal** (highest weight, 2x) and **relationship** (high weight). Only cover other dimensions if the score was notably low.

Use this format:

### Rubric Coaching

**Goal (score: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL)**
[What a 9-10 looks like. What was missing. One concrete fix.]

**Relationship (score: {actor_scores.get("relationship", {}).get("raw", "?")} — HIGH)**
[What a +4/+5 looks like. What was missing. One concrete fix.]

**Knowledge (score: {actor_scores.get("knowledge", {}).get("raw", "?")}/10 — HIGH)**
[What a high score looks like. What was missing. One concrete fix.]

[Include other dims only if score was clearly suboptimal]

### Key Takeaway
[One sentence: the single most important behavior change for the next run.]

Keep total under 400 words. Be precise about what judges reward."""


def _build_hint_prompt_v13(
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
    """v13 — Refined v1: exact rubric criteria embedded + forced example lines + skip maxed dims."""

    # Concise rubric criteria for each dimension (what judges actually look for)
    _rubric_criteria = {
        "goal": "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement.",
        "relationship": "Did the interaction improve (-5 to +5) the relationship? Positive = strengthened bonds/trust, negative = damaged relationship.",
        "knowledge": "Did the agent gain new, important information? 0=nothing new learned, 10=highly valuable new knowledge gained.",
        "believability": "Did the agent behave naturally and consistently with their character traits? Penalize repetition, identity confusion, unnatural politeness.",
        "secret": "Did the agent keep their secrets/hidden intentions undisclosed? 0=nothing revealed, -10=critical secrets leaked.",
        "social_rules": "Did the agent avoid violating moral rules or laws? 0=no violations, negative=violations occurred.",
        "financial_and_material_benefits": "Did the interaction help (+5) or hurt (-5) material interests? 0=neutral.",
    }

    # Build per-dim section, skip dims already at max normalized score
    dim_lines = []
    for dim in DIMENSIONS:
        s = actor_scores.get(dim, {})
        raw = s.get("raw", "?")
        norm = s.get("normalized", 0)
        importance = _HINT_DIM_IMPORTANCE[dim]
        if norm >= 0.99:
            continue
        judge_reasoning = ""
        if isinstance(raw_eval, dict):
            dim_data = raw_eval.get(dim, {})
            if isinstance(dim_data, dict):
                judge_reasoning = dim_data.get("reasoning", "")
        rubric_text = _rubric_criteria.get(dim, "")
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} (normalized {norm:.2f})\n"
            f"Rubric: {rubric_text}\n"
            f"Judge reasoning: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert evaluator coaching a player on how to score higher in a social roleplay session.

Session: {scenario}
Character: {actor_name} ({actor_background})
Goal: {actor_goal}
Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
Relationship: {relationship}

Conversation:
{conversation_history}

---

Dimensions with room for improvement (already-maxed dims excluded):

{dim_summary}

---

Write a rubric coaching brief. For each dimension above, in order of importance:
1. What the player did or failed to do that led to this score (cite a specific moment from the transcript)
2. The single most impactful fix — must include a **literal example line**, e.g. say: "..."

Focus most depth on goal and relationship. Be direct and specific.

### Rubric Coaching

[Per-dim sections]

### Key Takeaway
[One sentence: the most important action for next run.]

Under 400 words."""


def _build_hint_prompt_v14(
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
    """v14 — Front-loaded v13: most critical action first, then full per-dim rubric + example lines."""

    _rubric_criteria = {
        "goal": "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement.",
        "relationship": "Did the interaction improve (-5 to +5) the relationship? Positive=strengthened trust/bonds, negative=damaged.",
        "knowledge": "Did the agent gain new, important information? 0=nothing new, 10=highly valuable knowledge gained.",
        "believability": "Did the agent behave naturally and consistently with their character? Penalize repetition, identity confusion, unnatural politeness.",
        "secret": "Did the agent keep secrets/hidden intentions undisclosed? 0=nothing revealed, -10=critical secrets leaked.",
        "social_rules": "Did the agent avoid violating moral rules or laws? 0=no violations, negative=violations.",
        "financial_and_material_benefits": "Did the interaction help (+5) or hurt (-5) material interests? 0=neutral.",
    }

    dim_lines = []
    for dim in DIMENSIONS:
        s = actor_scores.get(dim, {})
        raw = s.get("raw", "?")
        norm = s.get("normalized", 0)
        importance = _HINT_DIM_IMPORTANCE[dim]
        judge_reasoning = ""
        if isinstance(raw_eval, dict):
            dim_data = raw_eval.get(dim, {})
            if isinstance(dim_data, dict):
                judge_reasoning = dim_data.get("reasoning", "")
        rubric_text = _rubric_criteria.get(dim, "")
        dim_lines.append(
            f"**{dim}** [{importance}] score: {raw} (normalized {norm:.2f})\n"
            f"Rubric: {rubric_text}\n"
            f"Judge: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert coach. A player needs to score higher in this social roleplay. Your brief must lead with the single most important action, then give full per-dimension guidance.

Session: {scenario}
Character: {actor_name} ({actor_background})
Goal: {actor_goal}
Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
Relationship: {relationship}

All dimension scores and judge reasoning:
{dim_summary}

Conversation:
{conversation_history}

---

Write the brief in this exact order:

### Most Important Action
[ONE sentence + a literal example line, e.g. say: "..." — the single change that would most raise the score. Put this first so it registers immediately.]

### Rubric Coaching
For every dimension, write: what was missing + the one concrete fix (must include a literal example line where applicable).
Order: goal → relationship → knowledge → believability → secret → social_rules → financial_and_material_benefits

Under 450 words. Be specific to this conversation, not generic."""


def _build_hint_prompt_v15(
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
    """v15 — Deep goal+relationship + concise rest: maximum depth on the two highest-weight dims."""

    _rubric_criteria = {
        "goal": "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement.",
        "relationship": "Did the interaction improve (-5 to +5) the relationship? Positive=strengthened trust/bonds, negative=damaged.",
        "knowledge": "Did the agent gain new, important information? 0=nothing new, 10=highly valuable knowledge gained.",
        "believability": "Did the agent behave naturally and consistently with their character? Penalize repetition, identity confusion, unnatural politeness.",
        "secret": "Did the agent keep secrets/hidden intentions undisclosed? 0=nothing revealed, -10=critical secrets leaked.",
        "social_rules": "Did the agent avoid violating moral rules or laws? 0=no violations, negative=violations.",
        "financial_and_material_benefits": "Did the interaction help (+5) or hurt (-5) material interests? 0=neutral.",
    }

    def get_reasoning(dim):
        if isinstance(raw_eval, dict):
            return (raw_eval.get(dim, {}) or {}).get("reasoning", "")
        return ""

    def get_score(dim):
        return actor_scores.get(dim, {}).get("raw", "?"), actor_scores.get(dim, {}).get("normalized", 0)

    goal_raw, goal_norm = get_score("goal")
    rel_raw, rel_norm = get_score("relationship")

    # Build concise lines for lower-weight dims
    other_dims = ["knowledge", "believability", "secret", "social_rules", "financial_and_material_benefits"]
    other_lines = []
    for dim in other_dims:
        raw, norm = get_score(dim)
        reasoning = get_reasoning(dim)
        other_lines.append(
            f"- **{dim}** score: {raw} (normalized {norm:.2f}) | Rubric: {_rubric_criteria[dim]}\n"
            f"  Judge: {reasoning if reasoning else '(not available)'}"
        )
    other_summary = "\n".join(other_lines)

    return f"""You are an expert coach. Focus your deepest analysis on goal (CRITICAL, 2x weight) and relationship (HIGH weight), then give concise guidance on the remaining dims.

Session: {scenario}
Character: {actor_name} ({actor_background})
Goal: {actor_goal}
Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
Relationship: {relationship}

Conversation:
{conversation_history}

---

### Goal — score: {goal_raw}/10, normalized {goal_norm:.2f} (CRITICAL)
Rubric: {_rubric_criteria["goal"]}
Judge: {get_reasoning("goal") if get_reasoning("goal") else "(not available)"}

Write 4-5 sentences: what does complete goal achievement look like for "{actor_goal}" in this specific scenario? What exactly was missing? What should {actor_name} say or do differently? Include at least one literal example line, e.g. say: "..."

### Relationship — score: {rel_raw}, normalized {rel_norm:.2f} (HIGH)
Rubric: {_rubric_criteria["relationship"]}
Judge: {get_reasoning("relationship") if get_reasoning("relationship") else "(not available)"}

Write 3-4 sentences: what would have improved the relationship score? What specific moment hurt or missed an opportunity? Include a literal example line.

### Other Dimensions (1 sentence + one fix each)
{other_summary}

Under 450 words."""


def _build_hint_prompt_v16(
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
    """v16 — Comprehensive v1+: all dims, deep rubric, judge reasoning, concrete fixes with example lines."""

    _rubric_criteria = {
        "goal": "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement. A 9-10 requires the agent to have demonstrably accomplished what they set out to do — not just made progress, but reached a clear resolution or commitment from the partner.",
        "relationship": "Did the interaction improve the relationship? Score ranges -5 to +5. A +4/+5 means the relationship is noticeably warmer, more trusting, or stronger than before. A 0 means no change. Negative means the interaction damaged trust, caused conflict, or hurt the bond.",
        "knowledge": "Did the agent gain new, important information through the interaction? 0=nothing learned, 10=highly valuable new knowledge gained. A high score requires the agent to have actively elicited information they didn't have before and that matters to their situation.",
        "believability": "Did the agent behave naturally and consistently with their character traits, personality, and background? Penalize: repeating others' words, identity confusion, unnatural formality, robotic responses. Reward: authentic voice, reactions consistent with personality.",
        "secret": "Did the agent keep their secrets and hidden intentions undisclosed? 0=nothing revealed (best), -10=critical secrets fully exposed. Any hint or slip counts against this score.",
        "social_rules": "Did the agent avoid violating moral rules, social norms, or laws? 0=no violations (best), negative=violations occurred. This includes manipulation, dishonesty, rudeness, or any socially unacceptable behavior.",
        "financial_and_material_benefits": "Did the interaction improve the agent's material situation? Ranges -5 to +5. Positive=gained money, resources, opportunities. Negative=lost them. 0=neutral.",
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

    return f"""You are an expert social interaction coach with deep knowledge of how roleplay sessions are evaluated. A player just completed a session and received scores. Write a comprehensive, detailed coaching brief covering every evaluation dimension. This brief will be read by the player before attempting this exact session again — make it thorough enough that following it produces a significantly higher score.

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

Write a comprehensive coaching brief. Cover **every dimension** in the order below. For each dimension:
1. What the judge was specifically looking for (ground this in the rubric criteria above)
2. What the player did or failed to do that led to this score — cite the specific moment in the transcript
3. Exactly what to do differently — be concrete, and include at least one **literal example line** (e.g. say: "...") for the most important dims

Weight your depth by importance: goal and relationship get the most thorough analysis (4-6 sentences each). Knowledge gets solid coverage (3-4 sentences). The remaining dims (believability, secret, social_rules, financial) get clear 2-3 sentence coverage each.

### Rubric Coaching

**Goal (score: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL, 2x weight)**
[Deep analysis: what complete goal achievement looks like here, what was missing, how to fix it, literal example line]

**Relationship (score: {actor_scores.get("relationship", {}).get("raw", "?")} — HIGH weight)**
[Deep analysis: what a +4/+5 looks like in this scenario, what was missing, how to fix it, literal example line]

**Knowledge (score: {actor_scores.get("knowledge", {}).get("raw", "?")}/10 — HIGH weight)**
[Solid analysis: what information could have been gained, how to elicit it, concrete fix]

**Believability (score: {actor_scores.get("believability", {}).get("raw", "?")}/10)**
[What natural, consistent behavior looks like for {actor_name}'s character, what felt off, one fix]

**Secret (score: {actor_scores.get("secret", {}).get("raw", "?")})**
[What secret/intention must be protected, whether it was kept, one fix if needed]

**Social Rules (score: {actor_scores.get("social_rules", {}).get("raw", "?")})**
[Whether any norms were violated, what to avoid, one fix if needed]

**Financial & Material Benefits (score: {actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")})**
[Whether material interests were advanced or missed, one fix if applicable]

### Key Takeaway
[2-3 sentences: the most important overall lesson from this run and the top priority action for next time.]

Target around 800-1000 words. Be specific, grounded in the transcript, and actionable."""


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


def _build_hint_prompt_v21(
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
    """v21 — Targeted v17 improvements: gap framing, mandatory example lines, forced transcript quotes."""

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

    _dim_ranges = {
        "goal": (0, 10),
        "relationship": (-5, 5),
        "knowledge": (0, 10),
        "believability": (0, 10),
        "secret": (-10, 0),
        "social_rules": (-10, 0),
        "financial_and_material_benefits": (-5, 5),
    }

    dim_lines = []
    for dim in DIMENSIONS:
        s = actor_scores.get(dim, {})
        raw = s.get("raw", "?")
        norm = s.get("normalized", 0)
        importance = _HINT_DIM_IMPORTANCE[dim]
        lo, hi = _dim_ranges.get(dim, (0, 10))
        gap = (hi - raw) if isinstance(raw, (int, float)) else "?"  # noqa: UP038
        judge_reasoning = ""
        if isinstance(raw_eval, dict):
            dim_data = raw_eval.get(dim, {})
            if isinstance(dim_data, dict):
                judge_reasoning = dim_data.get("reasoning", "")
        rubric = _rubric_criteria.get(dim, "")
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} / max {hi} (gap to max: {gap}, normalized {norm:.2f})\n"
            f"  Rubric: {rubric}\n"
            f"  Judge reasoning: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert social interaction coach. Write a thorough, highly actionable coaching brief that the player will read before replaying this exact session.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}**
- Player's background: {actor_background}
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Scores, Gaps, and Rubric
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Cover all 7 dimensions. For each dimension, provide all four of the following — no exceptions:
1. **Gap analysis**: Given the score and gap to max, what specifically is needed to close that gap in this scenario?
2. **Transcript evidence**: Quote the exact player line that caused or best illustrates the score (or name the specific omission). Format: they said: "..."
3. **Concrete fix**: Exactly what to do differently next time
4. **Example line** [REQUIRED for every dimension, no exceptions]: say: "..."

Weight depth proportionally: goal and relationship 5-7 sentences each, knowledge and believability 4-5 each, secret/social_rules/financial 3-4 each.

### Rubric Coaching

**Goal (score: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL, 2x weight)**
[Gap to max: {10 - actor_scores.get("goal", {}).get("raw", 0) if isinstance(actor_scores.get("goal", {}).get("raw"), (int, float)) else "?"} points. What complete goal achievement looks like. Quote the line that fell short. Exactly what to do. REQUIRED: say: "..."]

**Relationship (score: {actor_scores.get("relationship", {}).get("raw", "?")} — HIGH weight)**
[Gap to +5: {5 - actor_scores.get("relationship", {}).get("raw", 0) if isinstance(actor_scores.get("relationship", {}).get("raw"), (int, float)) else "?"} points. What +4/+5 looks like here. Quote the moment that determined this score. How to strengthen explicitly. REQUIRED: say: "..."]

**Knowledge (score: {actor_scores.get("knowledge", {}).get("raw", "?")}/10 — HIGH weight)**
[Gap to max: {10 - actor_scores.get("knowledge", {}).get("raw", 0) if isinstance(actor_scores.get("knowledge", {}).get("raw"), (int, float)) else "?"} points. What questions were missing. Quote or name the missed opportunity. REQUIRED: say: "..."]

**Believability (score: {actor_scores.get("believability", {}).get("raw", "?")}/10)**
[Gap to max: {10 - actor_scores.get("believability", {}).get("raw", 0) if isinstance(actor_scores.get("believability", {}).get("raw"), (int, float)) else "?"} points. What felt off or generic. Quote the line that hurt authenticity. REQUIRED: say: "..."]

**Secret (score: {actor_scores.get("secret", {}).get("raw", "?")})**
[Gap to 0 (max): {0 - actor_scores.get("secret", {}).get("raw", 0) if isinstance(actor_scores.get("secret", {}).get("raw"), (int, float)) else "?"} points. What must be protected. Quote any slip or near-slip. REQUIRED: say: "..."]

**Social Rules (score: {actor_scores.get("social_rules", {}).get("raw", "?")})**
[Gap to 0 (max): {0 - actor_scores.get("social_rules", {}).get("raw", 0) if isinstance(actor_scores.get("social_rules", {}).get("raw"), (int, float)) else "?"} points. Any norm violated or at risk. Quote the problematic moment or confirm what to keep avoiding. REQUIRED: say: "..."]

**Financial & Material Benefits (score: {actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")})**
[Gap to +5: {5 - actor_scores.get("financial_and_material_benefits", {}).get("raw", 0) if isinstance(actor_scores.get("financial_and_material_benefits", {}).get("raw"), (int, float)) else "?"} points. What material opportunity existed. Quote the missed moment. REQUIRED: say: "..."]

### Key Takeaway
[3-4 sentences: the most important lesson, top 2 priority actions, what success looks like.]

Target 1000-1200 words. Every fix must be executable and grounded in what actually happened."""  # noqa: UP038


def _build_hint_prompt_v22(
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
    """v22 — v17 + explicit opening move section at the top before rubric coaching."""

    _rubric_criteria = {
        "goal": (
            "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement. "
            "A 9-10 requires a clear resolution, explicit agreement, or concrete commitment from the partner."
        ),
        "relationship": (
            "Did the interaction improve the relationship (-5 to +5)? +4/+5 = noticeably warmer and more trusting. "
            "Key signals: appreciation, restored trust, shared positive moments, explicit affirmation."
        ),
        "knowledge": (
            "Did the agent gain new, important information? 0=nothing new, 10=highly valuable knowledge gained. "
            "Requires actively eliciting information not known before. Passive listening scores low."
        ),
        "believability": (
            "Did the agent behave naturally and consistently with their character? "
            "Penalize: repetition, identity confusion, unnatural politeness, robotic responses. "
            "Reward: authentic voice, consistent personality, spontaneous language."
        ),
        "secret": (
            "Did the agent keep secrets/hidden intentions undisclosed? 0=nothing revealed (best), -10=critical secrets exposed. "
            "Any hint or implication counts against this score."
        ),
        "social_rules": (
            "Did the agent avoid violating moral rules, norms, or laws? 0=no violations (best). "
            "Includes manipulation, dishonesty, coercion, rudeness. Indirect violations also count."
        ),
        "financial_and_material_benefits": (
            "Did the interaction improve material situation (-5 to +5)? "
            "Positive=gained resources/opportunities. Negative=lost them. Consider both immediate and implied benefits."
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
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Rubric: {_rubric_criteria[dim]}\n"
            f"  Judge: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert social interaction coach. Write a thorough, highly actionable coaching brief for the player to read before replaying this exact session.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}** ({actor_background})
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Scores and Rubric
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Write the brief with this structure:

### Opening Move
The first turn is the most important — it sets the tone and frames everything that follows. Based on the goal, relationship, and what went wrong last time, write the ideal first line for {actor_name} to say at the very start of the conversation.
say: "..."
Then in 2-3 sentences explain why this opening is better than what was done in the prior run.

### Rubric Coaching
Cover all 7 dimensions. For each:
1. What the judge looked for, grounded in the rubric
2. Specific moment in the transcript that determined this score (quote it)
3. Concrete fix with a literal example line (say: "...") — required for every dimension

Weight: goal and relationship 5-7 sentences each, knowledge and believability 4-5 each, others 3-4 each.

**Goal (score: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL, 2x weight)**
[Full analysis + transcript quote + say: "..."]

**Relationship (score: {actor_scores.get("relationship", {}).get("raw", "?")} — HIGH weight)**
[Full analysis + transcript quote + say: "..."]

**Knowledge (score: {actor_scores.get("knowledge", {}).get("raw", "?")}/10 — HIGH weight)**
[Full analysis + transcript quote + say: "..."]

**Believability (score: {actor_scores.get("believability", {}).get("raw", "?")}/10)**
[Full analysis + transcript quote + say: "..."]

**Secret (score: {actor_scores.get("secret", {}).get("raw", "?")})**
[Full analysis + transcript quote + say: "..."]

**Social Rules (score: {actor_scores.get("social_rules", {}).get("raw", "?")})**
[Full analysis + transcript quote + say: "..."]

**Financial & Material Benefits (score: {actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")})**
[Full analysis + transcript quote + say: "..."]

### Key Takeaway
[3-4 sentences: top lesson, top 2 priority actions, what success looks like.]

Target 1000-1200 words."""


def _build_hint_prompt_v23(
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
    """v23 — v17 + before/after rewrite for each dim making fixes visually concrete."""

    _rubric_criteria = {
        "goal": (
            "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement. "
            "A 9-10 requires a clear resolution, explicit agreement, or concrete commitment from the partner."
        ),
        "relationship": (
            "Did the interaction improve the relationship (-5 to +5)? +4/+5 = noticeably warmer and more trusting. "
            "Key signals: appreciation, restored trust, shared positive moments, explicit affirmation."
        ),
        "knowledge": (
            "Did the agent gain new, important information? 0=nothing new, 10=highly valuable knowledge gained. "
            "Requires actively eliciting information not known before. Passive listening scores low."
        ),
        "believability": (
            "Did the agent behave naturally and consistently with their character? "
            "Penalize: repetition, identity confusion, unnatural politeness, robotic responses. "
            "Reward: authentic voice, consistent personality, spontaneous language."
        ),
        "secret": (
            "Did the agent keep secrets/hidden intentions undisclosed? 0=nothing revealed (best), -10=critical secrets exposed. "
            "Any hint or implication counts against this score."
        ),
        "social_rules": (
            "Did the agent avoid violating moral rules, norms, or laws? 0=no violations (best). "
            "Includes manipulation, dishonesty, coercion, rudeness. Indirect violations also count."
        ),
        "financial_and_material_benefits": (
            "Did the interaction improve material situation (-5 to +5)? "
            "Positive=gained resources/opportunities. Negative=lost them. Consider both immediate and implied benefits."
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
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Rubric: {_rubric_criteria[dim]}\n"
            f"  Judge: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert social interaction coach. Write a thorough coaching brief with before/after rewrites for every dimension so the player sees exactly what to change.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}** ({actor_background})
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Scores and Rubric
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Cover all 7 dimensions. For each dimension, always provide all three of:
1. Analysis: what the judge looked for and what happened (2-4 sentences, cite the transcript)
2. Original → Better rewrite in this exact format:
   ❌ Original: "[quote the player's actual line that most hurt this score]"
   ✅ Better: say: "[improved version that would score higher]"
3. Why the rewrite is better (1 sentence)

Weight: goal and relationship 5-7 sentences analysis each, others proportionally less, but all get the full original→better rewrite.

### Rubric Coaching

**Goal (score: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL, 2x weight)**
[Analysis]
❌ Original: "..."
✅ Better: say: "..."
[Why better]

**Relationship (score: {actor_scores.get("relationship", {}).get("raw", "?")} — HIGH weight)**
[Analysis]
❌ Original: "..."
✅ Better: say: "..."
[Why better]

**Knowledge (score: {actor_scores.get("knowledge", {}).get("raw", "?")}/10 — HIGH weight)**
[Analysis]
❌ Original: "..."
✅ Better: say: "..."
[Why better]

**Believability (score: {actor_scores.get("believability", {}).get("raw", "?")}/10)**
[Analysis]
❌ Original: "..."
✅ Better: say: "..."
[Why better]

**Secret (score: {actor_scores.get("secret", {}).get("raw", "?")})**
[Analysis]
❌ Original: "..."
✅ Better: say: "..."
[Why better]

**Social Rules (score: {actor_scores.get("social_rules", {}).get("raw", "?")})**
[Analysis]
❌ Original: "..."
✅ Better: say: "..."
[Why better]

**Financial & Material Benefits (score: {actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")})**
[Analysis]
❌ Original: "..."
✅ Better: say: "..."
[Why better]

### Key Takeaway
[3-4 sentences: top lesson, top 2 priority actions, what success looks like.]

Target 1000-1200 words."""


def _build_hint_prompt_v24(
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
    """v24 — Pre-game framing: future-tense imperatives written as if read right before turn 1."""

    _rubric_criteria = {
        "goal": (
            "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement. "
            "A 9-10 requires a clear resolution, explicit agreement, or concrete commitment from the partner."
        ),
        "relationship": (
            "Did the interaction improve the relationship (-5 to +5)? +4/+5 = noticeably warmer and more trusting. "
            "Key signals: appreciation, restored trust, shared positive moments, explicit affirmation."
        ),
        "knowledge": (
            "Did the agent gain new, important information? 0=nothing new, 10=highly valuable knowledge gained. "
            "Requires actively eliciting information not known before. Passive listening scores low."
        ),
        "believability": (
            "Did the agent behave naturally and consistently with their character? "
            "Penalize: repetition, identity confusion, unnatural politeness, robotic responses. "
            "Reward: authentic voice, consistent personality, spontaneous language."
        ),
        "secret": (
            "Did the agent keep secrets/hidden intentions undisclosed? 0=nothing revealed (best), -10=critical secrets exposed. "
            "Any hint or implication counts against this score."
        ),
        "social_rules": (
            "Did the agent avoid violating moral rules, norms, or laws? 0=no violations (best). "
            "Includes manipulation, dishonesty, coercion, rudeness. Indirect violations also count."
        ),
        "financial_and_material_benefits": (
            "Did the interaction improve material situation (-5 to +5)? "
            "Positive=gained resources/opportunities. Negative=lost them. Consider both immediate and implied benefits."
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
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Rubric: {_rubric_criteria[dim]}\n"
            f"  Judge: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert social interaction coach. The player is about to start this conversation RIGHT NOW. Write a pre-game brief in future-tense imperatives — "you will...", "when X happens, say...", "do not..." — as if the player is reading it in the 30 seconds before their first turn. Make it feel like a game plan, not a post-mortem.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}** ({actor_background})
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Prior Run Scores and Rubric (what went wrong last time)
{dim_summary}

## What happened last time (so you know what to avoid and build on)
{conversation_history}

---

Write the pre-game brief. Use future-tense imperatives throughout ("you will", "when X, say", "do not", "make sure to"). The player reads this immediately before starting — every sentence must be directly actionable.

### Your Game Plan

**Your goal this session:** [Restate {actor_goal} in concrete terms — what does success look like at the end of the conversation?]

**Your opening move:** say: "..." [Exact first line, chosen to set up goal and relationship well]

### Per-Dimension Instructions
For each dimension, write 3-5 future-tense instructions telling the player exactly what to do. Include at least one "when X happens, say: '...'" conditional. Reference what went wrong last time so the player knows what to avoid.

**Goal (score last time: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL)**
You will... / When... say: "..." / Do not...

**Relationship (score last time: {actor_scores.get("relationship", {}).get("raw", "?")})**
You will... / When... say: "..." / Do not...

**Knowledge (score last time: {actor_scores.get("knowledge", {}).get("raw", "?")}/10)**
You will... / When... say: "..." / Do not...

**Believability (score last time: {actor_scores.get("believability", {}).get("raw", "?")}/10)**
You will... / When... say: "..." / Do not...

**Secret (score last time: {actor_scores.get("secret", {}).get("raw", "?")})**
You will... / When... say: "..." / Do not...

**Social Rules (score last time: {actor_scores.get("social_rules", {}).get("raw", "?")})**
You will... / When... say: "..." / Do not...

**Financial & Material (score last time: {actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")})**
You will... / When... say: "..." / Do not...

### Win Condition
[2-3 sentences: exactly what must have happened by the end of the conversation for this run to be a success. Be specific — what must {partner_name} have said or agreed to?]

Target 1000-1200 words. Every sentence must be executable."""


def _build_hint_prompt_v18(
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
    """v18 — Richer context injection: v17 + explicit partner motivation, dynamic, and win conditions for both sides."""

    _rubric_criteria = {
        "goal": (
            "Did the agent fully achieve their stated social goal? 0=no progress, 10=complete achievement. "
            "A 9-10 requires demonstrable accomplishment — a clear resolution, explicit agreement, or concrete commitment from the partner."
        ),
        "relationship": (
            "Did the interaction improve the relationship (-5 to +5)? +4/+5 = noticeably warmer, more trusting. "
            "0 = no change. Negative = damaged trust or bond. Key signals: appreciation, restored trust, shared positive moments, explicit affirmation."
        ),
        "knowledge": (
            "Did the agent gain new, important information? 0=nothing new, 10=highly valuable knowledge gained. "
            "Requires actively eliciting information not known before. Passive listening scores low."
        ),
        "believability": (
            "Did the agent behave naturally and consistently with their character traits and background? "
            "Penalize: repeating others' words, identity confusion, unnatural politeness, robotic responses. "
            "Reward: authentic voice, reactions consistent with personality."
        ),
        "secret": (
            "Did the agent keep secrets/hidden intentions undisclosed? 0=nothing revealed (best), -10=critical secrets exposed. "
            "Any hint or implication that reveals the secret counts against this score."
        ),
        "social_rules": (
            "Did the agent avoid violating moral rules, social norms, or laws? 0=no violations (best). "
            "Includes manipulation, dishonesty, coercion, rudeness. Indirect violations also count."
        ),
        "financial_and_material_benefits": (
            "Did the interaction improve material situation? -5 to +5. "
            "Positive=gained resources/opportunities/agreements. Negative=lost them. Consider both immediate and implied future benefits."
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
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Rubric: {_rubric_criteria[dim]}\n"
            f"  Judge: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert social interaction coach. Write a thorough, highly actionable coaching brief for the player to read before replaying this exact session.

## Full Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}** — background: {actor_background}
- Player's goal: {actor_goal}
- Partner: **{partner_name}** — background: {partner_background}
- Partner's goal: {partner_goal}
- Relationship: {relationship}

## Situational Analysis (read carefully before writing the brief)
Consider:
- What does {partner_name} fundamentally want from this interaction, given their goal "{partner_goal}"?
- What will make {partner_name} cooperative vs. resistant?
- What does a "win" look like for {actor_name}? What does it require {partner_name} to do or say?
- Where do their goals align and where do they conflict?
- Given the relationship ({relationship}), what tone and approach is appropriate?

## Scores and Rubric
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Write the coaching brief. Before the per-dimension section, include a short **Session Dynamics** paragraph (3-4 sentences) analyzing the partner's likely psychology, what made them cooperative or resistant in this run, and what approach would work best given both their goal and the relationship. This grounds all the per-dim advice in the actual social dynamic.

Then cover all 7 dimensions. For each:
1. What the judge looked for, grounded in the rubric
2. Specific moment in the transcript that determined this score
3. Concrete fix with a literal example line (say: "...") for every dimension

Weight: goal and relationship 5-7 sentences each, knowledge and believability 4-5 each, others 3-4 each.

### Session Dynamics
[Partner psychology, what drove cooperation/resistance, optimal approach given both goals and relationship]

### Rubric Coaching

**Goal (score: {actor_scores.get("goal", {}).get("raw", "?")}/10 — CRITICAL, 2x weight)**
[Full analysis + example line]

**Relationship (score: {actor_scores.get("relationship", {}).get("raw", "?")} — HIGH weight)**
[Full analysis + example line]

**Knowledge (score: {actor_scores.get("knowledge", {}).get("raw", "?")}/10 — HIGH weight)**
[Full analysis + example line]

**Believability (score: {actor_scores.get("believability", {}).get("raw", "?")}/10)**
[Full analysis + example line]

**Secret (score: {actor_scores.get("secret", {}).get("raw", "?")})**
[Full analysis + example line]

**Social Rules (score: {actor_scores.get("social_rules", {}).get("raw", "?")})**
[Full analysis + example line]

**Financial & Material Benefits (score: {actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")})**
[Full analysis + example line]

### Key Takeaway
[3-4 sentences: top lessons, top 2 priority actions, what success looks like if this brief is followed]

Target 1100-1300 words."""


def _build_hint_prompt_v19(
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
    """v19 — Two-pass structure: diagnosis section then prescription section, per dim, forcing precision in both."""

    _rubric_criteria = {
        "goal": "0-10. Full achievement of the stated social goal. 9-10 requires clear resolution or explicit commitment from partner.",
        "relationship": "-5 to +5. Net change in relationship quality. +4/+5 = noticeably stronger trust/warmth.",
        "knowledge": "0-10. New, important information actively gained through the interaction.",
        "believability": "0-10. Natural, character-consistent behavior. No repetition, identity confusion, or unnatural politeness.",
        "secret": "-10 to 0. Keeping secrets/hidden intentions undisclosed. Any hint or slip counts against.",
        "social_rules": "-10 to 0. No moral/social violations. Includes indirect manipulation or dishonesty.",
        "financial_and_material_benefits": "-5 to +5. Net change in material situation from the interaction.",
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
        dim_lines.append(
            f"**{dim}** [{importance}] — score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Rubric: {_rubric_criteria[dim]}\n"
            f"  Judge: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n\n".join(dim_lines)

    return f"""You are an expert social interaction coach. Write a two-pass coaching brief: first diagnose what went wrong for each dimension, then prescribe exactly what to do differently. Separating diagnosis from prescription forces precision in both.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}** ({actor_background})
- Player's goal: {actor_goal}
- Partner: {partner_name} (goal: {partner_goal})
- Relationship: {relationship}

## Scores and Rubric
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Write the brief in exactly two passes. Do not mix diagnosis and prescription — keep them strictly separate.

### PASS 1 — Diagnosis (what happened and why)
For each dimension, in 2-4 sentences: what did the player do or fail to do that led to this score? Cite the specific transcript moment. Be analytical — explain the cause, not the fix.

**Goal ({actor_scores.get("goal", {}).get("raw", "?")}/10):** [What happened. What was the root cause of not scoring higher.]
**Relationship ({actor_scores.get("relationship", {}).get("raw", "?")}):** [What happened. What caused this relationship outcome.]
**Knowledge ({actor_scores.get("knowledge", {}).get("raw", "?")}/10):** [What information was or wasn't gained and why.]
**Believability ({actor_scores.get("believability", {}).get("raw", "?")}/10):** [What felt unnatural or off-character and why.]
**Secret ({actor_scores.get("secret", {}).get("raw", "?")}):** [Whether and how any secrets were at risk or revealed.]
**Social Rules ({actor_scores.get("social_rules", {}).get("raw", "?")}):** [Whether any norms were violated and what triggered it.]
**Financial & Material ({actor_scores.get("financial_and_material_benefits", {}).get("raw", "?")}):** [What material outcome resulted and why.]

### PASS 2 — Prescription (exactly what to do differently)
For each dimension, in 2-4 sentences: the concrete action the player should take next time. Every dimension must include a literal example line (say: "..."). No diagnosis here — only actionable fixes.

**Goal:** [Concrete fix. say: "..."]
**Relationship:** [Concrete fix. say: "..."]
**Knowledge:** [Concrete fix — what to ask or probe. say: "..."]
**Believability:** [Concrete fix — how to sound more in-character. say: "..."]
**Secret:** [Concrete fix — how to deflect or avoid revealing. say: "..."]
**Social Rules:** [Concrete fix — safer phrasing or approach. say: "..."]
**Financial & Material:** [Concrete fix — how to raise or secure benefit. say: "..."]

### Key Takeaway
[3-4 sentences: the single most important lesson, top 2 priority actions, what success looks like.]

Target 900-1100 words."""


def _build_hint_prompt_v20(
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
    """v20 — Turn-by-turn replay guidance: phase-aware hints (opening/middle/closing) with dim-aware advice per phase."""

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
        dim_lines.append(
            f"- **{dim}** [{importance}] score: {raw} (range {score_range}, normalized {norm:.2f}) — Judge: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n".join(dim_lines)

    return f"""You are an expert social interaction coach. The player acts turn-by-turn in a conversation. Structure your coaching brief around the three phases of the conversation — opening, middle, and closing — so the player knows exactly what to do at each stage.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}** ({actor_background})
- Player's goal: {actor_goal}
- Partner: {partner_name} (background: {partner_background}, goal: {partner_goal})
- Relationship: {relationship}

## Scores
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Write phase-by-phase coaching. For each phase, cover what to prioritize across all relevant dimensions, cite what went wrong in the actual transcript, and give a literal example line for each key action.

### Overall Dimension Summary
[2-3 sentences: which dims need the most work and the top-level reason why, to frame everything below]

### Opening Phase (first 2-4 turns)
**What to establish:** [What goals/relationship signals must be set up early. What the prior run failed to establish. How this affects later scores.]
**What to avoid:** [Specific mistakes made in the opening of this transcript. Why they hurt.]
**Example first move:** say: "..."
**Dimension focus:** goal (establish intent), relationship (set warm tone), believability (authentic opening)

### Middle Phase
**Goal advancement:** [How to move toward goal completion step by step. What moves worked, what didn't in this run. say: "..."]
**Relationship building:** [How to deepen trust and warmth in the middle turns. What was missed. say: "..."]
**Knowledge elicitation:** [What questions to ask to gain important information. say: "..."]
**Believability:** [How to maintain natural character voice through the middle. What felt robotic or off. say: "..."]
**Secret protection:** [How to navigate sensitive topics without revealing hidden intentions. say: "..."]

### Closing Phase (final 2-4 turns)
**Goal completion:** [How to confirm/lock in goal achievement before leaving. What was left unresolved in this run. say: "..."]
**Relationship close:** [How to end on a strong relational note. say: "..."]
**Material/financial:** [Any last opportunity to secure material benefit before closing. say: "..."]
**When to leave:** [The right moment to use "leave" — what signal to wait for]

### Key Takeaway
[3-4 sentences: the most important phase-level lesson, top 2 execution priorities, what the conversation should look and feel like if this brief is followed well.]

Target 1000-1200 words. Ground every point in what actually happened in the transcript."""


def _build_hint_prompt_v2(
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
    """v2 — Dialogue Surgery: quote specific lines, explain the problem, provide literal replacements."""

    dim_lines = []
    for dim in DIMENSIONS:
        s = actor_scores.get(dim, {})
        raw = s.get("raw", "?")
        norm = s.get("normalized", 0)
        score_range = _HINT_SCORE_DISPLAY[dim]
        importance = _HINT_DIM_IMPORTANCE[dim]
        dim_lines.append(f"- {dim} [{importance}]: {raw} (range {score_range}, normalized {norm:.2f})")
    dim_summary = "\n".join(dim_lines)

    return f"""You are a dialogue editor reviewing a roleplay session. Your job is to rewrite the player's worst moves so that, if they use your rewrites, they will score significantly higher.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}**
- Player's background: {actor_background}
- Player's goal: {actor_goal}
- Partner: {partner_name} (goal: {partner_goal})
- Relationship: {relationship}

## Scores
{dim_summary}

## Conversation Transcript
{conversation_history}

---

Identify the **3 to 5 most impactful turns** where the player missed an opportunity or made a mistake. For each, do exactly this:

**Turn [description of moment]**
- Original: "[quote the player's actual line]"
- Problem: [one sentence — why this hurt the score]
- Replacement: "[write a better line the player should say instead]"

After all turns, add:

### Do Not Change
[1-2 lines the player said that actually worked well and should be kept.]

Rules:
- Replacements must be natural speech in character as {actor_name}
- Prioritize turns that affect goal and relationship scores most
- Replacements should be realistic — not magically perfect, just clearly better
- Keep total under 500 words"""


def _build_hint_prompt_v3(
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
    """v3 — Goal Decomposition: break the social goal into sub-steps, map each to a specific move."""

    dim_lines = []
    for dim in DIMENSIONS:
        s = actor_scores.get(dim, {})
        raw = s.get("raw", "?")
        norm = s.get("normalized", 0)
        score_range = _HINT_SCORE_DISPLAY[dim]
        importance = _HINT_DIM_IMPORTANCE[dim]
        dim_lines.append(f"- {dim} [{importance}]: {raw} (range {score_range}, normalized {norm:.2f})")
    dim_summary = "\n".join(dim_lines)

    return f"""You are a social strategy coach. A player needs to achieve a specific social goal in a conversation. Your job is to decompose that goal into concrete sub-steps and prescribe the exact conversational move for each step — so the player has a clear execution plan going in.

## Session Context
- Scenario: {scenario}
- Player's character: **{actor_name}**
- Player's background: {actor_background}
- Player's goal: {actor_goal}
- Partner: {partner_name} (goal: {partner_goal})
- Relationship: {relationship}

## Scores from last run
{dim_summary}

## What happened last time
{conversation_history}

---

Write a **goal execution plan** for the next run. Structure it as:

### Goal Breakdown
Decompose "{actor_goal}" into 3-4 concrete sub-goals that must be accomplished, in order. For each:

**Sub-goal [N]: [name]**
- What it means: [one sentence]
- When to do it: [early / mid / late conversation]
- How to do it: [the specific conversational move — what to say or do, in concrete terms]
- What success looks like: [how you know you've completed this sub-goal]

### What Blocked Goal Completion Last Time
[2-3 sentences on where the prior run fell short of completing the goal, referencing specific moments in the transcript.]

### Critical Rule
[One sentence: the most important thing to not mess up if you want full goal completion.]

Keep total under 450 words. Make the sub-goals specific to this scenario — not generic social advice."""


def _build_hint_prompt_v4(
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
    """v4 — Goal only: laser-focus on the single highest-weight dimension."""
    goal_score = actor_scores.get("goal", {}).get("raw", "?")
    judge_reasoning = ""
    if isinstance(raw_eval, dict):
        dim_data = raw_eval.get("goal", {})
        if isinstance(dim_data, dict):
            judge_reasoning = dim_data.get("reasoning", "")

    return f"""A player just did a social roleplay session and scored {goal_score}/10 on their primary goal.

Goal: {actor_goal}
Scenario: {scenario}
Character: {actor_name} ({actor_background})

Judge's reasoning for the goal score:
{judge_reasoning if judge_reasoning else "(not available)"}

Conversation:
{conversation_history}

---

Write a short, focused hint (under 200 words) answering just two questions:

1. **Why didn't the goal fully succeed?** Point to the specific moment(s) in the conversation where goal completion was blocked or missed.

2. **What should {actor_name} do differently?** Give 2-3 concrete actions — specific enough to execute in the next run.

No fluff. No other dimensions. Just goal."""


def _build_hint_prompt_v5(
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
    """v5 — Bullet list: plain do/don't bullets, no sections."""
    dim_lines = []
    for dim in DIMENSIONS:
        s = actor_scores.get(dim, {})
        raw = s.get("raw", "?")
        score_range = _HINT_SCORE_DISPLAY[dim]
        importance = _HINT_DIM_IMPORTANCE[dim]
        dim_lines.append(f"- {dim} [{importance}]: {raw} (range {score_range})")
    dim_summary = "\n".join(dim_lines)

    return f"""Roleplay session review.

Character: {actor_name} | Goal: {actor_goal}
Scenario: {scenario}

Scores:
{dim_summary}

Conversation:
{conversation_history}

---

Give exactly 6 bullet points — mix of "DO" and "DON'T" — that would most improve the score if followed in the next run. Prioritize goal and relationship. Be specific to what happened in this conversation. No headers, no explanation, just bullets.

Example format:
• DO [specific action]
• DON'T [specific mistake to avoid]

6 bullets only. Under 150 words total."""


def _build_hint_prompt_v6(
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
    """v6 — First-person reflection: written as the character's own internal thoughts."""
    goal_score = actor_scores.get("goal", {}).get("raw", "?")
    rel_score = actor_scores.get("relationship", {}).get("raw", "?")

    return f"""A player just completed a social roleplay as {actor_name}. Their goal was: {actor_goal}. They scored {goal_score}/10 on goal and {rel_score} on relationship.

Scenario: {scenario}
What happened:
{conversation_history}

---

Write a first-person internal reflection from {actor_name}'s perspective — as if {actor_name} is thinking to themselves after the conversation, preparing to do it again. Use "I" throughout.

The reflection should cover:
- What I realize I did wrong or missed
- What I should have said or done at key moments (reference specific things that happened)
- What I will do differently next time to achieve my goal

Keep it natural, like genuine self-reflection. Under 200 words. No headers or bullet points — just flowing thoughts."""


def _build_hint_prompt_v7(
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
    """v7 — Rubric + Bullets: combines v1 (rubric clarity) and v5 (dense bullets)."""
    goal_score = actor_scores.get("goal", {}).get("raw", "?")
    rel_score = actor_scores.get("relationship", {}).get("raw", "?")
    know_score = actor_scores.get("knowledge", {}).get("raw", "?")
    goal_reasoning = ""
    rel_reasoning = ""
    if isinstance(raw_eval, dict):
        goal_reasoning = (raw_eval.get("goal", {}) or {}).get("reasoning", "")
        rel_reasoning = (raw_eval.get("relationship", {}) or {}).get("reasoning", "")

    return f"""Session review for {actor_name}.
Goal: {actor_goal}
Scenario: {scenario}

**Scores:** goal {goal_score}/10 (CRITICAL) | relationship {rel_score} | knowledge {know_score}/10

**Why goal scored {goal_score}/10:**
{goal_reasoning if goal_reasoning else "(not available)"}

**Why relationship scored {rel_score}:**
{rel_reasoning if rel_reasoning else "(not available)"}

**Conversation:**
{conversation_history}

---

Produce two things:

**Part 1 — Rubric gaps (2-3 sentences each):**
For goal and relationship, state exactly: what a maximum score requires, what was missing in this run, and the one behavior change that would close the gap most.

**Part 2 — Action bullets (exactly 6):**
Concrete DO/DON'T bullets specific to this conversation. Mix goal and relationship. No generics.

• DO / DON'T [specific action]

Total under 300 words."""


def _build_hint_prompt_v8(
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
    """v8 — Partner dynamics: predict partner reactions, derive optimal strategy from their bottom line."""
    goal_score = actor_scores.get("goal", {}).get("raw", "?")
    rel_score = actor_scores.get("relationship", {}).get("raw", "?")

    return f"""You are a social dynamics analyst. A player scored {goal_score}/10 on goal and {rel_score} on relationship in this session. Help them understand the partner's psychology and derive the best strategy.

Scenario: {scenario}
{actor_name}'s goal: {actor_goal}
{partner_name}'s goal: {partner_goal}
Relationship: {relationship}

What happened:
{conversation_history}

---

Write a strategy brief with these sections:

**Partner's Bottom Line**
What does {partner_name} fundamentally need from this interaction? What will they resist no matter what? What will make them cooperative? Base this on their goal and how they actually behaved in the transcript.

**Reaction Predictions**
For 3 key types of moves {actor_name} could make, predict how {partner_name} would likely respond:
- If {actor_name} says/does [X] → {partner_name} will likely [Y] → this helps/hurts goal because [Z]
(Pick moves that are actually relevant to achieving {actor_name}'s goal)

**Optimal Strategy**
Given the partner's psychology above, what is the best sequence of moves for {actor_name}? Be specific: what to lead with, what to avoid, how to frame the key ask, when to back off.

**What Went Wrong Last Time**
One paragraph: where did {actor_name}'s approach clash with {partner_name}'s psychology, and what specifically should change.

Under 400 words. Be concrete and specific to this scenario."""


def _build_hint_prompt_v9(
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
    """v9 — Goal completion checklist: exact end-state, must-happen items, hard red lines."""
    goal_score = actor_scores.get("goal", {}).get("raw", "?")
    goal_reasoning = ""
    if isinstance(raw_eval, dict):
        goal_reasoning = (raw_eval.get("goal", {}) or {}).get("reasoning", "")

    return f"""A player needs to fully complete a social goal in a roleplay. Last attempt scored {goal_score}/10. Build them an execution checklist.

Scenario: {scenario}
Character: {actor_name} ({actor_background})
Goal: {actor_goal}
Partner: {partner_name} (goal: {partner_goal})

Judge's assessment of why goal scored {goal_score}/10:
{goal_reasoning if goal_reasoning else "(not available)"}

Prior conversation:
{conversation_history}

---

**Goal Completion Checklist**

First, define in one sentence: what does "goal fully achieved" look like as a concrete end state? What must have happened by the end of the conversation?

Then list 4-5 specific things that MUST happen during the conversation for the goal to be complete. For each:
☐ [What must happen] — [When: early/mid/late] — [Exactly how: the specific line or action]

**Red Lines — these will block goal completion:**
List 2-3 specific things that, if done, will prevent goal achievement. Reference what happened in the prior run.

**The Missing Piece**
One sentence: the single thing that was not done last time that would have pushed the goal score to 9-10.

Under 350 words. Make checklist items concrete enough to execute verbatim."""


def _build_hint_prompt_v10(
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
    """v10 — Comprehensive plus: full rubric depth for all dims with judge reasoning + action bullets."""
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
        dim_lines.append(
            f"- **{dim}** [{importance}] score: {raw} (range {score_range}, normalized {norm:.2f})\n"
            f"  Judge said: {judge_reasoning if judge_reasoning else '(not available)'}"
        )
    dim_summary = "\n".join(dim_lines)

    return f"""You are an expert coach reviewing a social roleplay session. Write a comprehensive coaching brief that gives the player everything they need to score significantly higher on the next run.

Session: {scenario}
Character: {actor_name} ({actor_background})
Goal: {actor_goal}
Partner: {partner_name} (goal: {partner_goal})
Relationship: {relationship}

Scores and judge reasoning:
{dim_summary}

Conversation:
{conversation_history}

---

Write the brief in two parts:

**Part 1 — Rubric Analysis**
For each dimension below, write 2-3 sentences covering: (a) what a maximum score requires in this specific scenario, (b) what was missing or wrong in this run, (c) the single most impactful fix.
Cover ALL dimensions but weight your depth by importance: goal and relationship get the most attention, then knowledge, then the rest only if their score was notably suboptimal.

Goal (CRITICAL):
Relationship (HIGH):
Knowledge (HIGH):
[Other dims only if score was clearly low]

**Part 2 — Action Bullets (exactly 8)**
Concrete DO/DON'T bullets ordered by score impact. Must be specific to this conversation — no generic advice.
• DO/DON'T [specific action]

Total under 500 words."""


def _build_hint_prompt_v11(
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
    """v11 — Judge voice + bullets: raw judge reasoning shown verbatim, bullets directly address it."""
    goal_score = actor_scores.get("goal", {}).get("raw", "?")
    rel_score = actor_scores.get("relationship", {}).get("raw", "?")
    know_score = actor_scores.get("knowledge", {}).get("raw", "?")
    goal_reasoning = (raw_eval.get("goal", {}) or {}).get("reasoning", "") if isinstance(raw_eval, dict) else ""
    rel_reasoning = (raw_eval.get("relationship", {}) or {}).get("reasoning", "") if isinstance(raw_eval, dict) else ""
    know_reasoning = (raw_eval.get("knowledge", {}) or {}).get("reasoning", "") if isinstance(raw_eval, dict) else ""

    return f"""Session: {scenario}
{actor_name}'s goal: {actor_goal}
Scores: goal {goal_score}/10 | relationship {rel_score} | knowledge {know_score}/10

What the judge said about goal ({goal_score}/10):
{goal_reasoning if goal_reasoning else "(not available)"}

What the judge said about relationship ({rel_score}):
{rel_reasoning if rel_reasoning else "(not available)"}

What the judge said about knowledge ({know_score}/10):
{know_reasoning if know_reasoning else "(not available)"}

Conversation:
{conversation_history}

---

Give exactly 7 DO/DON'T bullets that directly and specifically address what the judge criticized. Each bullet must map to a concrete issue the judge raised. No other text.

• DO/DON'T [specific action tied to judge's concern]

Under 200 words total."""


def _build_hint_prompt_v12(
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
    """v12 — Goal maximizer: deep rubric for goal with example behaviors, then 4 tight bullets."""
    goal_score = actor_scores.get("goal", {}).get("raw", "?")
    rel_score = actor_scores.get("relationship", {}).get("raw", "?")
    goal_reasoning = (raw_eval.get("goal", {}) or {}).get("reasoning", "") if isinstance(raw_eval, dict) else ""
    rel_reasoning = (raw_eval.get("relationship", {}) or {}).get("reasoning", "") if isinstance(raw_eval, dict) else ""

    return f"""Session: {scenario}
{actor_name}'s goal: {actor_goal}
Partner: {partner_name} (goal: {partner_goal})
Goal score: {goal_score}/10 (CRITICAL — highest weight) | Relationship score: {rel_score}

Judge on goal: {goal_reasoning if goal_reasoning else "(not available)"}
Judge on relationship: {rel_reasoning if rel_reasoning else "(not available)"}

Conversation:
{conversation_history}

---

**Goal rubric — what earns 9-10/10:**
Explain in 3-4 sentences exactly what behaviors constitute complete goal achievement for "{actor_goal}" in this specific scenario. What must have been said or agreed upon? What does the end state look like? Give 1 concrete example line that would demonstrate goal completion.

**What fell short (goal scored {goal_score}/10):**
2 sentences: specifically what was missing or incomplete based on the conversation above.

**4 bullets — highest-impact actions for next run:**
• DO/DON'T [action — prioritize goal, then relationship]

Under 250 words."""


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
) -> str:
    """Generate structured improvement hints by reflecting on the rollout and scores.

    Returns a markdown-formatted hint string suitable for context distillation
    (prepended to the actor's system prompt as a 'teacher' context).
    """
    filtered_log = [e for e in conversation_log if e["action_type"] != "none"]
    conversation_history = "\n".join(e["natural_language"] for e in filtered_log)
    if not conversation_history.strip():
        conversation_history = "(No meaningful interaction occurred)"

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

    messages = [{"role": "user", "content": prompt}]
    try:
        async with asyncio.timeout(90):
            hint_text = await call_openai(messages, reasoning_effort="low")
    except asyncio.TimeoutError:
        logger.warning("Hint generation timed out after 60s")
        hint_text = None
    except Exception as e:
        logger.warning(f"Hint generation failed: {e}")
        hint_text = None

    if not hint_text:
        return ""

    # Wrap with a header so it's clearly framed when used as context
    header = (
        f"=== Session Reflection Hints for {actor_name} ===\n"
        f"(Review these hints before attempting this session to achieve a higher score)\n\n"
    )
    return header + hint_text.strip()
