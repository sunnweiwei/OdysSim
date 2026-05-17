"""
Sotopia hint agent — measures improvement from reflection hints.

For each task, this agent:
  1. Reads a prior rollout (from data/sotopia_rollout.jsonl via row['rollout'])
  2. Generates session-specific improvement hints from that rollout
  3. Does a fresh teacher rollout using the hint-augmented prompt
  4. Evaluates the teacher rollout with the same LLM judge
  5. Returns reward = new_score - old_score (improvement delta)

This lets us quantify how much the hint actually helps before using it for
context distillation (KL training).
"""

import re
import logging
import asyncio
import time

from agents.utils import Agent, process_post_chat, remove_think
from agents.sotopia.hint import generate_hint, get_teacher_character_prompt
from agents.sotopia.agent import (
    SimpleAgent,
    get_character_prompt,
    parse_action,
    action_to_natural_language,
    evaluate_episode_single,
    hack_judge,
    DIMENSIONS,
    DIMENSION_RANGES,
    normalize_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout parsing
# ---------------------------------------------------------------------------

def parse_rollout_conversation(rollout: dict, actor_name: str, partner_name: str) -> list:
    """Reconstruct a conversation_log list from a saved rollout's output field.

    The rollout output is formatted as alternating ``assistant\\n`` / ``user\\n``
    blocks.  Actor (assistant) blocks contain think-tags + JSON actions;
    partner (user) blocks contain natural-language lines already formatted by
    ``action_to_natural_language``.
    """
    output_text = rollout.get("output", "")
    if not output_text:
        return []

    # Split on role markers — yields ['', role, content, role, content, ...]
    segments = re.split(r'(assistant|user)\n', output_text)

    conversation_log = []
    turn = 0
    for i in range(1, len(segments) - 1, 2):
        role = segments[i]
        content = segments[i + 1] if i + 1 < len(segments) else ""

        if role == "assistant":
            clean = remove_think(content.strip(), remove_unclosed=True)
            action = parse_action(clean)
            nl = action_to_natural_language(actor_name, action)
            conversation_log.append({
                "turn": turn + 1,
                "agent": actor_name,
                "action_type": action["action_type"],
                "argument": action["argument"],
                "natural_language": nl,
            })
        else:
            # Partner turn — already natural language
            content = content.strip()
            if content:
                conversation_log.append({
                    "turn": turn + 1,
                    "agent": partner_name,
                    "action_type": "speak",
                    "argument": content,
                    "natural_language": content,
                })
        turn += 1

    return conversation_log


def extract_old_scores(rollout: dict) -> dict:
    """Build actor_scores dict (matching evaluate_episode_single output) from rollout fields."""
    actor_scores = {}
    for dim in DIMENSIONS:
        raw = rollout.get(f"sotopia/{dim}", 0)
        lo, hi = DIMENSION_RANGES[dim]
        raw = max(lo, min(hi, int(raw) if isinstance(raw, (int, float)) else (lo + hi) // 2))
        actor_scores[dim] = {"raw": raw, "normalized": normalize_score(raw, dim)}
    return actor_scores


# ---------------------------------------------------------------------------
# agent_loop: Harmony interface
# ---------------------------------------------------------------------------

async def agent_loop(data, context):
    """
    Hint agent: generate hints from a prior rollout, re-run the session with
    those hints, and return the score improvement delta as the reward.

    Expected extra_info fields (same as sotopia agent, plus 'rollout'):
      - scenario, agent1_name, agent2_name, agent1_background, agent2_background,
        agent1_goal, agent2_goal, relationship, eval_position
      - rollout: dict — a single line from data/sotopia_rollout.jsonl
    """
    row = data["extra_info"]
    agent1_name = row["agent1_name"]
    agent2_name = row["agent2_name"]
    agent1_background = row.get("agent1_background", "")
    agent2_background = row.get("agent2_background", "")
    agent1_goal = row.get("agent1_goal", "")
    agent2_goal = row.get("agent2_goal", "")
    relationship = row.get("relationship", "")
    # Replace Agent1/Agent2 placeholders in scenario with actual names
    scenario = (row["scenario"]
                .replace("Agent1", agent1_name)
                .replace("Agent2", agent2_name)
                .replace("agent1", agent1_name)
                .replace("agent2", agent2_name))
    eval_position = row.get("eval_position", "agent1")

    if eval_position == "agent1":
        actor_name, actor_background, actor_goal = agent1_name, agent1_background, agent1_goal
        partner_name, partner_background, partner_goal = agent2_name, agent2_background, agent2_goal
        actor_goes_first = True
    else:
        actor_name, actor_background, actor_goal = agent2_name, agent2_background, agent2_goal
        partner_name, partner_background, partner_goal = agent1_name, agent1_background, agent1_goal
        actor_goes_first = False

    hint = row.get("hint")
    rollout = row.get("rollout")  # dict from jsonl
    old_actor_scores = extract_old_scores(rollout)
    old_reward = float(rollout.get("sotopia/reward", rollout.get("reward", 0.0)))
    if not hint:
        prior_conversation_log = parse_rollout_conversation(rollout, actor_name, partner_name)
        hint = await generate_hint(
            conversation_log=prior_conversation_log,
            scenario=scenario,
            actor_name=actor_name,
            actor_background=actor_background,
            actor_goal=actor_goal,
            partner_name=partner_name,
            partner_background=partner_background,
            partner_goal=partner_goal,
            relationship=relationship,
            actor_scores=old_actor_scores,
            raw_eval={},
            context=context,
        )

    # ------------------------------------------------------------------
    # Phase 3: Teacher rollout — same task but actor sees the hint
    # ------------------------------------------------------------------
    teacher_system = get_teacher_character_prompt(
        agent_name=actor_name,
        background=actor_background,
        scenario=scenario,
        goal=actor_goal,
        other_name=partner_name,
        relationship=relationship,
        hint=hint,
    )
    teacher_chat = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": "=== Conversation Start ==="},
    ]

    actor_agent = Agent(context.llm_client, teacher_chat, context.tokenizer, context.config, prompt_turn=2)
    partner_chat = [
        {"role": "system",
         "content": get_character_prompt(partner_name, partner_background, scenario, partner_goal, actor_name,
                                         relationship)},
        {"role": "user", "content": "=== Conversation Start ==="},
    ]
    partner_agent = SimpleAgent(partner_chat)

    conversation_log = []
    max_turns = 15
    consecutive_none = 0

    for turn in range(max_turns):
        if (turn % 2 == 0) == actor_goes_first:
            current_agent, current_name, other_agent = actor_agent, actor_name, partner_agent
        else:
            current_agent, current_name, other_agent = partner_agent, partner_name, actor_agent

        response = await current_agent.step()
        response = remove_think(response, remove_unclosed=True)

        if not response:
            break

        action = parse_action(response)
        action_nl = action_to_natural_language(current_name, action)
        conversation_log.append({
            "turn": turn + 1,
            "agent": current_name,
            "action_type": action["action_type"],
            "argument": action["argument"],
            "natural_language": action_nl,
        })

        other_agent.append({"role": "user", "content": action_nl})

        if action["action_type"] == "leave" and turn >= 5:
            break
        if action["action_type"] == "none":
            consecutive_none += 1
        else:
            consecutive_none = 0
        if consecutive_none > 2:
            break

    # ------------------------------------------------------------------
    # Phase 4: Evaluate teacher rollout
    # ------------------------------------------------------------------
    _eval_start = time.monotonic()
    eval_result, hack_result = await asyncio.gather(
        evaluate_episode_single(
            conversation_log,
            scenario,
            actor_name=actor_name,
            actor_background=actor_background,
            actor_goal=actor_goal,
            partner_name=partner_name,
            partner_background=partner_background,
            partner_goal=partner_goal,
            relationship=relationship,
            structured_output=True,
        ),
        hack_judge(
            conversation_log,
            actor_name=actor_name,
            actor_goal=actor_goal,
            scenario=scenario,
        ),
    )
    eval_time = time.monotonic() - _eval_start

    new_reward = eval_result["reward"]
    if hack_result.risk_level != "low":
        new_reward = 0
        logger.warning(f"[hack_judge] HIGH risk detected — reward zeroed. Reason: {hack_result.reason}")
    new_avg = eval_result["actor_avg"]
    new_scores = eval_result["actor_scores"]
    reward_delta = new_reward - old_reward
    old_avg = float(rollout.get("sotopia/eval_avg", 0.0))

    # ------------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------------
    extra_info = {
        "sotopia-hint/reward_delta": reward_delta,
        "sotopia-hint/delta_positive": int(reward_delta > 0),
        "sotopia-hint/eval_time": eval_time,
        "sotopia-hint/eval_avg_delta": new_avg - old_avg,
        "sotopia-hint/hack_low": int(hack_result.risk_level == "low"),
        "sotopia-hint/hack_medium": int(hack_result.risk_level == "medium"),
        "sotopia-hint/hack_high": int(hack_result.risk_level == "high"),
    }
    for dim in DIMENSIONS:
        old_raw = old_actor_scores[dim]["raw"]
        new_raw = new_scores[dim]["raw"]
        extra_info[f"sotopia-hint/{dim}_delta"] = new_raw - old_raw

    reward = new_reward
    # reward = reward_delta
    # reward = delta so the training signal reflects improvement from the hint
    if actor_agent.think_format_correct() == 0 and context.is_train:
        use_reward = reward / 2
    else:
        use_reward = reward
    output = await actor_agent.get_agent_output(use_reward, extra_info=extra_info)

    return output
