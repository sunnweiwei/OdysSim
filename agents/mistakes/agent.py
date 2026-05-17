"""
Mistakes agent for Harmony evaluation.

Single-turn benchmark:
- Input row is one expanded case (problem + misconception + target wrong option)
- Agent generates one response
- Score by whether extracted final option letter matches target option
"""

from __future__ import annotations

import copy
import re
import uuid
from typing import Any, Dict, Optional
from agents.utils import Agent, process_post_chat, remove_think

OPTION_LETTERS = ("A", "B", "C", "D")
STUDENT_SYSTEM_PROMPT = (
    "You are an AI assistant designed to simulate student misconceptions in solving K-12 math problems.\n"
    "You will be shown a math problem and a student misconception.\n"
    "Simulate the answer that a student with that misconception would give.\n"
    "Do NOT generate the correct answer. Solve as if you had the misconception.\n"
    "Think concisely from the mistaken student perspective, then output your final answer in the format:\n"
    "<answer>X</answer>\n"
    "where X is exactly one letter: A, B, C, or D.\n"
    "If no option seems perfectly correct, output the closest option letter."
)



def _extract_option_letter(text: str) -> Optional[str]:
    """Extract option letter strictly from <answer>X</answer> tag."""
    if not text:
        return None
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _build_prompt_from_row(row: Dict[str, Any]) -> str:
    """Build zero-shot user prompt: problem + options + misconception."""
    options = []
    for letter in OPTION_LETTERS:
        text = str(row.get(f"Answer{letter}Text") or "").strip()
        if text:
            options.append(f"{letter}) {text}")
    options_text = "\n".join(options)
    misconception = str(row.get("MisconceptionName") or row.get("misconception_text") or "").strip()
    question = str(row.get("QuestionText") or row.get("question") or "").strip()
    return (
        "Math Problem:\n"
        f"Question: {question}\n"
        f"Answer Choices:\n{options_text}\n\n"
        f"Student Misconception:\n{misconception}"
    )


def compute_mistakes_aggregates(results: list[dict]) -> dict:
    """Compute accuracy by target option and overall."""
    by_option: dict[str, list[float]] = {}
    all_rewards: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        reward = float(result.get("reward", 0.0))
        target = str(result.get("correct", "")).strip().upper() or "unknown"
        by_option.setdefault(target, []).append(reward)
        all_rewards.append(reward)

    aggregates: dict[str, float] = {}
    for option, rewards in sorted(by_option.items()):
        aggregates[f"accuracy_target_{option}"] = sum(rewards) / len(rewards)
    if all_rewards:
        aggregates["accuracy_overall"] = sum(all_rewards) / len(all_rewards)
    return aggregates


async def agent_loop(data, context):
    """
    Mistakes: single-turn evaluation for one expanded row.

    Input:
      data["row"]: expanded JSONL row from mistakes prepare_dataset.py
    Output:
      {reward, chat, predicted, correct, ...}
    """
    row = data["extra_info"]

    # Always build the zero-shot prompt to avoid inheriting old dataset prompt formats.
    prompt = _build_prompt_from_row(row)
    if not prompt.strip():
        prompt = str(row.get("raw_prompt") or "").strip()

    chat = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    content = remove_think(response)
    predicted = _extract_option_letter(content)
    correct = str(row.get("TargetOption") or row.get("target_option") or "").strip().upper()
    reward = 1.0 if (predicted is not None and predicted == correct) else 0.0

    output = await agent.get_agent_output(reward, extra_info={
        "mistakes/reward": reward,
        "mistakes/response_length": len(response.split()) if response else 0,
        "all/score": reward,
        "all/score_v1": reward,
    })

    # ===========================================================================
    # Hint + second attempt
    # ===========================================================================
    extra = {}
    hint = None
    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and reward < 1.0 and context.global_step < 90):
        from agents.mistakes.hint import generate_hint
        hint = await generate_hint(row, content)
        if hint:
            extra['hint'] = hint

    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and context.is_train
            and hint):
        from agents.mistakes.hint_agent import agent_loop as hint_agent_loop
        data['extra_info']['hint'] = hint
        data['extra_info']['old_reward'] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = 'hint_agent'
        output = [output, copy_agent_output, hint_agent_output]
        # output = [output, copy_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output

    return {
        "reward": reward,
        "chat": chat,
        "predicted": predicted,
        "correct": correct,
        "final_answer": final_answer,
        "final_answer_mode": final_answer_mode,
        "target_text": str(row.get("TargetAnswer") or row.get("target_text") or "").strip(),
        "misconception_id": str(row.get("MisconceptionId") or "").strip(),
        "misconception_name": str(row.get("MisconceptionName") or "").strip(),
        "case_id": str(row.get("id") or ""),
    }
