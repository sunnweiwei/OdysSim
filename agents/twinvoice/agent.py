"""
TwinVoice agent for Harmony evaluation.

Discriminative MCQ benchmark for evaluating LLM-based persona simulation
("digital twins"). Each instance presents a persona's conversation history,
an anchor post (stimulus), and 4 response choices (A/B/C/D). The model must
pick the response that best matches the persona.

Based on: "TwinVoice: Digital Twin Evaluation for Personalized LLMs"
(https://arxiv.org/abs/2510.25536)

Dataset: bangdedadi/TwinVoice on HuggingFace
- 5,687 instances across 3 dimensions: Social, Interpersonal, Narrative
"""

import copy
import json
import re
import uuid
from agents.utils import Agent, process_post_chat, remove_think


PROMPT_TEMPLATE = """You are given a persona's conversation history and an anchor post. Based on the persona's past behavior and communication style, select the response (A, B, C, or D) that the persona would most likely write in reply to the anchor post.

# Persona Conversation History:
{history}

# Anchor Post (Stimulus):
{anchor_post}

# Response Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Based on the persona's conversation history and style, which response (A, B, C, or D) would this persona most likely write? Provide your reasoning within the <reasoning></reasoning> tag. For the answer, use <answer>(put your answer here)</answer> and include only the letter (A, B, C, or D) of your chosen option."""


def build_prompt(row: dict) -> str:
    """Build the prompt for a single TwinVoice instance."""
    history = row.get("conversation_history", row.get("history", []))
    if isinstance(history, str):
        history = json.loads(history)
    history_text = "\n".join(f"- {h}" for h in history)

    choices = row.get("answer_choices", row.get("choices", []))
    if isinstance(choices, str):
        choices = json.loads(choices)
    return PROMPT_TEMPLATE.format(
        history=history_text,
        anchor_post=row.get("anchor_post", ""),
        choice_a=choices[0] if len(choices) > 0 else "",
        choice_b=choices[1] if len(choices) > 1 else "",
        choice_c=choices[2] if len(choices) > 2 else "",
        choice_d=choices[3] if len(choices) > 3 else "",
    )


def extract_choice(response: str) -> str | None:
    """Extract letter answer strictly from <answer>X</answer> tag. Returns None if not found."""
    if not response:
        return None
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def compute_twinvoice_aggregates(results: list[dict]) -> dict:
    """Compute accuracy breakdown by dimension (social/interpersonal/narrative)."""
    by_dim: dict[str, list[float]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        extra = r.get("extra_fields", {}).get("reward_extra_info", {})
        dim = str(extra.get("twinvoice/dimension", r.get("twinvoice/dimension", r.get("dimension", "unknown"))))
        by_dim.setdefault(dim, []).append(r.get("reward", 0.0))

    aggregates = {}
    all_rewards = []
    for dim, rewards in sorted(by_dim.items()):
        aggregates[f"accuracy_{dim}"] = sum(rewards) / len(rewards)
        all_rewards.extend(rewards)
    if all_rewards:
        aggregates["accuracy_overall"] = sum(all_rewards) / len(all_rewards)
    return aggregates


async def agent_loop(data: dict, context):
    """
    TwinVoice: Evaluate persona simulation on a single MCQ instance.

    Args:
        data: {
            "extra_info": dict with keys: id, user_id, anchor_post, choices,
                          answer_idx, history, dimension, pccd_metrics
        }
        context: EvalContext with llm_client, tokenizer, config

    Returns:
        Agent output dict with reward and metadata.
    """
    row = data["extra_info"]

    prompt = build_prompt(row)
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    content = remove_think(response)
    predicted = extract_choice(content)

    correct_idx = row["answer_idx"]  # 0-3
    correct_letter = "ABCD"[correct_idx]
    is_correct = predicted == correct_letter if predicted else False
    reward = 1.0 if is_correct else 0.0

    output = await agent.get_agent_output(reward, extra_info={
        "twinvoice/reward": reward,
        "all/score": reward,
        "all/score_v1": reward,
    })

    # ===========================================================================
    # Hint + second attempt
    # ===========================================================================
    extra = {}
    hint = None
    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and reward < 1.0):
        from agents.twinvoice.hint import generate_hint
        hint = await generate_hint(row, content)
        if hint:
            extra['hint'] = hint

    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and context.is_train
            and hint):
        from agents.twinvoice.hint_agent import agent_loop as hint_agent_loop
        data['extra_info']['hint'] = hint
        data['extra_info']['old_reward'] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = 'hint_agent'
        output = [output, copy_agent_output, hint_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output
