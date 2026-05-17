"""
HiToM agent for Harmony evaluation.

Evaluates higher-order Theory-of-Mind (ToM) reasoning using the HiToM
dataset. Single-phase: prompt LLM with story + multiple-choice question,
extract letter answer (A-Z), validate against choices dict.

Based on: "HiToM: A Benchmark for Evaluating Higher-Order Theory of Mind
Reasoning in Large Language Models" (Wu et al., 2023).

HiToM extends standard ToM benchmarks with:
- Deception: agents lie; spoken statements do NOT update actual beliefs
- Exit-order trust: agents trust whichever agent exited the room later
- Private vs. public communication constraints
- Chain-of-thought prompting (CoTP) rows only

Scoring: the extracted letter must map to the correct_answer text via
the parsed choices dict. Reward is 1.0 if correct, 0.0 otherwise.
"""

import copy
import re
import logging
import uuid
from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are analysing a social interaction and need to answer a question about it. The following story happens in chronological order. You will be given a multiple-choice question and a note at the end. You should assume the following: (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. What an agent A tells others doesn't affect A's actual belief. An agent tends to trust an agent that exited the room later than himself. The exit order is known to all agents. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims. First give step-by-step analysis about the question. Then output the answer. For the answer, use <answer>(put your answer here)</answer> and include only the letter corresponding to your choice but not other information.

Below is the story and question:
## Story
{story}

## Extra Information
(to help you better understand and answer the question)
{extra_info}

## Question
{question}"""


def parse_choices(choices_str: str) -> dict[str, str]:
    """Parse 'A. option1, B. option2, ...' into {letter: option_text} dict."""
    return dict(re.findall(r"([A-Z])\. ([^,]+?)(?=,\s*[A-Z]\.|$)", choices_str))


def extract_answer(response: str) -> str:
    """Extract letter answer strictly from <answer>X</answer> tag. Returns '' if not found."""
    if not response:
        return ""
    match = re.search(r"<answer>\s*([A-Z])\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def evaluate_answer(letter: str, choices_str: str, correct_answer: str) -> bool:
    """Check whether letter maps to correct_answer text in the choices dict."""
    choices = parse_choices(choices_str)
    if letter not in choices:
        return False
    return choices[letter].strip() == correct_answer.strip()


def compute_hitom_aggregates(results: list[dict]) -> dict:
    """Compute accuracy overall, by deception flag, story_length, and question_order."""
    all_rewards: list[float] = []
    by_deception: dict[str, list[float]] = {}
    by_length: dict[str, list[float]] = {}
    by_order: dict[str, list[float]] = {}

    for r in results:
        if not isinstance(r, dict):
            continue
        reward = r.get("reward", 0.0)
        all_rewards.append(reward)
        key_d = str(r.get("deception", "unknown"))
        by_deception.setdefault(key_d, []).append(reward)
        key_l = str(r.get("story_length", "unknown"))
        by_length.setdefault(key_l, []).append(reward)
        key_o = str(r.get("question_order", "unknown"))
        by_order.setdefault(key_o, []).append(reward)

    aggregates = {}
    if all_rewards:
        aggregates["accuracy_overall"] = sum(all_rewards) / len(all_rewards)
    for k, v in sorted(by_deception.items()):
        aggregates[f"accuracy_deception_{k}"] = sum(v) / len(v)
    for k, v in sorted(by_length.items(), key=lambda x: (len(x[0]), x[0])):
        aggregates[f"accuracy_length_{k}"] = sum(v) / len(v)
    for k, v in sorted(by_order.items(), key=lambda x: (len(x[0]), x[0])):
        aggregates[f"accuracy_order_{k}"] = sum(v) / len(v)
    return aggregates


async def agent_loop(data, context):
    """
    HiToM: Evaluate higher-order ToM reasoning on a single question.

    Args:
        data: {
            "row": dict with keys:
                story         - story text (chronological social interaction)
                question      - the question text
                choices       - 'A. opt1, B. opt2, ...' multiple-choice string
                correct_answer - correct answer option text
                deception     - bool string ('True'/'False')
                story_length  - int (number of agents/movements)
                question_order - int
                index         - row identifier
                set_id        - story group id
        }
        context: {
            "client": AsyncOpenAI instance,
            "model": str model name
        }

    Returns:
        {
            "reward": float (1.0 if correct, 0.0 otherwise),
            "chat": list of message dicts,
            "predicted": str extracted letter,
            "correct": str ground truth option text,
            "is_correct": bool,
            "deception": str,
            "story_length": str,
            "question_order": str,
            "index": str,
        }
    """
    row = data["extra_info"]

    story = row["story"]
    # Combine question text with choices
    question_with_choices = row["question"] + "\n" + row["choices"]
    extra_info = row.get("extra_info", "")

    prompt = PROMPT_TEMPLATE.format(
        story=story,
        extra_info=extra_info,
        question=question_with_choices,
    )
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    content = remove_think(response)
    predicted = extract_answer(content)
    correct_answer = str(row.get("correct_answer", ""))
    is_correct = evaluate_answer(predicted, row.get("choices", ""), correct_answer)
    reward = 1.0 if is_correct else 0.0

    output = await agent.get_agent_output(reward, extra_info={
        "hitom/reward": reward,
        "hitom/response_length": len(response.split()) if response else 0,
        "all/score": reward,
        "all/score_v1": reward,
    })

    # ===========================================================================
    # Hint + second attempt (mirrors sotopia/lifechoices/fantom copy-agent pattern)
    # ===========================================================================
    extra = {}
    hint = None
    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and not is_correct and context.global_step < 100):
        from agents.hitom.hint import generate_hint
        hint = await generate_hint(row, content)
        if hint:
            extra['hint'] = hint

    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and context.is_train
            and hint):
        from agents.hitom.hint_agent import agent_loop as hint_agent_loop
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

    return {
        "reward": reward,
        "chat": chat,
        "predicted": predicted,
        "correct": correct_answer,
        "is_correct": is_correct,
        "deception": str(row.get("deception", "")),
        "story_length": str(row.get("story_length", "")),
        "question_order": str(row.get("question_order", "")),
        "index": str(row.get("index", "")),
    }
