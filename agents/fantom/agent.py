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
FanToM agent for Harmony evaluation.

Evaluates Theory-of-Mind (ToM) reasoning in multi-party conversations using
the FanToM dataset. Single-phase: prompt LLM with conversation context +
question, extract answer, evaluate per question type.

Based on: "FanToM: A Benchmark for Stress-Testing Machine Theory of Mind in
Interactions" (Kim et al., 2023).

FanToM contains multiple question types per conversational scenario:
  - tom:belief:*:multiple-choice   Which answer option does character believe?
  - tom:answerability:list         List characters who know the answer
  - tom:answerability:binary       Does character Y know the answer? (yes/no)
  - tom:info_accessibility:list    List characters who can access the info
  - tom:info_accessibility:binary  Can character Y access the info? (yes/no)
  - fact:*                         Factual question about the conversation

Reward per question:
  - multiple-choice / list / binary: 1.0 if correct, 0.0 otherwise
  - fact: token-level F1 score (0.0–1.0)
"""

import copy
import logging
import re
import string
import uuid
from collections import Counter

from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are analyzing a social conversation and need to answer a question about it. Assume that the characters do not know any other information than what is provided in the conversation. For the answer, use <answer>(put your answer here)</answer>.

## Context
{context}

## Extra Information
(to help you better understand and answer the question)
{extra_info}

## Question
{question}"""


# ---------------------------------------------------------------------------
# Data helpers (ported from social_world_model/task_modules/fantom.py)
# ---------------------------------------------------------------------------


def str_to_list(s: str) -> list[str]:
    """Convert a bracket/comma list string like "['A', 'B']" to a list."""
    parts = s.split(",")
    return [c.strip(" []'\"") for c in parts if c.strip(" []'\"")]


def flatten_fantom_data(entry: dict) -> list[dict]:
    """
    Flatten a single FanToM JSONL entry into individual question rows.

    Each entry contains nested question dicts under keys like:
      beliefQAs (list), infoAccessibilityQAs_binary (list),
      answerabilityQAs_binary (list), infoAccessibilityQA_list (single),
      answerabilityQA_list (single), factQA (single, excluded).

    Returns a list of flat row dicts, one per question.
    """
    data_list: list[dict] = []
    fact_qa_question = entry["factQA"]["question"]
    fact_qa_answer = entry["factQA"]["correct_answer"]

    for key in entry.keys():
        if "QAs" in key:
            # Plural key → list of question dicts
            for question in entry[key]:
                row = {
                    "question": question["question"],
                    "question_type": question["question_type"],
                    "tom_type": question.get("tom_type", ""),
                    "correct_answer": question["correct_answer"],
                    "wrong_answer": question.get("wrong_answer", ""),
                    "missed_info_accessibility": question.get("missed_info_accessibility", ""),
                    "context": entry["short_context"],
                    "full_context": entry["full_context"],
                    "set_id": entry["set_id"],
                    "part_id": entry["part_id"],
                    "complete_question": question["complete_question"],
                    "fact_question": fact_qa_question,
                    "fact_answer": fact_qa_answer,
                }
                data_list.append(row)
        elif "QA" in key and "fact" not in key:
            # Singular key (not factQA) → single question dict
            question = entry[key]
            row = {
                "question": question["question"],
                "question_type": question["question_type"],
                "tom_type": question.get("tom_type", ""),
                "correct_answer": question["correct_answer"],
                "wrong_answer": question.get("wrong_answer", ""),
                "missed_info_accessibility": question.get("missed_info_accessibility", ""),
                "context": entry["short_context"],
                "full_context": entry["full_context"],
                "set_id": entry["set_id"],
                "part_id": entry["part_id"],
                "complete_question": question["complete_question"],
                "fact_question": fact_qa_question,
                "fact_answer": fact_qa_answer,
            }
            data_list.append(row)
        # factQA is intentionally excluded (no "QAs" and has "fact")

    return data_list


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_answer(response: str) -> str:
    """Extract content strictly from <answer>...</answer> tags. Returns '' if not found."""
    if not response:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Scoring helpers (ported from FantomEvalAgent)
# ---------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles/punct, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def compute_f1(ground_truth: str, model_response: str) -> float:
    """Compute token-level F1 score between ground truth and model response."""
    gt_tokens = _normalize_text(ground_truth).split()
    pred_tokens = _normalize_text(model_response).split()
    if not gt_tokens or not pred_tokens:
        return 0.0
    common = Counter(gt_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def evaluate_mc_belief(correct_answer, model_response: str) -> bool:
    """
    Evaluate multiple-choice belief question.
    correct_answer is an int (0-3) mapping to letter (a-d).
    Expects the extracted <answer> content to be just the letter, optionally
    wrapped in parens or with a trailing period (e.g., "a", "(a)", "a)", "a.").
    """
    int_to_alpha = {0: "a", 1: "b", 2: "c", 3: "d"}
    try:
        letter = int_to_alpha[int(correct_answer)]
    except (ValueError, KeyError):
        return False
    resp = model_response.strip().lower()
    m = re.match(r"^\(?([a-d])\)?\.?$", resp)
    if not m:
        return False
    return m.group(1) == letter


def _has_whole_name(name: str, text: str) -> bool:
    """Whole-word/substring safe containment check for character names."""
    if not name:
        return False
    return re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", text.lower()) is not None


def evaluate_list_q(correct_answer, wrong_answer, model_response: str) -> bool:
    """
    Evaluate list question: all correct characters included, no wrong ones.
    Uses whole-word matching to avoid substring leakage (e.g., 'Al' vs 'Alice').
    """
    if isinstance(correct_answer, str):
        correct_list = str_to_list(correct_answer)
        wrong_list = str_to_list(str(wrong_answer)) if wrong_answer else []
    else:
        correct_list = [str(c) for c in correct_answer]
        wrong_list = [str(c) for c in wrong_answer] if wrong_answer else []

    if not correct_list:
        return False

    all_correct_in = all(_has_whole_name(c, model_response) for c in correct_list)
    any_wrong_in = any(_has_whole_name(c, model_response) for c in wrong_list)
    return all_correct_in and not any_wrong_in


def _has_yes_pattern(text: str) -> bool:
    t = text.lower().strip().strip("'").strip('"')
    return t.startswith("yes") or t.startswith("true") or " yes," in t or " yes " in t or " yes." in t or " knows " in t


def _has_no_pattern(text: str) -> bool:
    t = text.lower().strip().strip("'").strip('"')
    return (
        t.startswith("no")
        or t.startswith("false")
        or " no," in t
        or " no " in t
        or " no." in t
        or " does not know " in t
        or " doesn't know " in t
    )


def yesno_to_int(yesno_str: str) -> int:
    """Map correct_answer string to int."""
    mapping = {"yes": 1, "no": 0, "no:long": 0, "error": -1}
    return mapping.get(yesno_str, -1)


def evaluate_binary(correct_answer: str, model_response: str) -> bool:
    """
    Evaluate binary (yes/no) question by requiring the correct polarity present
    AND the wrong polarity absent. Exclusion-based rather than EM to avoid
    format constraints but still block hedged/both-polarity hacks.
    """
    gold = yesno_to_int(correct_answer)
    if gold == -1:
        return False
    yes_present = _has_yes_pattern(model_response)
    no_present = _has_no_pattern(model_response)
    if gold == 1:
        return yes_present and not no_present
    return no_present and not yes_present


def score_response(question_type: str, correct_answer, wrong_answer, model_response: str) -> float:
    """Score a single response based on question_type."""
    qt = str(question_type)
    if qt.startswith("tom:belief:") and qt.endswith(":multiple-choice"):
        return 1.0 if evaluate_mc_belief(correct_answer, model_response) else 0.0
    elif qt.endswith(":list"):
        return 1.0 if evaluate_list_q(correct_answer, wrong_answer, model_response) else 0.0
    elif qt.endswith(":binary"):
        return 1.0 if evaluate_binary(str(correct_answer), model_response) else 0.0
    elif qt.startswith("fact"):
        return compute_f1(str(correct_answer).lower(), model_response.lower())
    else:
        raise NotImplementedError(f"Unknown question_type: {qt}")


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------


def compute_fantom_aggregates(results: list[dict]) -> dict:
    """
    Compute accuracy breakdown by question_type and missed_info_accessibility.
    Fact questions report mean F1. Other types report accuracy (0/1).
    """
    by_type: dict[str, list[float]] = {}
    by_accessibility: dict[str, list[float]] = {}
    all_non_fact: list[float] = []

    for r in results:
        if not isinstance(r, dict):
            continue
        reward = r.get("reward", 0.0)
        qt = str(r.get("question_type", "unknown"))
        acc = str(r.get("missed_info_accessibility", "unknown"))
        by_type.setdefault(qt, []).append(reward)
        by_accessibility.setdefault(acc, []).append(reward)
        if not qt.startswith("fact"):
            all_non_fact.append(reward)

    aggregates = {}
    for qt, rewards in sorted(by_type.items()):
        key = f"f1_{qt}" if qt.startswith("fact") else f"accuracy_{qt}"
        aggregates[key] = sum(rewards) / len(rewards)
    for acc, rewards in sorted(by_accessibility.items()):
        aggregates[f"accuracy_accessibility_{acc}"] = sum(rewards) / len(rewards)
    if all_non_fact:
        aggregates["accuracy_tom_overall"] = sum(all_non_fact) / len(all_non_fact)
    return aggregates


# ---------------------------------------------------------------------------
# agent_loop
# ---------------------------------------------------------------------------


async def agent_loop(data, context):
    """
    FanToM: Evaluate ToM reasoning on a single question from a multi-party
    conversation.

    Args:
        data: {
            "row": dict with keys:
                context               - short conversation context
                complete_question     - full question with any embedded instructions
                question_type         - e.g. 'tom:belief:inaccessible:multiple-choice'
                correct_answer        - ground truth (int index, list string, or yes/no)
                wrong_answer          - wrong characters/answers for list questions
                missed_info_accessibility - 'inaccessible' or 'accessible'
                set_id, part_id, tom_type - metadata
        }
        context: {
            "client": AsyncOpenAI instance,
            "model": str model name
        }

    Returns:
        {
            "reward": float (1.0/0.0 for classification; F1 for fact questions),
            "chat": list of message dicts,
            "predicted": str raw extracted answer,
            "correct": str ground truth,
            "is_correct": bool (reward >= 1.0 for display),
            "question_type": str,
            "set_id": int/str,
            "part_id": int/str,
            "missed_info_accessibility": str,
        }
    """
    row = data["extra_info"]
    context_text = row.get("context", "")
    complete_question = row.get("complete_question", row.get("question", ""))
    extra_info = row.get("extra_info", "")

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        extra_info=extra_info,
        question=complete_question,
    )
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    content = remove_think(response)
    predicted = extract_answer(content)
    question_type = str(row.get("question_type", ""))
    correct_answer = row.get("correct_answer", "")
    wrong_answer = row.get("wrong_answer", "")

    reward = score_response(question_type, correct_answer, wrong_answer, predicted)
    is_correct = reward >= 1.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "fantom/reward": reward,
            "fantom/response_length": len(response.split()) if response else 0,
            "all/score": reward,
            "all/score_v1": reward,
        },
    )

    # ===========================================================================
    # Hint + second attempt (mirrors sotopia/lifechoices copy-agent pattern)
    # Generate a hint when the model answered incorrectly or partially.
    # ===========================================================================
    extra = {}
    hint = None
    if getattr(context.config.algorithm, "agent_version", None) == "copy" and reward < 1.0:
        from agents.fantom.hint import generate_hint

        hint = await generate_hint(row, content, reward)
        if hint:
            extra["hint"] = hint

    if getattr(context.config.algorithm, "agent_version", None) == "copy" and context.is_train and hint:
        from agents.fantom.hint_agent import agent_loop as hint_agent_loop

        data["extra_info"]["hint"] = hint
        data["extra_info"]["old_reward"] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = "hint_agent"
        output = [output, copy_agent_output, hint_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output

    return {
        "reward": reward,
        "chat": chat,
        "predicted": predicted,
        "correct": str(correct_answer),
        "is_correct": is_correct,
        "question_type": question_type,
        "set_id": row.get("set_id", ""),
        "part_id": row.get("part_id", ""),
        "tom_type": row.get("tom_type", ""),
        "missed_info_accessibility": row.get("missed_info_accessibility", ""),
    }
