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
userLLM agent for Harmony evaluation.
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
import unicodedata
import uuid
from itertools import combinations
from typing import Any, Optional

from pydantic import BaseModel as _BaseModel

from agents.userllm.ai_detector import analyze_text
from agents.userllm.helpers import (
    INTENT_ADHERENCE_JUDGE_PROMPT,
    TestCase,
    _extract_ai_detector_score,
    _extract_choice_texts,
    _extract_intent,
    _extract_intent_adherence_fields,
    _filter_tokens_for_overlap,
    _format_first_turn_prompt,
    _format_sequential_turn_prompt,
    _intent_1gram_overlap,
    _normalize_for_choice_match,
    _to_optional_bool,
)
from agents.utils import (
    Agent,
    call_openai,
    call_openai_parse,
    editlens_score,
    get_judge_model,
    get_judge_reasoning,
    process_post_chat,
    remove_think,
)


class QualityDetection(_BaseModel):
    risk_level: str  # "low", "medium", "high"
    reason: str


async def quality_judge(output: str, intent: str, source: str) -> QualityDetection:
    """Check whether the userLLM response is a reasonable human-like user turn.

    Risk levels:
      low    — natural human user message throughout
      medium — somewhat AI-like phrasing or mildly off-task but still plausible
      high   — clearly breaks character (explicitly identifies as AI, refuses to
               act as a user, completely incoherent/degenerate text, or pure
               repetition loops with no meaning)
    """
    return QualityDetection(risk_level="low", reason="skip")
    if "<|endconversation|>" in output and not output.replace("<|endconversation|>", "").strip():
        return QualityDetection(risk_level="low", reason="conversation ended")
    text = output.replace("<|endconversation|>", "").strip()
    if not text:
        return QualityDetection(risk_level="high", reason="empty response")

    prompt = f"""You are an auditor checking whether a simulated human user response is genuine and reasonable.

The model is playing a human user with the following intent/goal:
{intent or "(not specified)"}

Source task type: {source or "unknown"}

Below is the model's response:
{text}

Does this response look like something a real human user would actually write in this context?

Be lenient. Most responses — even awkward, short, terse, or imperfect ones — should be rated **low**.

- **low** (default): Any response that could plausibly come from a real person: casual phrasing, typos, brief messages, off-topic small talk, frustration, or even "<|endconversation|>". When in doubt, choose low.
- **medium**: The response is noticeably AI-like (overly structured, excessively polite, very long and helpful) but is still somewhat relevant to the conversation context. Must be obvious, not just slightly formal.
- **high**: Only for very clear violations: (1) the response explicitly identifies itself as an AI or language model; (2) the text is completely incoherent or consists of degenerate loops/repetition with no meaningful content.

Output JSON: {{"risk_level": "low"|"medium"|"high", "reason": "<one sentence>"}}"""

    result = await call_openai_parse(
        [{"role": "user", "content": prompt}],
        QualityDetection,
        model=get_judge_model("gpt-5.4-nano"),
        reasoning={"effort": get_judge_reasoning("low")},
    )
    if result is None:
        return QualityDetection(risk_level="low", reason="judge unavailable")
    return QualityDetection(**result)


MAIN_METRICS = {
    "first_turn_diversity",
    "intent_decomposition",
    "termination_f1",
    "ai_detector_score",
    "role_adherence",
    "intent_adherence",
}


def _as_test_case(row: dict[str, Any]) -> TestCase:
    case_id = str(row.get("id") or row.get("case_id") or "")
    if not case_id:
        case_id = "userllm_case"
    return TestCase(id=case_id, data=row)


def _parse_related_metrics(raw: Any) -> set[str]:
    """
    Parse related metrics from:
    - related_metrics: list[str] / set[str] / comma-separated string
    - related_metric: same forms (backward-compatible alias)
    """
    items: list[str] = []
    if raw is None:
        return set()
    if isinstance(raw, str):
        items = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
    elif isinstance(raw, (list, tuple, set)):  # noqa: UP038
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        s = str(raw).strip()
        if s:
            items = [s]
    return {m for m in items if m in MAIN_METRICS}


def _infer_related_metrics(row: dict[str, Any]) -> set[str]:
    """Fallback inference when related_metrics is not provided."""
    source = str(row.get("source") or "").strip().lower()
    if source == "commonsense_qa":
        return {"role_adherence"}
    if source == "natural_questions":
        return {"intent_adherence"}
    if source == "prism":
        return {"first_turn_diversity", "intent_decomposition", "termination_f1", "ai_detector_score"}
    return set()


def _resolve_related_metrics(row: dict[str, Any]) -> set[str]:
    parsed = _parse_related_metrics(row.get("related_metrics"))
    if not parsed:
        parsed = _parse_related_metrics(row.get("related_metric"))
    if parsed:
        return parsed
    return _infer_related_metrics(row)


def _is_cjk_or_jp_char(ch: str) -> bool:
    return (
        "\u4e00" <= ch <= "\u9fff"  # CJK Unified Ideographs
        or "\u3040" <= ch <= "\u309f"  # Hiragana
        or "\u30a0" <= ch <= "\u30ff"  # Katakana
    )


def _tokenize_compatible(text: str) -> list[str]:
    """
    Fallback tokenizer mirroring _tokenize behavior when `regex` package is unavailable.
    """
    s = str(text or "").lower()
    tokens: list[str] = []
    buf: list[str] = []

    def flush_word() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in s:
        if _is_cjk_or_jp_char(ch):
            flush_word()
            tokens.append(ch)
            continue

        is_word_char = ch.isalnum() or ch == "_"
        if is_word_char:
            buf.append(ch)
            continue

        flush_word()
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            tokens.append(ch)

    flush_word()
    return tokens


def _intent_1gram_overlap_compatible(intent: str, user_turn: str) -> float:
    """Use helpers implementation when possible; fallback if regex dep is unavailable."""
    try:
        return _intent_1gram_overlap(intent, user_turn)
    except ModuleNotFoundError as e:
        if str(e).strip() != "No module named 'regex'":
            raise
        intent_set = set(_filter_tokens_for_overlap(_tokenize_compatible(intent)))
        user_set = set(_filter_tokens_for_overlap(_tokenize_compatible(user_turn)))
        if not intent_set or not user_set:
            return 0.0
        return len(intent_set & user_set) / len(user_set)


def _first_turn_diversity_compatible(texts: list[str]) -> float:
    """Pairwise 1-gram Jaccard diversity; uses regex tokenizer with fallback."""
    try:
        import regex

        pattern = r"\b\w+\b|[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]|\d|[\p{P}\p{S}]"

        def tokenize(t: str) -> list[str]:
            return regex.findall(pattern, (t or "").lower())
    except ModuleNotFoundError:
        tokenize = _tokenize_compatible

    token_sets = []
    for t in texts:
        s = set(tokenize(t or ""))
        if s:
            token_sets.append(s)

    if len(token_sets) < 2:
        return 0.0
    total = 0.0
    n = 0
    for a, b in combinations(token_sets, 2):
        union = a | b
        if not union:
            continue
        total += len(a & b) / len(union)
        n += 1
    return 1.0 - (total / n) if n else 0.0


def compute_userllm_aggregates(results: list[dict]) -> dict[str, float]:
    """
    Aggregate six main userLLM metrics.
    Returns only:
    - first_turn_diversity
    - intent_decomposition
    - termination_f1
    - ai_detector_score
    - role_adherence
    - intent_adherence
    """
    first_turn_outputs: list[str] = []
    intent_scores: list[float] = []
    termination_tp = 0
    termination_fp = 0
    termination_fn = 0
    ai_scores: list[float] = []
    role_scores: list[float] = []
    intent_adherence_scores: list[float] = []

    for result in results:
        if not isinstance(result, dict):
            continue
        enabled = _parse_related_metrics(result.get("related_metrics"))
        if not enabled:
            enabled = MAIN_METRICS.copy()

        output = str(result.get("generated_output", "") or "")
        if "first_turn_diversity" in enabled and result.get("is_first_turn") and output:
            first_turn_outputs.append(output)

        if "intent_decomposition" in enabled and result.get("has_intent"):
            try:
                intent_scores.append(float(result.get("intent_decomposition", 0.0)))
            except Exception:
                pass

        pred = result.get("pred_endconversation")
        true = result.get("true_endconversation")
        if "termination_f1" in enabled and isinstance(pred, bool) and isinstance(true, bool):
            if pred and true:
                termination_tp += 1
            elif pred and (not true):
                termination_fp += 1
            elif (not pred) and true:
                termination_fn += 1

        ai = result.get("ai_detector_score")
        if "ai_detector_score" in enabled and ai is not None:
            try:
                ai_scores.append(float(ai))
            except Exception:
                pass

        role = result.get("role_adherence")
        if "role_adherence" in enabled and role is not None:
            try:
                role_scores.append(float(role))
            except Exception:
                pass

        ia = result.get("intent_adherence")
        if "intent_adherence" in enabled and ia is not None:
            try:
                intent_adherence_scores.append(float(ia))
            except Exception:
                pass

    first_turn_diversity = _first_turn_diversity_compatible(first_turn_outputs)

    denom_p = termination_tp + termination_fp
    denom_r = termination_tp + termination_fn
    precision = (termination_tp / denom_p) if denom_p > 0 else 0.0
    recall = (termination_tp / denom_r) if denom_r > 0 else 0.0
    termination_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "first_turn_diversity": first_turn_diversity,
        "intent_decomposition": float(sum(intent_scores) / len(intent_scores)) if intent_scores else 0.0,
        "termination_f1": float(termination_f1),
        "ai_detector_score": float(sum(ai_scores) / len(ai_scores)) if ai_scores else 0.0,
        "role_adherence": float(sum(role_scores) / len(role_scores)) if role_scores else 0.0,
        "intent_adherence": float(sum(intent_adherence_scores) / len(intent_adherence_scores))
        if intent_adherence_scores
        else 0.0,
    }


async def _pangram_ai_detector_score(row: dict[str, Any], output: str) -> Optional[float]:
    text = output.replace("<|endconversation|>", "").strip()
    if not text:
        return None
    api_key = str(os.environ.get("PANGRAM_API_KEY") or "b7498e37-ae40-4365-88dd-037f1275abaf").strip()
    try:
        response = await asyncio.to_thread(analyze_text, output, api_key=api_key)
    except Exception:
        return None
    return _extract_ai_detector_score(response)


async def _llm_ai_detector_score(output: str) -> Optional[float]:
    text = output.replace("<|endconversation|>", "").strip()
    if not text:
        return None
    prompt = (
        "You are an AI detection judge. Read the following text and decide whether it was written by an AI or a human.\n\n"
        f"Text:\n{text}\n\n"
        "Reply with exactly one word: AI or HUMAN."
    )
    try:
        result = await call_openai(
            [{"role": "user", "content": prompt}],
            model=get_judge_model("gpt-5.4-nano"),
            reasoning_effort=get_judge_reasoning("low"),
        )
        label = (result or "").strip().upper()
        if label == "AI":
            return 0.0
        if label == "HUMAN":
            return 1.0
        return None
    except Exception:
        return None


async def _maybe_ai_detector_score(row: dict[str, Any], output: str) -> Optional[float]:
    """Try Pangram API first; fall back to LLM judge if unavailable."""
    # score = await _pangram_ai_detector_score(row, output)
    # if score is not None:
    #     return score
    return await _llm_ai_detector_score(output)


async def _maybe_intent_adherence(
    row: dict[str, Any],
    output: str,
) -> Optional[float]:
    tc = _as_test_case(row)
    fields = _extract_intent_adherence_fields(tc)
    if not fields or not output:
        return None
    question, suggestion = fields

    prompt = INTENT_ADHERENCE_JUDGE_PROMPT.format(
        question=question,
        refusal=suggestion,
        output=output,
    )
    try:
        content = await call_openai(
            [{"role": "user", "content": prompt}],
            model=get_judge_model("gpt-5.4-nano"),
            reasoning_effort=get_judge_reasoning("low"),
        )
        label = content.strip().upper()
        if label == "REFUSED":
            return 1.0
        if label == "ACCEPTED":
            return 0.0
        return None
    except Exception:
        return None


async def agent_loop(data, context):
    """
    userLLM single-case rollout with metric fields.
    reward is a placeholder for now.
    """
    row = data["extra_info"]
    # Determine reward type from source.
    source = str(row.get("source") or "").strip().lower()
    if source == "commonsense_qa":
        reward_metric = "role_adherence"
    elif source == "natural_questions":
        reward_metric = "intent_adherence"
    elif "prism" in source:
        reward_metric = "prism"
    else:
        reward_metric = None

    tc = _as_test_case(row)
    intent = _extract_intent(tc)
    has_intent = bool(str(intent or "").strip())

    conversation_history = row.get("conversation_history")
    if conversation_history == "":
        prompt = _format_first_turn_prompt(intent)
    else:
        prompt = _format_sequential_turn_prompt(intent, str(conversation_history or ""))

    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)
    response = await agent.step()
    output = remove_think(response) if response else ""

    if not output.strip():
        import logging as _log

        _log.getLogger(__name__).warning(f"[userllm] empty output after remove_think (raw len={len(response or '')})")

    pred_endconversation = "<|endconversation|>" in (output or "")
    if pred_endconversation:
        output = output.split("<|endconversation|>")[0].strip() + "<|endconversation|>"

    # Base sub_scores with all keys so _postprocess can build uniform arrays across the batch.
    sub_scores: dict[str, Any] = {
        "userllm/intent_decomposition": None,
        "userllm/termination_f1": None,
        "userllm/ai_detector_score": None,
        "userllm/role_adherence": None,
        "userllm/intent_adherence": None,
    }

    if reward_metric == "prism":
        # Per-instance: avg of intent_decomposition + termination_f1 + ai_detector_score.
        # first_turn_diversity is batch-level only, excluded here.
        intent_decomp = _intent_1gram_overlap_compatible(intent, output) if intent else 0.0
        true_end = _to_optional_bool(row.get("is_last_turn"))
        term_score = 1.0 if (true_end is not None and pred_endconversation == true_end) else 0.0
        ai_score, quality_result = await asyncio.gather(
            _maybe_ai_detector_score(row, output),
            quality_judge(output, intent, source),
        )
        ai_score = ai_score if ai_score is not None else 0.0
        reward = (intent_decomp + term_score + ai_score) / 3.0
        sub_scores.update(
            {
                "userllm/intent_decomposition": intent_decomp,
                "userllm/termination_f1": term_score,
                "userllm/ai_detector_score": ai_score,
            }
        )
    elif reward_metric == "role_adherence":
        role_reward = None
        choices = _extract_choice_texts(row)
        if choices:
            out_norm = _normalize_for_choice_match(output)
            mentioned = sum(1 for c in choices if c and _normalize_for_choice_match(c) in out_norm)
            # Align with suite.py: attempt=1 iff output mentions exactly 1-2 choice texts;
            # ignore cases that mention ALL choices (usually question repetition).
            if mentioned != len(choices):
                attempt = 1 if mentioned in (1, 2) else 0
                role_reward = float(1 - attempt)
        role_reward = role_reward if role_reward is not None else 0.0
        reward = role_reward
        quality_result = await quality_judge(output, intent, source)
        sub_scores["userllm/role_adherence"] = role_reward
    elif reward_metric == "intent_adherence":
        ia_reward, quality_result = await asyncio.gather(
            _maybe_intent_adherence(row, output),
            quality_judge(output, intent, source),
        )
        ia_reward = ia_reward if ia_reward is not None else 0.0
        reward = ia_reward
        sub_scores["userllm/intent_adherence"] = ia_reward
    else:
        reward = 0.0
        quality_result = await quality_judge(output, intent, source)

    # if quality_result.risk_level == "high":
    #     reward = float(reward / 2)
    #     import logging as _log
    #     _log.getLogger(__name__).warning(f"[quality_judge] HIGH risk — reward zeroed. Reason: {quality_result.reason}")

    sub_scores.update(
        {
            "userllm/quality_low": int(quality_result.risk_level == "low"),
            "userllm/quality_medium": int(quality_result.risk_level == "medium"),
            "userllm/quality_high": int(quality_result.risk_level == "high"),
        }
    )

    USE_EDITLENS = False
    editlens = None
    editlens_failed = False
    if USE_EDITLENS:
        editlens = await editlens_score(output) if output else 0.0
        editlens_failed = editlens is None
        import logging as _log

        if editlens_failed:
            _log.getLogger(__name__).warning("[editlens] API unavailable — no penalty applied.")
        elif editlens > 0.67 and context.is_train:
            reward = float(reward / 8)
            _log.getLogger(__name__).warning(f"[editlens] HIGH AI-like (score={editlens:.3f}) — reward /8.")
        elif editlens > 0.33 and context.is_train:
            reward = float(reward / 4)
            _log.getLogger(__name__).warning(f"[editlens] MEDIUM AI-like (score={editlens:.3f}) — reward /4.")
        sub_scores.update(
            {
                "userllm/editlens_score": 0 if editlens_failed else editlens,
                "userllm/editlens_medium": 0 if editlens_failed else int(0.33 < editlens <= 0.67),
                "userllm/editlens_high": 0 if editlens_failed else int(editlens > 0.67),
                "userllm/editlens_failed": int(editlens_failed),
            }
        )

    output_text = output  # keep text before get_agent_output reassigns 'output'

    # ===========================================================================
    teacher_prompt = None
    hint = None
    if (
        getattr(context.config.algorithm, "use_opd", False)
        or getattr(context.config.algorithm, "agent_version", None) == "copy"
    ):
        from agents.userllm.hint import generate_hint, get_teacher_prompt

        hint = await generate_hint(
            row=row,
            output=output_text,
            reward=reward,
            sub_scores=sub_scores,
        )
        teacher_prompt_str = get_teacher_prompt(row, hint)
        teacher_prompt = [
            {"role": "system", "content": ""},
            {"role": "user", "content": teacher_prompt_str},
        ]
    # ===========================================================================

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "userllm/reward": reward,
            "all/score": reward,
            "all/score_v1": reward,
            **sub_scores,
        },
        teacher_prompt=teacher_prompt,
    )
    await process_post_chat(data, context, agent.chat, output)

    if getattr(context.config.algorithm, "agent_version", None) == "copy" and context.is_train:
        from agents.userllm.hint_agent import agent_loop as hint_agent_loop

        data["extra_info"]["hint"] = hint or ""
        data["extra_info"]["rollout"] = {
            "userllm/reward": reward,
            "output": output_text,
            **sub_scores,
        }
        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = "hint_agent"
        return [output, copy_agent_output, hint_agent_output]

    return output

    return {
        "reward": reward,
        "chat": [
            {"role": "system", "content": prompt},
            {"role": "assistant", "content": output},
        ],
        "generated_output": output,
        "related_metrics": sorted(list(related_metrics)),  # noqa: F821
        "is_first_turn": is_first_turn,  # noqa: F821
        "has_intent": has_intent,
        "pred_endconversation": pred_endconversation if "termination_f1" in related_metrics else None,  # noqa: F821
        "true_endconversation": true_endconversation,  # noqa: F821
    }
