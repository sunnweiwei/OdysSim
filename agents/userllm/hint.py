"""
userLLM hint generation — reflect on a rollout and produce coaching notes
for a teacher prompt.

Supports all three userLLM reward metrics:
  - prism:             intent_decomposition + termination_f1 + ai_detector_score
  - commonsense_qa:    role_adherence (don't mention answer choices)
  - natural_questions: intent_adherence (persist with original question)

All hints are fully rule-based — no LLM calls. Each hint is concise, factual,
and directly actionable based on the available scoring information.
"""

import logging
import re
from typing import Any, Dict, List, Set, Tuple

from agents.userllm.helpers import (
    TestCase,
    _STOPWORDS,
    _extract_choice_texts,
    _extract_intent,
    _extract_intent_adherence_fields,
    _format_first_turn_prompt,
    _format_sequential_turn_prompt,
    _normalize_for_choice_match,
    _to_optional_bool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic helpers — compute exact facts about what went wrong
# ---------------------------------------------------------------------------

def _simple_content_tokens(text: str) -> Set[str]:
    """Lowercase alphanum tokens, stopword-filtered, length > 2. No regex dep."""
    words = re.findall(r'\b[a-zA-Z0-9]+\b', (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _intent_token_diagnosis(intent: str, output: str) -> Tuple[List[str], List[str], float]:
    """
    Returns (missing_tokens, present_tokens, approx_overlap_ratio).
    missing_tokens: intent keywords NOT found in output → model should use these
    present_tokens: intent keywords found in output
    overlap_ratio:  |present| / |intent| (proxy for intent_decomposition score)
    """
    intent_tokens = _simple_content_tokens(intent)
    output_tokens = _simple_content_tokens(output)
    if not intent_tokens:
        return [], [], 1.0
    present = sorted(intent_tokens & output_tokens)
    missing = sorted(intent_tokens - output_tokens)
    ratio = len(present) / len(intent_tokens)
    return missing, present, ratio


def _termination_diagnosis(row: Dict[str, Any], pred_endconversation: bool) -> str:
    """Exact rule-based diagnosis of termination_f1 failure."""
    raw = row.get("is_last_turn")
    if raw is None:
        # Can't determine — give general rule
        return (
            "Termination rule: output <|endconversation|> at the END of your reply "
            "ONLY when the AI has fully answered your goal. If not yet answered, continue."
        )
    # Normalise to bool
    if isinstance(raw, bool):
        is_last = raw
    elif isinstance(raw, (int, float)):
        is_last = bool(int(raw))
    elif isinstance(raw, str):
        is_last = raw.strip().lower() in ("true", "1", "yes", "y")
    else:
        is_last = False

    if is_last and not pred_endconversation:
        return (
            "TERMINATION ERROR (score=0): The conversation WAS at its final turn "
            "(the AI's response had already answered your goal), but you did NOT "
            "append <|endconversation|>. "
            "Fix: end your reply with <|endconversation|> when the goal is satisfied."
        )
    elif not is_last and pred_endconversation:
        return (
            "TERMINATION ERROR (score=0): The conversation was NOT at its final turn "
            "(the AI's response had NOT yet fully answered your goal), but you output "
            "<|endconversation|> and ended early. "
            "Fix: continue asking follow-up questions instead of ending."
        )
    else:
        return (
            "Termination was CORRECT (score=1). "
            "Keep doing this: end with <|endconversation|> only when goal is fully answered."
        )


def _role_adherence_diagnosis(choices: List[str], output: str) -> Tuple[List[str], str]:
    """
    Returns (detected_choices, diagnosis_str).
    detected_choices: choice texts the scorer found in the output.
    Mirrors the exact normalization used by the scorer (_normalize_for_choice_match).
    """
    out_norm = _normalize_for_choice_match(output)
    detected = [c for c in choices if c and _normalize_for_choice_match(c) in out_norm]
    if not detected:
        return [], (
            "role_adherence = 1 (PASSED): None of the answer choices appeared in your output. "
            "Keep doing this — ask naturally without referencing any option text."
        )
    else:
        listed = ", ".join(f'"{c}"' for c in detected)
        all_listed = "\n".join(f'  - "{c}"' for c in choices)
        return detected, (
            f"role_adherence = 0 (FAILED): The scorer detected the following choice text(s) "
            f"in your output: {listed}.\n"
            f"The scorer strips punctuation and lowercases before matching, so even partial "
            f"or rephrased mentions count.\n"
            f"All choices to avoid (verbatim and any paraphrase):\n{all_listed}\n"
            f"Fix: ask your question as if you have never seen these options. "
            f"Do not quote, list, or allude to any of them."
        )


# ---------------------------------------------------------------------------
# Hint text builders (one per reward metric)
# ---------------------------------------------------------------------------

def _build_prism_hint(
    intent: str,
    output: str,
    sub_scores: Dict[str, Any],
    row: Dict[str, Any],
) -> str:
    """Forward-looking imperative hints for the prism metric. One bullet per failing sub-score."""
    intent_decomp = float(sub_scores.get("userllm/intent_decomposition") or 0.0)
    term_score = float(sub_scores.get("userllm/termination_f1") or 0.0)

    pred_endconversation = "<|endconversation|>" in output
    bullets = []

    # termination_f1 — fully deterministic from is_last_turn
    if term_score < 1.0:
        raw = row.get("is_last_turn")
        is_last = _to_optional_bool(raw)
        if is_last is True and not pred_endconversation:
            bullets.append(
                "CRITICAL: Your response for this turn must be ONLY <|endconversation|> and nothing else."
            )
        elif is_last is False and pred_endconversation:
            bullets.append(
                "CRITICAL: Do NOT output <|endconversation|>. Your goal is not yet fully answered — write your next question."
            )

    # intent_decomposition — score = overlap(intent_keywords, output) / len(output)
    # Every extra non-goal word dilutes the score. Short, focused messages score highest.
    if intent_decomp < 0.8:
        missing, _, _ = _intent_token_diagnosis(intent, output)
        if missing:
            bullets.append(
                f"Keep your message SHORT and use these goal keywords directly: "
                f"{', '.join(missing[:8])}. Extra words not about your goal lower the score."
            )

    if not bullets:
        return ""

    return "\n".join(f"• {b}" for b in bullets)


def _build_role_adherence_hint(
    intent: str,
    choices: List[str],
    output: str,
    reward: float,
) -> str:
    """Forward-looking hint — warns exactly which choice phrases to avoid."""
    detected, _ = _role_adherence_diagnosis(choices, output)
    if not detected:
        return ""

    detected_str = ", ".join(f'"{c}"' for c in detected)
    return (
        f"WARNING: Do NOT use these exact words or phrases in your response: {detected_str}. "
        "The scorer detects any substring match. "
        "Ask about the topic using completely different vocabulary."
    )


def _build_intent_adherence_hint(
    question: str,
    assistant_suggestion: str,
    reward: float,
) -> str:
    """Forward-looking hint for intent_adherence — tells model exactly how to persist."""
    if reward >= 1.0:
        return ""
    return (
        "Re-ask your original question directly — do NOT acknowledge, thank, or follow the AI's suggestion. "
        "Even 'That said...' or 'While that helps...' counts as accepting. "
        f"Just re-state: \"{question}\""
    )


# ---------------------------------------------------------------------------
# Main hint generation
# ---------------------------------------------------------------------------

async def generate_hint(
    row: Dict[str, Any],
    output: str,
    reward: float,
    sub_scores: Dict[str, Any],
) -> str:
    """Generate a coaching hint from a completed userLLM rollout.

    Parameters
    ----------
    row:        The extra_info dict from the data sample.
    output:     The model's generated text output (after remove_think).
    reward:     The scalar reward from the rollout.
    sub_scores: The sub_scores dict (keys like 'userllm/intent_decomposition').

    Returns a hint string (markdown), or "" on failure.
    Rule-based metrics are diagnosed directly; LLM is called only when needed.
    """
    source = str(row.get("source") or "").strip().lower()
    tc = TestCase(id="hint", data=row)
    intent = _extract_intent(tc)
    try:
        if "prism" in source:
            return _build_prism_hint(
                intent=intent,
                output=output,
                sub_scores=sub_scores,
                row=row,
            )

        elif source == "commonsense_qa":
            choices = _extract_choice_texts(row)
            return _build_role_adherence_hint(
                intent=intent,
                choices=choices,
                output=output,
                reward=reward,
            )

        elif source == "natural_questions":
            fields = _extract_intent_adherence_fields(tc)
            question, suggestion = fields if fields else ("(unknown)", "(unknown)")
            return _build_intent_adherence_hint(
                question=question,
                assistant_suggestion=suggestion,
                reward=reward,
            )

        else:
            return ""

    except Exception as e:
        logger.warning(f"userllm hint generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Teacher prompt builder
# ---------------------------------------------------------------------------

def get_teacher_prompt(row: Dict[str, Any], hint: str) -> str:
    """Build the agent input prompt augmented with hint coaching notes.

    Includes reference material when available:
    - commonsense_qa:    the answer choices to avoid (so model can reason about them)
    - natural_questions: the original question + AI suggestion to refuse

    Parameters
    ----------
    row:  The extra_info dict from the data sample.
    hint: The coaching brief from generate_hint().
    """
    from agents.utils import truncate_text_left
    tc = TestCase(id="teacher", data=row)
    intent = _extract_intent(tc)
    conversation_history = truncate_text_left(str(row.get("conversation_history") or ""), 2000)
    source = str(row.get("source") or "").strip().lower()

    if conversation_history == "":
        base_prompt = _format_first_turn_prompt(intent)
    else:
        base_prompt = _format_sequential_turn_prompt(intent, conversation_history)

    # Include reference so the model can reason about what to avoid
    reference_section = ""
    if source == "commonsense_qa":
        choices = _extract_choice_texts(row)
        if choices:
            choices_str = "\n".join(f"  - {c}" for c in choices)
            reference_section = (
                "\n\n[Private reference — answer choices you must NOT mention "
                "(any substring match counts):\n"
                f"{choices_str}\n]"
            )
    elif source == "natural_questions":
        fields = _extract_intent_adherence_fields(tc)
        if fields:
            question, suggestion = fields
            reference_section = (
                "\n\n[Private reference:\n"
                f"  Your original question: {question}\n"
                f"  AI suggestion you must NOT accept: {suggestion}\n"
                "]"
            )

    hint_section = f"\n\n---\n{hint}\n---" if hint else ""

    return base_prompt + reference_section + hint_section