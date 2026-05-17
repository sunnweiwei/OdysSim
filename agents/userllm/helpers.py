"""
Helpers for the userLLM agent: prompts, TestCase, and suite utility functions.
Ported from eval/suites/userLLM so that the agent has no dependency on eval.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Prompts ──────────────────────────────────────────────────────────────────

FIRST_TURN_PROMPT = """You are a human user interacting with an AI system to {INTENT}.
Users can make typos, they don't always use perfect punctuation, and they tend to be lazy because typing requires effort.
You have to also split information across turns and not give everything at the start.
However, you should not make overdo these things in your outputs, you must realistically act like a human.
Generate the first prompt you would say to the system to achieve your goal."""

SEQUENTIAL_TURN_PROMPT = """You are a human user interacting with an AI system to {INTENT}.
Users can make typos, they don't always use perfect punctuation, and they tend to be lazy because typing requires effort.
You have to also split information across turns and not give everything at the start.
However, you should not make overdo these things in your outputs, you must realistically act like a human.
Here is your conversation so far with the AI system:
====================================================
{CONVERSATION_HISTORY}
====================================================
If your goal of your conversation is answered by this response, respond ONLY with "<|endconversation|>"
If not, generate the next prompt you would say to the system to achieve your goal."""

INTENT_ADHERENCE_JUDGE_PROMPT = """You are given a conversation between a user and an AI system.
User Question: {question}
AI Response: {refusal}
User Reply: {output}
Classify if the user accepted the AI's suggestion or repeated their original question.
If the user's reply repeats or rephrases their original question, output only: REFUSED
If the user's reply follows the AI's suggestion, output only: ACCEPTED
"""

# ── TestCase ──────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ── Constants ─────────────────────────────────────────────────────────────────

_INTENT_PREFIX = "You are a user chatting with an assistant language model to"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "might", "my", "of", "on",
    "or", "our", "should", "so", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your",
}


# ── Tokenizer (requires `regex` package) ─────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Tokenize text using the `regex` package (may raise ModuleNotFoundError)."""
    import regex
    pattern = r'\b\w+\b|[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]|\d|[\p{P}\p{S}]'
    return regex.findall(pattern, (text or "").lower())


# ── Token helpers ─────────────────────────────────────────────────────────────

def _is_cjk_char(s: str) -> bool:
    return len(s) == 1 and (
        ("\u4e00" <= s <= "\u9fff") or ("\u3040" <= s <= "\u30ff")
    )


def _filter_tokens_for_overlap(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens or []:
        if not t or t in _STOPWORDS:
            continue
        if all(ch in string.punctuation for ch in t):
            continue
        if t.isalnum() or _is_cjk_char(t):
            out.append(t)
    return out


def _intent_1gram_overlap(intent: str, user_turn: str) -> float:
    """
    Overlap of stopword-filtered 1-grams between a user turn and the intent.
    Returns |I ∩ U| / |U|. Lower is better. Requires `regex`.
    """
    intent_set = set(_filter_tokens_for_overlap(_tokenize(intent or "")))
    if not intent_set:
        return 0.0
    user_set = set(_filter_tokens_for_overlap(_tokenize(user_turn or "")))
    if not user_set:
        return 0.0
    return len(intent_set & user_set) / len(user_set)


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_ai_detector_score(obj: Any) -> Optional[float]:
    """Extract human-likeness score (1.0 = human, 0.0 = AI) from Pangram API response.

    Supports v3 response (fraction_human) and legacy v2 response (ai_likelihood).
    """
    if not isinstance(obj, dict):
        return None
    # v3: fraction_human is directly the human score (1.0 = human, 0.0 = AI)
    if "fraction_human" in obj:
        v = obj["fraction_human"]
        if isinstance(v, bool) or v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None
    # v2 legacy: ai_likelihood (0.0 = human, 1.0 = AI) — invert to match convention
    v = obj.get("ai_likelihood", obj.get("aiLikelihood"))
    if isinstance(v, bool) or v is None:
        return None
    try:
        return 1.0 - float(v)
    except Exception:
        return None


def _extract_choice_texts(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    ch = data.get("choices")
    if isinstance(ch, str):
        try:
            import json as _json
            ch = _json.loads(ch)
        except Exception:
            return []
    out: List[str] = []
    if isinstance(ch, dict):
        texts = ch.get("text")
        if isinstance(texts, list):
            out = [str(x or "").strip() for x in texts]
    elif isinstance(ch, list):
        for c in ch:
            if isinstance(c, dict):
                out.append(str(c.get("text") or "").strip())
            else:
                out.append(str(c or "").strip())
    return [x for x in out if x]


def _normalize_intent(intent: str) -> str:
    s = str(intent or "").strip()
    if not s:
        return ""
    low = s.lower()
    pref_low = _INTENT_PREFIX.lower()
    i = low.find(pref_low)
    if i == -1:
        return s
    tail = s[i + len(_INTENT_PREFIX):].strip()
    while tail[:1] in (":", "-", "\u2014"):
        tail = tail[1:].lstrip()
    return tail or s


def _extract_intent(tc: TestCase) -> str:
    raw = tc.data.get("intent") or tc.data.get("intent_summary") or ""
    intent = _normalize_intent(str(raw))
    if intent:
        return intent
    opening = tc.data.get("opening_prompt")
    if isinstance(opening, str) and opening.strip():
        return opening.strip()
    history = tc.data.get("conversation_history")
    if isinstance(history, str) and history.strip():
        first = history.strip().splitlines()[0].strip()
        if first.lower().startswith("<user>:"):
            first = first.split(":", 1)[1].strip()
        return first
    return ""


def _extract_intent_adherence_fields(tc: TestCase) -> Optional[Tuple[str, str]]:
    if not isinstance(tc.data, dict):
        return None
    q = str(tc.data.get("question") or "").strip()
    a = str(tc.data.get("assistant_suggestion_turn") or "").strip()
    if q and a:
        return (q, a)
    return None


def _normalize_for_choice_match(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f" {s} " if s else " "


def _format_first_turn_prompt(intent: str) -> str:
    return FIRST_TURN_PROMPT.replace("{INTENT}", intent)


def _format_sequential_turn_prompt(intent: str, conversation_history: str) -> str:
    return SEQUENTIAL_TURN_PROMPT.replace("{INTENT}", intent).replace("{CONVERSATION_HISTORY}", conversation_history)


def _is_first_turn_from_metadata(metadata: Dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return True
    t = metadata.get("turn")
    if t is None:
        return True
    try:
        return int(t) == 0
    except Exception:
        return True


def _to_optional_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return None