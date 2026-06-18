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

import re
from collections import Counter
from typing import Any

FIELD_ORDINAL = {
    "task_success": {
        "No - Task failed": 1,
        "No - Due to a policy issue, which the agent clearly explained": 2,
        "Partially - Some progress": 3,
        "Yes - Task completed": 4,
        "Fully - Exceeded expectations": 5,
    },
    "efficiency": {
        "Very inefficient - Too many steps": 1,
        "Somewhat inefficient": 2,
        "About right": 3,
        "Very efficient": 4,
    },
    "question_amount_preference": {
        "Too many": 1,
        "About right": 2,
        "Too few": 1,
    },
    "answer_effort_time": {"High": 1, "Medium": 2, "Low": 3},
    "human_like": {"No": 1, "Partially": 2, "Yes": 3},
    "interaction_flow": {
        "Not smooth": 1,
        "OK": 2,
        "Smooth": 3,
        "Excellent": 4,
    },
    "overall_score": {
        "1 (Very poor)": 1,
        "2 (Poor)": 2,
        "3 (Acceptable)": 3,
        "4 (Good)": 4,
        "5 (Excellent)": 5,
    },
    "reuse": {
        "Absolutely no": 1,
        "No": 2,
        "Maybe": 3,
        "Yes": 4,
        "Absolutely yes": 5,
    },
}

DIMENSION_KEYS = {
    "D1_conv": [
        "politeness_rate",
        "short_msg_rate",
        "formality_rate",
        "ack_only_rate",
        "verbosity_cv",
        "repeat_rate",
        "id_confuse_rate",
    ],
    "D2_info": [
        "info_frontload",
        "id_density",
        "user_words_per_turn",
        "opening_words",
    ],
    "D3_clarif": [
        "uncertainty_rate",
        "certainty_rate",
        "pushback_q_rate",
        "clarify_q_rate",
        "info_q_rate",
    ],
    "D4_react": ["emotion_rate", "accusation_rate", "pivot_rate"],
}


_AGENT_MARKUP_RE = re.compile(
    r"<\|think\|>.*?<\|/think\|>|<function=.*?(?:</function>|$)|<\|tool\|>.*?<\|/tool\|>",
    re.DOTALL,
)
POLITE_PATTERNS = re.compile(
    r"\b(please|thank you|thanks|sorry|excuse me|appreciate|grateful|"
    r"kind of you|wonderful|courteous|considerate|i understand|"
    r"no worries|my apologies|pardon)\b"
)
EMOTION_PATTERNS = re.compile(
    r"\b(frustrated|upset|angry|confused|worried|annoyed|furious|"
    r"ridiculous|terrible|horrible|awful|ugh|"
    r"stressed|anxious|nervous|uncomfortable|panic|disappointed|disappointing|"
    r"irritated|irritating|aggravated|outraged|outrageous|exasperated|"
    r"bothered|disgusted|miserable|devastated|heartbroken|desperate)\b"
)
EMDASH_RE = re.compile(r"[\u2014\u2013]")
ACK_ONLY_RE = re.compile(
    r"^(yes|yeah|yea|yep|yup|ok|okay|okey|sure|right|fine|great|perfect|"
    r"awesome|cool|agree|agreed|correct|absolutely|indeed|alright|"
    r"got it|go ahead|sounds good|that works|proceed|confirm|do it|"
    r"no problem|thank you|thanks|please|mhm|uh-huh|aight)[\s!.\-,]*$"
)
ID_DENSITY_RE = re.compile(
    r"(\b[a-z0-9]{6,}\b|#?\w*\d{5,}\w*|"
    r"\w+_\w+_\d{3,}|"
    r"gift_card_\w+|credit_card_\w+|paypal_\w+|"
    r"\w+@\w+\.\w+)"
)
UNCERTAINTY_RE = re.compile(
    r"\b(i think|not sure|i don'?t remember|i don'?t recall|i believe|probably|"
    r"maybe|i forgot|don'?t know|i guess|i'?m not sure|if i recall|"
    r"i don'?t have|can'?t recall|can'?t find|i'?m unsure|i might|"
    r"perhaps|apparently|sort of|kind of|somewhat|partly|possibly|"
    r"vaguely|roughly|hesitant|doubtful|confused|uncertain|dunno)\b"
)
CERTAINTY_RE = re.compile(
    r"\b(definitely|absolutely|certainly|exactly|precisely|clearly|always|"
    r"never|completely|entirely|fully|obvious|obviously|undoubtedly|"
    r"without a doubt|for sure|of course|guaranteed|no doubt)\b"
)
PUSHBACK_Q_RE = re.compile(
    r"\b(are you sure|that'?s (not right|wrong|incorrect|not what)|"
    r"i already (told|said|gave|provided|mentioned)|you already (asked|have)|"
    r"try again|check again|that can'?t be (right|correct)|"
    r"that'?s (crazy|nonsense|ridiculous|absurd)|this is (not fair|nonsense)|"
    r"why (can'?t|won'?t|didn'?t|isn'?t|aren'?t|would)|"
    r"doesn'?t (seem|sound|look|make sense)|not what i (asked|said|meant|wanted)|"
    r"do(n'?t| not) ask me .* again|i would never)\b"
)
CLARIFY_Q_RE = re.compile(
    r"\b(what do you mean|what does that mean|what did you mean|"
    r"could you (clarify|elaborate|specify)|can you (explain|clarify|elaborate)|"
    r"i don'?t understand|i dont understand|"
    r"what'?s the difference|whats the difference|"
    r"what exactly|what specifically|"
    r"which (one|reservation|order|flight|option|plan|account|item|product) (is|was|do|should|did|would)|"
    r"i'?m not sure which|im not sure which|which is which|"
    r"you (said|mentioned|told me|wrote|indicated) .{0,30}\?|"
    r"that mean|what does that (involve|include|entail|look like)|"
    r"(sorry|wait),? (what|which|how|i don'?t)|"
    r"can you (be more specific|give me more detail|break that down|walk me through)|"
    r"i'?m (confused|lost)|im (confused|lost)|"
    r"(how|what) (exactly|specifically) (does|do|is|are|would|will|should)|"
    r"could you (repeat|say) that|can you (repeat|say) that|"
    r"what (is|are) the (options|choices|alternatives|details)|"
    r"i need (more|some) (info|information|details|clarification)|"
    r"(so|wait),? (you'?re saying|does that mean|is that)|"
    r"meaning\??|how so\??|in what (way|sense)\??)\b"
)
INFO_Q_RE = re.compile(
    r"\b(what is (the|my)|what'?s (the|my)|how much|how many|"
    r"when (will|does|is|can|did)|where (is|are|was|can)|"
    r"how (do|can|would|should) i|what are (the|my)|"
    r"can you (check|tell|find|look|see|show|help|list)|"
    r"do you (have|know|see)|is (there|it|that) (a |any )?|"
    r"what options|how long|what.s the (status|price|cost|total|balance))\b"
)
ACCUSATION_RE = re.compile(
    r"\b(blame|fault|ridiculous|wrong|useless|incompetent|failure|"
    r"misleading|irresponsible|unacceptable|disgrace|insult|"
    r"scam|shame|stupid|trouble|disappointing|ruined)\b"
)
PIVOT_RE = re.compile(
    r"\b((wait|actually) (can|let|could|would)|i('?d| would) (like|prefer|rather)|"
    r"(can you|can we|could you) just|"
    r"instead|rather than|how about|what about|what if|"
    r"on second thought|"
    r"is there (a |any |another )?way to|"
    r"(let'?s|can we|lets) (try|do|go with|switch|change)|"
    r"maybe (we|i|you) (should|could|can)|"
    r"(change|switch|try) (it |that )?(to|something|a different))\b"
)
ID_CONFUSE_RE = re.compile(
    r"\b(how can i (help|assist)|how may i (help|assist)|"
    r"what can i (do|help|assist) (for|with)|"
    r"do you (want|require|wish|prefer)|would you (like|prefer)|"
    r"let me (check|look|verify|find|pull up|search|assist|help)|"
    r"i('?ll| will) (check|look into|verify|process|handle|assist|help)|"
    r"i('?m| am) (happy|glad|here) to (help|assist)|"
    r"is there anything else i can|"
    r"thank you for (calling|contacting|reaching|choosing|your patience)|"
    r"for (security|verification) (purposes|reasons)|"
    r"(may|could) i (verify)|"
    r"allow me to|permit me to|"
    r"your (order|account|reservation|booking|request|ticket|case|inquiry)|"
    r"i (see|understand) (that|your)|"
    r"according to (our|the) (records|system|policy)|"
    r"(our|the) (policy|system|records) (shows?|indicates?)|"
    r"i (can|could) (offer|suggest|recommend)|"
    r"(have you|did you) (tried?|considered?)|"
    r"i apologize for (the|any) (inconvenience|confusion|delay|trouble))\b"
)

_RATE_FEATURES: frozenset[str] = frozenset(
    DIMENSION_KEYS["D1_conv"]
    + DIMENSION_KEYS["D2_info"][:1]  # info_frontload
    + DIMENSION_KEYS["D3_clarif"]
    + DIMENSION_KEYS["D4_react"]
)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance**0.5


def _is_meta_message(message: dict[str, Any], source: str) -> bool:
    content = str(message.get("content", ""))
    if source == "human":
        if content.startswith("\\tau ") or content.startswith("\\reward"):
            return True
        if "<|canvas|>" in content or "<|highlight|>" in content or "<|survey|>" in content:
            return True
        if content.strip() == "/stop":
            return True
    elif source == "llm":
        if content.strip() == "###STOP###":
            return True
    return False


def _clean_conversation(conversation, source: str) -> list[tuple[str, str]]:
    return [
        (str(message.get("role", "")), str(message.get("content", "")))
        for message in conversation
        if not _is_meta_message(message, source)
    ]


def _strip_agent_markup(text: str) -> str:
    return _AGENT_MARKUP_RE.sub("", text)


def _word_count(text: str) -> int:
    return len(_strip_agent_markup(text).split())


def extract_conversation_features(
    conversation,
    source: str = "llm",
) -> dict[str, float] | None:
    """
    Extract per-conversation behavioral features (same schema as analyze_interaction.py).
    """
    turns = _clean_conversation(conversation, source)
    if not turns:
        return None

    user_turns = [(role, content) for role, content in turns if role == "user"]
    if not user_turns:
        return None

    n_turns = len(turns)
    word_counts = [_word_count(content) for _, content in user_turns]
    total_words = sum(word_counts)
    user_words_per_turn = _safe_mean([float(count) for count in word_counts])
    cleaned_texts = [_strip_agent_markup(content) for _, content in user_turns]
    lowered_texts = [text.lower() for text in cleaned_texts]

    politeness_rate = _safe_mean([1.0 if POLITE_PATTERNS.search(text) else 0.0 for text in lowered_texts])
    short_msg_rate = _safe_mean([1.0 if count <= 3 else 0.0 for count in word_counts])
    formality_rate = _safe_mean([1.0 if EMDASH_RE.search(text) else 0.0 for text in cleaned_texts])
    ack_only_rate = _safe_mean([1.0 if ACK_ONLY_RE.match(text.strip()) else 0.0 for text in lowered_texts])
    verbosity_cv = 0.0
    if user_words_per_turn > 0:
        word_counts_float = [float(count) for count in word_counts]
        verbosity_cv = _safe_std(word_counts_float) / user_words_per_turn

    trigram_counter: Counter[tuple[str, str, str]] = Counter()
    trigram_size = 3
    for text in lowered_texts:
        words = text.split()
        for index in range(len(words) - trigram_size + 1):
            trigram = tuple(words[index : index + trigram_size])
            trigram_counter[trigram] += 1
    has_duplicate_trigram = any(count > 10 for count in trigram_counter.values())
    repeat_rate = float(has_duplicate_trigram)

    id_confuse_rate = float(any(ID_CONFUSE_RE.search(text) is not None for text in lowered_texts))

    info_frontload = sum(word_counts[:2]) / total_words if total_words > 0 else 0.0
    id_counts = [len(ID_DENSITY_RE.findall(text)) for text in lowered_texts]
    id_density = _safe_mean([float(count) for count in id_counts])
    opening_words = float(word_counts[0]) if word_counts else 0.0

    uncertainty_rate = _safe_mean([1.0 if UNCERTAINTY_RE.search(text) else 0.0 for text in lowered_texts])
    certainty_rate = _safe_mean([1.0 if CERTAINTY_RE.search(text) else 0.0 for text in lowered_texts])

    pushback_count = 0
    clarify_count = 0
    info_count = 0
    for text in lowered_texts:
        if PUSHBACK_Q_RE.search(text):
            pushback_count += 1
        elif CLARIFY_Q_RE.search(text):
            clarify_count += 1
        elif INFO_Q_RE.search(text):
            info_count += 1
    user_turn_count = len(cleaned_texts)
    pushback_q_rate = pushback_count / user_turn_count if user_turn_count else 0.0
    clarify_q_rate = clarify_count / user_turn_count if user_turn_count else 0.0
    info_q_rate = info_count / user_turn_count if user_turn_count else 0.0

    emotion_rate = _safe_mean([1.0 if EMOTION_PATTERNS.search(text) else 0.0 for text in lowered_texts])
    accusation_rate = _safe_mean([1.0 if ACCUSATION_RE.search(text) else 0.0 for text in lowered_texts])
    pivot_rate = _safe_mean([1.0 if PIVOT_RE.search(text) else 0.0 for text in lowered_texts])

    return {
        "n_turns": float(n_turns),
        "user_words_per_turn": float(user_words_per_turn),
        "politeness_rate": float(politeness_rate),
        "short_msg_rate": float(short_msg_rate),
        "formality_rate": float(formality_rate),
        "ack_only_rate": float(ack_only_rate),
        "verbosity_cv": float(verbosity_cv),
        "repeat_rate": float(repeat_rate),
        "id_confuse_rate": float(id_confuse_rate),
        "info_frontload": float(info_frontload),
        "id_density": float(id_density),
        "opening_words": float(opening_words),
        "uncertainty_rate": float(uncertainty_rate),
        "certainty_rate": float(certainty_rate),
        "pushback_q_rate": float(pushback_q_rate),
        "clarify_q_rate": float(clarify_q_rate),
        "info_q_rate": float(info_q_rate),
        "emotion_rate": float(emotion_rate),
        "accusation_rate": float(accusation_rate),
        "pivot_rate": float(pivot_rate),
    }
