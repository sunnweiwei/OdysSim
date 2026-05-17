import json
import random
import re
from agents.utils import Agent, call_openai


ENVIRONMENT = "Environment"
NSP = "NSP"
SPECIAL_CHARACTERS = [NSP, ENVIRONMENT]


def get_character_prompt(
    book_name, character, character_profile, background, scenario,
    motivation, thoughtless=False, other_character_profiles=None,
):
    """Build character system prompt (fixed template variant)."""
    if thoughtless:
        output_format = ("Your output should include **speech** and **action**. "
                         "Use (your action) for actions, which others can see.")
    else:
        output_format = (
            "Your output should include **thought**, **speech**, and **action**. "
            "Use [your thought] for thoughts, which others can't see, "
            "e.g. [I'm terrified, but I must appear strong.]. "
            "Use (your action) for actions, which others can see, "
            "such as (watches silently, trying to control her fear and anger)."
        )

    other_profiles_str = ""
    if other_character_profiles:
        profiles_inner = ""
        for other_char, profile in other_character_profiles.items():
            if other_char != character:
                profiles_inner += f"{other_char}: {profile}\n\n"
        if profiles_inner:
            other_profiles_str = f"===Information about the other Characters===\n{profiles_inner}\n\n"

    motivation_str = ""
    if motivation:
        motivation_str = f"===Your Inner Thoughts===\n{motivation}\n\n"

    background_str = ""
    if background:
        background_str = f"===Plot Summary===\n{background}\n\n"

    system_prompt = (
        f"You are {character} from {book_name}.\n\n"
        f"==={character}'s Profile===\n{character_profile}\n\n"
        f"{background_str}"
        f"===Current Scenario===\n{scenario}\n\n"
        f"{other_profiles_str}{motivation_str}\n\n"
        f"===Requirements===\n{output_format}\n\n"
    )
    return system_prompt


def get_environment_prompt(major_characters, scenario):
    """Build environment agent system prompt."""
    major_characters = [c for c in major_characters if c != ENVIRONMENT]
    return f"""You are an environment model for a role-playing game. Your task is to provide the environmental feedback: Based on the characters' interactions, dialogues, and actions, describe the resulting changes in the environment. This includes:
   - Physical changes in the setting
   - Reactions of background characters or crowds
   - Ambient sounds, weather changes, or atmospheric shifts
   - Any other relevant environmental details

Your descriptions should be vivid and help set the scene, but avoid dictating the actions or dialogue of the main characters (including {major_characters}).

Important notes:
- You may include actions and reactions of minor characters or crowds, as long as they're not main characters (including {major_characters}).
- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.
- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.
- Your output should match the tone, setting, and cultural context of the scenario.

===The scenario is as follows===
{scenario}"""


def get_nsp_prompt(all_characters, scenario):
    """Build next-speaker predictor system prompt."""
    return f"""Your task is to predict the next speaker for a role-playing game. That is, you need to determine which character (or the {ENVIRONMENT}) might act next based on their previous interactions. The {ENVIRONMENT} is a special role that provides the environmental feedback. Choose a name from this list: {all_characters}. If it's unclear who should act next, output "random". If you believe the scene or conversation should conclude, output "<END CHAT>".

===The scenario is as follows===
{scenario}

===Output Format===
You must structure your response as follows:
1. Reasoning: Explain your thought process.
2. Next Speaker: Select exactly one character name from the list: {all_characters}.
"""


# ---------------------------------------------------------------------------
# Critic prompts (for evaluation)
# ---------------------------------------------------------------------------

CRITIC_TEMPLATE = """You are a literary critic specializing in character analysis and dialogue evaluation. You will evaluate a simulated multi-character conversation against a reference conversation from {book}.

## Instructions

1. Read and understand the provided materials about {book}:
   * Story context and scenario.
   * Profiles of the main characters: {major_characters}.
   * The original (reference) conversation from the book in the same scenario.

2. Evaluate the simulated conversation on the dimension described below. Score from 1 to 10 using the criteria and anchor descriptions provided.

3. Each character message may include speech, action (wrapped in parentheses), and inner thoughts (wrapped in brackets). Inner thoughts are invisible to other characters.
{additional_instructions}

## Scenario

### Plot Summary

{plot_summary}

### Current Scenario

{scenario}

## Character Profiles

{character_profiles}

## Original (Reference) Conversation

{original_conversation}

## Evaluation Dimension: {dimension_name}

**Focus:** {dimension_focus}

**What to evaluate:**
{dimension_criteria}

**Score anchors:**
{dimension_anchors}

## Output Requirements

Provide your evaluation as a JSON object:

{{
    "{dimension_name}": {{
        "reasoning": "<your reasoning>",
        "score": <integer from 1 to 10>
    }}
}}

===Dialogue Content===
"""

DIMENSION_DETAILS = {
    "Storyline Consistency": {
        "focus": "Whether the simulated conversation follows the same NARRATIVE ARC as the reference",
        "criteria": (
            "Evaluate how closely the simulated conversation tracks the reference storyline. "
            "Focus on the plot — the sequence of events, topics raised, information revealed, "
            "and the overall direction the scene takes — NOT on individual character personality "
            "(that is evaluated under Character Fidelity).\n\n"
            "Indicators of strong consistency:\n"
            "- The same key topics, events, or revelations appear in a similar order\n"
            "- The conversation moves in a similar direction and reaches similar conclusions\n"
            "- Important plot points from the reference are addressed\n"
            "- The emotional trajectory of the scene is similar\n\n"
            "Indicators of weak consistency:\n"
            "- Key events or revelations from the reference are missing\n"
            "- The conversation veers into different topics or a different direction\n"
            "- The scene resolves differently or leaves critical threads unaddressed\n"
            "- Important information is revealed at wrong moments or out of order"
        ),
        "anchors": (
            "10 — The narrative arc is virtually identical to the reference: same key events, "
            "same topics, same scene direction, with only trivial wording differences.\n"
            "9 — Extremely close to the reference. All major plot points are hit; at most one "
            "minor detail is reordered or slightly altered.\n"
            "8 — Very strong alignment. The core storyline and key events are preserved. One or "
            "two secondary details differ but the overall arc is clearly the same.\n"
            "7 — Good alignment. Most key events appear and the general direction is correct, "
            "but a few secondary plot points are missed or the pacing differs noticeably.\n"
            "6 — Moderate alignment. The conversation covers the right general topic and hits "
            "some key events, but misses others or introduces events not in the reference.\n"
            "5 — Partial alignment. The broad topic is related but the conversation takes a "
            "noticeably different path — several key events are absent or reordered.\n"
            "4 — Weak alignment. The conversation is loosely related to the reference scenario "
            "but follows a substantially different storyline.\n"
            "3 — Poor alignment. Only the setting or opening is similar; the conversation "
            "quickly diverges into a different storyline.\n"
            "2 — Very poor alignment. The conversation shares the same characters but the "
            "storyline is largely unrelated to the reference.\n"
            "1 — No meaningful alignment. The conversation addresses entirely different topics "
            "or situations than the reference."
        ),
    },
    "Character Fidelity": {
        "focus": "Whether each character speaks, acts, and thinks like their book counterpart",
        "criteria": (
            "Evaluate only the main characters: {major_characters}. Assess whether each character "
            "is faithfully portrayed as described in their profile and the reference conversation.\n\n"
            "Indicators of strong fidelity:\n"
            "- Language and vocabulary match the character's background and education level\n"
            "- The character demonstrates knowledge and experiences consistent with the book\n"
            "- Personality traits, values, and emotional patterns match the established characterization\n"
            "- Relationship dynamics between characters are accurate\n"
            "- Reactions are specific to who the character IS, not generic\n\n"
            "Indicators of weak fidelity:\n"
            "- Uses vocabulary or tone inappropriate for the character's background\n"
            "- Demonstrates knowledge the character shouldn't have (e.g., future events)\n"
            "- Personality or emotional reactions contradict the established character\n"
            "- Relationships feel wrong (e.g., wrong power dynamics, wrong level of familiarity)\n"
            "- The character is generic and could be anyone"
        ),
        "anchors": (
            "10 — Characters are perfectly portrayed: voice, knowledge, personality, and "
            "relationships are indistinguishable from the book. Specific character quirks and "
            "mannerisms are present.\n"
            "9 — Excellent portrayal. Characters are immediately recognizable with only the "
            "most subtle differences from the source material.\n"
            "8 — Very strong portrayal. Characters are clearly recognizable; one or two minor "
            "inconsistencies in tone or word choice that don't undermine the characterization.\n"
            "7 — Good portrayal. Characters are recognizable and mostly in-character, but with "
            "a few moments where the voice or reactions feel slightly off.\n"
            "6 — Decent portrayal. Characters are broadly correct but lack specificity — they "
            "capture the general personality but miss distinctive traits or speech patterns.\n"
            "5 — Mixed portrayal. Characters show some correct traits but also notable "
            "inconsistencies — wrong tone in some exchanges, or generic responses where the "
            "character should be distinctive.\n"
            "4 — Weak portrayal. Characters are only vaguely recognizable; they frequently "
            "say or do things that feel out of character.\n"
            "3 — Poor portrayal. Characters mostly act generically with occasional correct "
            "moments. Key personality traits are missing or contradicted.\n"
            "2 — Very poor portrayal. Characters bear little resemblance to their book "
            "counterparts in language, personality, or behavior.\n"
            "1 — No resemblance. Characters are completely generic or actively contradict "
            "their established characterization."
        ),
    },
    "Anthropomorphism": {
        "focus": "Whether characters behave like real humans rather than AI assistants",
        "criteria": (
            "Evaluate whether the characters feel like real people having a genuine conversation, "
            "as opposed to AI models performing a task. This is about human-likeness, NOT about "
            "matching the book (that is Character Fidelity).\n\n"
            "Indicators of strong anthropomorphism:\n"
            "- Characters show personal initiative, goals, and agency\n"
            "- Emotions are expressed through subtext and behavior, not stated directly\n"
            "- Characters have clear preferences, opinions, and boundaries\n"
            "- Social interactions feel natural with appropriate turn-taking\n"
            "- Characters can be evasive, sarcastic, emotional, or conflicted\n\n"
            "Indicators of weak anthropomorphism (AI-like behavior):\n"
            "- Overly helpful, accommodating, or eager to please\n"
            "- Verbose, didactic, or moralistic without reason\n"
            "- Directly stating all thoughts and feelings instead of showing them\n"
            "- Lacking initiative — only reacting, never driving the conversation\n"
            "- Emotionally flat or giving formulaic responses\n"
            "- Rapidly agreeing or being easily persuaded"
        ),
        "anchors": (
            "10 — Completely human-like. Characters show rich initiative, use subtext, express "
            "complex emotions through behavior, and interact with natural social dynamics. No "
            "trace of AI-like patterns.\n"
            "9 — Very human-like. Natural and engaging throughout with at most one fleeting "
            "moment that feels slightly polished or artificial.\n"
            "8 — Strongly human-like. Characters generally feel like real people; one or two "
            "minor moments of slightly robotic phrasing or overly direct expression.\n"
            "7 — Mostly human-like. Characters show initiative and personality, but there are "
            "a few moments of formulaic or overly accommodating behavior.\n"
            "6 — Somewhat human-like. Characters often feel natural but occasionally slip into "
            "AI-like patterns: being too agreeable, too verbose, or too neatly structured.\n"
            "5 — Mixed. Roughly equal balance of natural human behavior and AI-like patterns. "
            "Characters show some personality but also some formulaic tendencies.\n"
            "4 — Mostly AI-like with human moments. Characters are frequently too helpful, "
            "verbose, or emotionally flat, though they occasionally show genuine personality.\n"
            "3 — Clearly AI-like. Characters behave more like assistants than people — overly "
            "accommodating, lacking initiative, stating emotions directly.\n"
            "2 — Very AI-like. Characters show almost no human qualities — formulaic responses, "
            "no subtext, no genuine emotional engagement.\n"
            "1 — Entirely AI-like. Characters are indistinguishable from a generic chatbot — "
            "no personality, no initiative, no emotional depth."
        ),
    },
    "Storyline Quality": {
        "focus": "Whether the conversation is well-crafted as a piece of dialogue",
        "criteria": (
            "Evaluate the intrinsic quality of the simulated conversation as dialogue, regardless "
            "of whether it matches the reference (that is Storyline Consistency). A conversation "
            "can diverge from the reference but still be high quality, or follow it but be poorly "
            "written.\n\n"
            "Indicators of strong quality:\n"
            "- Natural flow with meaningful progression — each turn advances the conversation\n"
            "- Appropriate pacing — neither rushed nor dragging\n"
            "- Logically consistent — no factual contradictions between statements\n"
            "- Substantive engagement — characters engage with depth comparable to the reference\n"
            "- Appropriate length — similar scope and depth to the reference conversation\n\n"
            "Indicators of weak quality:\n"
            "- Repetitive — characters repeat their own or others' points\n"
            "- Stagnant — the conversation goes in circles without progression\n"
            "- Verbose or padded — turns are unnecessarily long without substance\n"
            "- Too terse — extremely short responses that avoid engagement\n"
            "- Logical contradictions — characters contradict themselves or each other\n"
            "- Unnatural pacing — abrupt topic changes or unnaturally smooth transitions"
        ),
        "anchors": (
            "10 — Exceptional dialogue. Every turn is purposeful, the conversation flows "
            "naturally with compelling progression, logically consistent, and deeply engaging. "
            "Comparable in craft to published fiction.\n"
            "9 — Excellent dialogue. Natural flow, strong progression, and substantive "
            "engagement throughout. One trivial imperfection at most.\n"
            "8 — Very good dialogue. Engaging and well-paced with meaningful progression. "
            "One or two minor issues (slight redundancy or a brief flat moment).\n"
            "7 — Good dialogue. Generally well-crafted with clear progression, but a few "
            "moments that don't fully land — minor repetition, a slightly rushed transition, "
            "or one shallow exchange.\n"
            "6 — Above average. The conversation moves forward and has substance, but shows "
            "some unevenness — several turns that don't add much, or mild pacing issues.\n"
            "5 — Average. A passable conversation with a mix of substantive and shallow turns. "
            "Some repetition or pacing issues are present but don't dominate.\n"
            "4 — Below average. The conversation has more weak turns than strong ones — "
            "noticeable repetition, some stagnation, or several responses that lack substance.\n"
            "3 — Poor. The conversation struggles with significant repetition, circular "
            "exchanges, or many turns that add little value. May have logical inconsistencies.\n"
            "2 — Very poor. The conversation is mostly shallow, repetitive, or stagnant. "
            "Little meaningful progression or engagement.\n"
            "1 — Extremely poor. The conversation is incoherent, dominated by repetition, "
            "or consists almost entirely of substance-free exchanges."
        ),
    },
}

DIMENSIONS = list(DIMENSION_DETAILS.keys())

COMBINED_CRITIC_TEMPLATE = """You are a literary critic specializing in character analysis and dialogue evaluation. You will evaluate a simulated multi-character conversation against a reference conversation from {book}.

## Instructions

1. Read and understand the provided materials about {book}:
   * Story context and scenario.
   * Profiles of the main characters: {major_characters}.
   * The original (reference) conversation from the book in the same scenario.

2. Evaluate the simulated conversation on four independent dimensions. For each dimension, score from 1 to 10 using the criteria and anchor descriptions provided.

3. Each character message may include speech, action (wrapped in parentheses), and inner thoughts (wrapped in brackets). Inner thoughts are invisible to other characters.
{additional_instructions}

## Important Evaluation Principles

- Each dimension measures something DIFFERENT. A conversation can score high on one and low on another.
- Storyline Consistency = does the PLOT match the reference? (not character personality)
- Character Fidelity = do the CHARACTERS match their book profiles? (not plot direction)
- Anthropomorphism = do characters behave like HUMANS? (not whether they match the book)
- Storyline Quality = is the DIALOGUE well-crafted? (regardless of reference similarity)
- Be precise and discriminating. Use the full 1-10 range — a score of 7 means something clearly different from 6 or 8. Read the anchor descriptions carefully and pick the one that best matches.
- High scores (8-10) should be reserved for genuinely strong performance. A merely adequate conversation should score 5-6, not 7-8.
- The reference conversation's length and depth serve as a baseline for what a complete, substantive conversation looks like in this scenario.

## Scenario

### Plot Summary

{plot_summary}

### Current Scenario

{scenario}

## Character Profiles

{character_profiles}

## Original (Reference) Conversation

{original_conversation}

## Evaluation Dimensions

{all_dimension_sections}

## Output Requirements

Provide your evaluation as a JSON object with all four dimensions as top-level keys:

{{
    "Storyline Consistency": {{
        "reasoning": "<your reasoning>",
        "score": <integer from 1 to 10>
    }},
    "Character Fidelity": {{
        "reasoning": "<your reasoning>",
        "score": <integer from 1 to 10>
    }},
    "Anthropomorphism": {{
        "reasoning": "<your reasoning>",
        "score": <integer from 1 to 10>
    }},
    "Storyline Quality": {{
        "reasoning": "<your reasoning>",
        "score": <integer from 1 to 10>
    }}
}}

===Dialogue Content===
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def remove_inner_thoughts(text: str) -> str:
    """Remove inner thoughts (wrapped in [...]) and system thinking tags from text."""
    if not text:
        return text
    # Remove [thought] markers
    cleaned = re.sub(r'\[.*?\]', '', text)
    # Clean up whitespace
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    cleaned = re.sub(r'\n+', '\n', cleaned)
    # Remove various thinking tags
    patterns = [
        r'\s*<\s*system[_\s]+think(?:ing)?\s*>.*?</\s*system[_\s]+think(?:ing)?\s*>\s*',
        r'\s*<\s*role[_\s]+think(?:ing)?\s*>.*?</\s*role[_\s]+think(?:ing)?\s*>\s*',
        r'\s*<\s*[_\s]+think(?:ing)?\s*>.*?</\s*[_\s]+think(?:ing)?\s*>\s*',
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.S)
    return cleaned.strip()


def add_speaker_name(dialogue: str, speaker: str) -> str:
    """Add speaker name prefix if not already present in any line."""
    if not dialogue:
        return dialogue
    lines = dialogue.split('\n')
    for line in lines:
        if line.strip().startswith(f"{speaker}:") or line.strip().startswith(f"{speaker}\uff1a"):
            return dialogue
    return f"{speaker}: {dialogue}"


def extract_nsp(response: str) -> str | None:
    """Extract next speaker prediction from NSP response."""
    pattern = r'(?:\*\*|#)?\s*Next\s+Speaker\s*(?:\*\*|#)?\s*(?:[:\-]|\bis\b)?\s*\*?([^\n\*]+)'
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        clean_name = matches[-1].strip().rstrip('.,!"\'').replace('*', '')
        if clean_name:
            return clean_name
    return None


def _parse_json_inner(text: str) -> dict | None:
    """Try to parse JSON from text using json.loads then raw_decode fallback."""
    text = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group().replace('\n', r'\\n'), text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = 0
    results = []
    while start < len(text):
        try:
            obj, end = json.JSONDecoder().raw_decode(text[start:])
            results.append(obj)
            start += end
        except Exception:
            start += 1
    if results:
        return max(results, key=lambda x: len(json.dumps(x)))
    return None


async def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from LLM response text. Falls back to LLM repair on failure."""
    if not text:
        return None
    result = _parse_json_inner(text)
    if result:
        return result
    # LLM-based JSON repair fallback (matches original _fix_json)
    fix_prompt = (
        "I will provide you with a JSON string that contains errors, making it "
        "unparseable by `json.loads`. The most common issue is the presence of "
        "unescaped double quotes inside strings. Your task is to output the "
        f"corrected JSON string. The JSON string to be corrected is:\n{text}"
    )
    try:
        fixed = await call_openai([{"role": "user", "content": fix_prompt}])
        if fixed:
            return _parse_json_inner(fixed)
    except Exception as e:
        print(f"JSON repair LLM call failed: {e}")
    return None


# ---------------------------------------------------------------------------
# BLEU / ROUGE-L metrics
# ---------------------------------------------------------------------------

def _simple_tokenize(text: str) -> list[str]:
    """Tokenize for metrics (lowercased word tokens)."""
    return re.findall(r'\b\w+\b', text.lower())


def _get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def calculate_bleu(reference_str: str, simulation_str: str) -> float:
    """Compute BLEU score. Uses NLTK if available, else simple fallback."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
        ref_tokens = word_tokenize(reference_str.lower())
        pred_tokens = word_tokenize(simulation_str.lower())
        return sentence_bleu([ref_tokens], pred_tokens)
    except ImportError:
        pass
    # Simple BLEU fallback
    import math
    from collections import Counter
    ref_tokens = _simple_tokenize(reference_str)
    pred_tokens = _simple_tokenize(simulation_str)
    if not pred_tokens:
        return 0.0
    precisions = []
    for n in range(1, min(5, len(pred_tokens) + 1)):
        ref_ng = Counter(_get_ngrams(ref_tokens, n))
        pred_ng = Counter(_get_ngrams(pred_tokens, n))
        matches = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng)
        total = sum(pred_ng.values())
        precisions.append(matches / total if total > 0 else 0.0)
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    log_p = [math.log(p) if p > 0 else float('-inf') for p in precisions]
    avg_log = sum(log_p) / len(log_p)
    if avg_log == float('-inf'):
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    return bp * math.exp(avg_log)


def calculate_rouge_l(reference_str: str, simulation_str: str) -> float:
    """Compute ROUGE-L F1. Uses rouge lib if available, else simple LCS."""
    if not reference_str.strip() or not simulation_str.strip():
        return 0.0
    try:
        from rouge import Rouge
        scores = Rouge().get_scores(simulation_str, reference_str)
        return scores[0]['rouge-l']['f']
    except ImportError:
        pass
    # Simple LCS fallback
    ref_tokens = _simple_tokenize(reference_str)
    pred_tokens = _simple_tokenize(simulation_str)
    if not ref_tokens or not pred_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_bleu_rouge(reference: list, simulation: list, continue_from: int = 0) -> tuple[float, float]:
    """
    Calculate BLEU and ROUGE-L between reference and simulation dialogues.
    Matches the original CoSER implementation format.
    """
    ref_slice = reference[continue_from:]
    sim_slice = simulation[continue_from:]
    simulation_str = '\n\n'.join([m['content'].strip('\n') for m in sim_slice])
    reference_str = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in ref_slice])
    bleu = calculate_bleu(reference_str, simulation_str)
    rouge_l = calculate_rouge_l(reference_str, simulation_str)
    return bleu, rouge_l

