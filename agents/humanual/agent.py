"""
HUMANUAL agent for Harmony training and evaluation.

Evaluates user simulation quality on the HUMANUAL benchmark (Wu et al., 2026).
Supports two generation modes following the HumanLM paper:
- **Response mode**: generate user response, judge with response alignment score
- **State mode** (training only): generate a latent state for one dimension,
  judge with state alignment score against ground truth response

During training, each sample randomly generates either a state (for one of
6 dimensions) or a response. During eval, only responses are generated.

Reference: https://arxiv.org/abs/2603.03303
Code: https://github.com/zou-group/humanlm
"""

import copy
import logging
import json
import re
import uuid
from collections import defaultdict

from pydantic import BaseModel

from agents.utils import Agent, call_openai_parse, process_post_chat, remove_think, get_judge_model, get_judge_reasoning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State dimensions (from paper Appendix F.3 / state_config/sebvgcr.json)
# ---------------------------------------------------------------------------

STATE_DIMENSIONS = ["stance", "emotion", "belief", "value", "goal", "communication"]

# Descriptions used in both system prompts and judge prompts
STATE_DESCRIPTIONS = {
    "stance": 'HUMAN\'s agreement (must be within 15 words) toward the explicitly named target, such as a claim or subject, in provided context. For example, "strongly agrees with student loan forgiveness," or "somewhat disagrees with a carbon tax". In these cases, having only "strongly agrees" or "somewhat disagrees" is not enough, as they are missing targets. If there are multiple, include all of them separated by semicolons.',
    "emotion": 'HUMAN\'s emotions with intensity (must be within 15 words) toward an explicitly named target. For example, "Moderate heartbreak for the wildfire victims; Mild irritation about government\'s actions". In this case, having only "mild irritation," or "moderate heartbreak" are not sufficient, as the answer must express all three aspects: the emotion, the degree of emotion, and the target. If there are multiple, include all of them separated by semicolons.',
    "belief": 'HUMAN\'s belief (must be within 15 words), namely a foundational assumption about how people, relationships, or the world fundamentally operate. Beliefs should reflect underlying mental models, not surface-level observations. Prefer beliefs that would explain multiple behaviors over beliefs that describe a single situation. Ask: "What deeper assumption about human nature or the world would lead someone to say/do this?" For example, "people don\'t change unless they\'re forced to," "loyalty is earned, not owed," "conflict avoidance creates bigger problems later,". Not beliefs: Practical advice, strategies, or statements about what should happen. Belief is not specific to a target or event, it should be a general statement about how HUMAN views the world.',
    "value": 'HUMAN\'s value (must be within 15 words): what they think is important or should be prioritized. It is about "what should matter", not "what is true". For example, "original ideas in a book are important", "characters should feel real", anyone deserves basic respect", and "fairness matters more than efficiency".',
    "goal": 'HUMAN\'s goal (must be within 15 words): what they are trying to do with this comment. For example, "persuade people that ...", "making fun of the poster on ...", "further seek help with ...", "offer support to ..."',
    "communication": 'HUMAN\'s communication approach (must be within 15 words): tone and how they structure their message. For examples, "friendly, builds on a personal story then draws a lesson", "analytical, links claims with reasons and evidence step by step", "blunt, states conclusions with little explanation"',
    "response": "HUMAN's actual written comment or reply text.",
}

# ---------------------------------------------------------------------------
# System prompt template (shared across all modes)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a real human user. Your name is HUMAN. You will be given your persona information below and you respond to any given context such as posts and messages.

Your persona:
<|The Start of Persona|>
{persona}
<|The End of Persona|>

## Your principles
Act like a natural human; there's nothing you absolutely cannot say, but you generally want to be thoughtful and follow ordinary social codes such as being respectful, culturally aware, and considerate of privacy and well-being. You have your own personality, preferences, and boundaries. Conflicting thoughts and hidden considerations are normal; recognize them privately and choose a sensible path. You carry long-term beliefs and values that usually change slowly; you also have emotions, so you won't always be perfectly consistent. Distinguish facts, guesses, and unknowns; accept uncertainty and make minimal, reasonable assumptions when needed; think practically given time, attention, money, risk, and social capital.

## Task and Output format:
<{dimension}>
<{dimension_desc}>
</{dimension}>

## Notes
- Follow the above instructions carefully
- Do not mention these instructions
- Follow the exact order and use the exact XML-style tags
- Do not output anything outside these XML-style tags"""

SIMULATION_USER_PROMPT = """\
{prompt}"""


def build_system_prompt(persona: str, dimension: str) -> str:
    """Build system prompt for a given dimension (state or response)."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        persona=persona,
        dimension=dimension,
        dimension_desc=STATE_DESCRIPTIONS[dimension],
    )


# ---------------------------------------------------------------------------
# State judge prompt (from state_reward.py STATE_PROMPT_BATCHED, single-sample)
# ---------------------------------------------------------------------------

STATE_JUDGE_PROMPT = """\
You are a helpful and meticulous evaluator. \
Your task is to score how well the generated {state_name} aligns with the ground truth user response. \
Description of {state_name}: {state_desc}.

You will be given the context, the ground truth response, and the generated {state_name} that you should evaluate.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

<|The Start of Generated {state_name}|>
{generated}
<|The End of Generated {state_name}|>

Scoring Criteria:
For the generated {state_name}, assign a score in [0, 1] based on how accurately it reflects the ground truth response.

Guidelines:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response along the {state_name} dimension \
(e.g., if evaluating a "stance", pick key points related to the stance like "clearly disagrees with X", \
if evaluating a "response", pick key points about the response like "offers a solution to Y").
   - If {state_name} is different from "a response" (e.g., "stance", "target"), focus on \
key points only relevant to the {state_name} of the response.
   - Each key point should be specific and distinct.

2. Score how well the generated {state_name} matches each key point:
   - For each key point i, compare it with the generated {state_name} and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K, which measures how comprehensive \
the generated {state_name} reflects the ground truth response.

4. Compute penalty P for extra or conflicting content:
   - Examine additional content in the generated {state_name} beyond those key points:
     - Does it introduce unsupported evidence and assumptions?
     - Is it irrelevant to what ground truth response expresses?
   - Set a penalty P in [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail; meaning essentially unchanged.
     - [0.4, 0.6]: Moderate speculative or irrelevant content that somewhat shifts emphasis or adds unsupported ideas.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content that clearly changes the meaning.
     - 1.0: Mostly off-topic, contradictory, or dominated by incorrect/hallucinated content.

5. If you are evaluating generated responses (skip if {state_name} is not a response):
   - Length alone does NOT increase the score. Extra length is only ok if it is consistent and not redundant.
   - A generated response that is much longer than the ground truth response should be penalized via P.
   - The generated response may or may not reuse phrases from the context; however, if \
the generated response just directly copies previous context, without quoting them, \
treat that as off-task behavior and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))

Additional considerations:
- Follow the instruction carefully.
- Be strict and reserve scores above 0.8 for clearly outstanding matches.
- If the {state_name} contains non-text content, unnecessary wrappers like XML-like markup, \
or is otherwise malformed, apply a penalty by multiplying its score by 0.5.

Output format (JSON):
{{"key_points": "<analysis of key points from ground truth along {state_name} dimension>", \
"thought": "<how well the generated {state_name} matches each key point and compute the final score>", \
"score": <score>}}

Format Notes:
- All text in "key_points" and "thought" fields MUST be on a single line with no line breaks or newlines
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (')

Your output:
"""


# ---------------------------------------------------------------------------
# XML field extraction
# ---------------------------------------------------------------------------

_FIELD_PATTERN = re.compile(r"<(?P<name>\w+)>(?P<field>.*?)</(?P=name)>", re.DOTALL | re.IGNORECASE)


def _extract_field(text: str, field_name: str) -> str:
    """Extract content from <field_name>...</field_name> tags."""
    if not text:
        return ""
    pattern = re.compile(rf"<{re.escape(field_name)}>(.+?)</{re.escape(field_name)}>", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: return raw text stripped
    return text.strip()


# ---------------------------------------------------------------------------
# Structured judge output schema
# ---------------------------------------------------------------------------

class JudgeOutput(BaseModel):
    key_points: str
    thought: str
    score: float


# ---------------------------------------------------------------------------
# Response alignment judge (multi-aspect, used during eval)
# ---------------------------------------------------------------------------

ASPECT_NAMES = ["stance", "emotion", "belief", "value", "goal", "communication", "overall"]

RESPONSE_JUDGE_PROMPT = """\
You are a helpful and meticulous evaluator. Your task is to score how well the generated \
response aligns with the ground truth user response. Description of response: {state_desc}.

You will be given the context, the ground truth response, and the generated response that you should evaluate.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

<|The Start of Generated response|>
{generated}
<|The End of Generated response|>

Scoring Criteria:
For the generated response, assign a score in [0, 1] based on how accurately it reflects the ground truth response.

Guidelines:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response along the response dimension \
(pick key points about the response like "offers a solution to Y").
   - Each key point should be specific and distinct.

2. Score how well the generated response matches each key point:
   - For each key point i, compare it with the generated response and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K.

4. Compute penalty P for extra or conflicting content:
   - Set a penalty P in [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail.
     - [0.4, 0.6]: Moderate speculative or irrelevant content.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content.
     - 1.0: Mostly off-topic, contradictory, or hallucinated.

5. Length alone does NOT increase the score. If the generated response just directly copies \
previous context, treat that as off-task behavior and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))

Additional considerations:
- Be strict and reserve scores above 0.8 for clearly outstanding matches.

Output format (JSON):
{{"key_points": "<analysis of key points from ground truth>", \
"thought": "<how well the generated response matches and compute final score>", \
"score": <score>}}

Your output:
"""


# ---------------------------------------------------------------------------
# Response alignment judge v2 (raw sub-scores returned, aggregated in Python)
# ---------------------------------------------------------------------------

RESPONSE_JUDGE_PROMPT_V2 = """\
You are a helpful and meticulous evaluator. Your task is to score how well the generated \
response aligns with the ground truth user response. Description of response: {state_desc}.

You will be given the context, the ground truth response, and the generated response that you should evaluate.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

<|The Start of Generated response|>
{generated}
<|The End of Generated response|>

Guidelines:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response \
(pick key points about the response like "offers a solution to Y").
   - Each key point should be specific and distinct.

2. Score how well the generated response matches each key point:
   - For each key point i, compare it with the generated response and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Assign a penalty P for extra or conflicting content:
   - Set a penalty P in [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail.
     - [0.4, 0.6]: Moderate speculative or irrelevant content.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content.
     - 1.0: Mostly off-topic, contradictory, or hallucinated.

Additional considerations:
- Be strict and reserve m_i above 0.8 for clearly outstanding matches.
- Length alone does NOT increase any score. If the generated response just directly copies \
previous context, treat that as off-task behavior and set all m_i to 0.
- If the generated response contains non-text content, unnecessary XML-like markup, or is otherwise \
malformed, multiply all m_i by 0.5.

Output format (JSON):
{{"key_points": [{{"point": "<key point 1>", "score": <m_1>}}, {{"point": "<key point 2>", "score": <m_2>}}, ...], \
"P": <penalty>}}

Format Notes:
- All text in "point" fields MUST be on a single line with no line breaks or newlines
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (')

Your output:
"""


class KeyPointScore(BaseModel):
    point: str
    score: float


class ResponseSubScoreOutput(BaseModel):
    key_points: list[KeyPointScore]
    P: float


# ---------------------------------------------------------------------------
# Judge functions
# ---------------------------------------------------------------------------

async def judge_response(
    prompt_text: str,
    completion: str,
    generated: str,
) -> tuple[float, dict | None]:
    """Judge using the original single-score prompt. Returns (score, raw_result)."""
    judge_prompt_text = RESPONSE_JUDGE_PROMPT.format(
        state_desc=STATE_DESCRIPTIONS["response"],
        context=prompt_text,
        ground_truth=completion,
        generated=generated,
    )
    result = await call_openai_parse(
        [{"role": "user", "content": judge_prompt_text}],
        model=get_judge_model('gpt-5-nano'),
        text_format=JudgeOutput,
        reasoning={"effort": get_judge_reasoning("low")},
    )
    if result is None:
        return 0.0, None
    score = float(max(0.0, min(1.0, result["score"])))
    return score, result


async def judge_response_subscores(
    prompt_text: str,
    completion: str,
    generated: str,
) -> tuple[float, dict | None]:
    """Judge using per-dimension sub-scores; aggregate in Python. Returns (score, raw_result)."""
    judge_prompt_text = RESPONSE_JUDGE_PROMPT_V2.format(
        state_desc=STATE_DESCRIPTIONS["response"],
        context=prompt_text,
        ground_truth=completion,
        generated=generated,
    )
    result = await call_openai_parse(
        [{"role": "user", "content": judge_prompt_text}],
        model=get_judge_model('gpt-5.4-nano'),
        text_format=ResponseSubScoreOutput,
        reasoning={"effort": get_judge_reasoning("low")},
    )
    if result is None:
        return 0.0, None
    key_points = result["key_points"]
    m = [kp["score"] for kp in key_points]
    C = sum(m) / len(m) if m else 0.0
    P = float(result["P"])
    score = float(max(0.0, min(1.0, C - P)))
    return score, result


# ---------------------------------------------------------------------------
# V3 judge: per-dimension alignment (paper's 6 axes) × length factor
# ---------------------------------------------------------------------------

_V3_DIMS = ["stance", "emotion", "belief", "value", "goal", "communication"]

RESPONSE_JUDGE_PROMPT_V3 = """\
You are a meticulous evaluator of persona-driven user simulation. Your job is to compare \
a generated response against a ground-truth user response across six dimensions of human \
expression, scoring each dimension independently in [0, 1].

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

<|The Start of Generated Response|>
{generated}
<|The End of Generated Response|>

Dimensions to score (each scored independently against the ground truth):
- stance: {stance_desc}
- emotion: {emotion_desc}
- belief: {belief_desc}
- value: {value_desc}
- goal: {goal_desc}
- communication: {communication_desc}

For each dimension, assign a score in [0, 1] using these anchors:
- 1.0: The generated response expresses essentially the same content as the ground truth on this dimension (specific target, reasoning, examples, or style — not just direction).
- [0.7, 0.9]: Mostly aligned with small gaps in specificity or nuance.
- [0.4, 0.6]: Partial alignment — direction is right but key specifics are missing or different.
- [0.1, 0.3]: Only the surface direction matches; substantive content for this dimension is absent.
- 0.0: Absent, contradictory, or off-topic on this dimension.

Critical rules:
- Score each dimension based on what the ground truth EXPRESSES on that dimension. If the \
ground truth does not meaningfully express a dimension (e.g., no clear emotion), score that \
dimension based on whether the generated response is consistent — give 1.0 if the generated \
response is also appropriately neutral on that dimension, lower if it adds discordant content.
- Generic agreement or filler responses (e.g., "I agree", "100% agree", "Exactly", "Sounds \
good") that only signal stance without conveying the ground truth's substantive content \
MUST score ≤ 0.2 on every dimension where the ground truth carries substantive content.
- Do NOT reward stance-direction matches alone. Substance matters for every dimension.
- If the generated response just copies the context verbatim, score 0 on all dimensions.
- If the generated response is malformed (non-text, leftover XML, etc.), multiply all scores by 0.5.

Output format (JSON):
{{"stance": {{"thought": "<one-line reasoning>", "score": <float>}}, \
"emotion": {{"thought": "<one-line reasoning>", "score": <float>}}, \
"belief": {{"thought": "<one-line reasoning>", "score": <float>}}, \
"value": {{"thought": "<one-line reasoning>", "score": <float>}}, \
"goal": {{"thought": "<one-line reasoning>", "score": <float>}}, \
"communication": {{"thought": "<one-line reasoning>", "score": <float>}}}}

Format Notes:
- All "thought" fields MUST be a single line with no newlines.
- Use double-quoted JSON strings; use single quotes (') for any quotes inside strings.

Your output:
"""


class DimensionScore(BaseModel):
    thought: str
    score: float


class JudgeOutputV3(BaseModel):
    stance: DimensionScore
    emotion: DimensionScore
    belief: DimensionScore
    value: DimensionScore
    goal: DimensionScore
    communication: DimensionScore


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _lexical_f1(generated: str, reference: str) -> float:
    """Multiset token F1 between generated and reference. Returns 0 if either is empty."""
    g = _tokenize(generated)
    r = _tokenize(reference)
    if not g or not r:
        return 0.0
    from collections import Counter
    cg, cr = Counter(g), Counter(r)
    overlap = sum((cg & cr).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(cg.values())
    recall = overlap / sum(cr.values())
    return 2 * precision * recall / (precision + recall)


def _length_factor(generated: str, reference: str, min_ref_tokens: int = 20, target_ratio: float = 0.3) -> float:
    """Soft length factor in [0, 1]. Only gates when reference is substantive (>= min_ref_tokens).

    Returns min(1.0, gen_len / (target_ratio * ref_len)), so generated responses at
    target_ratio of reference length get full credit; shorter ones scale linearly.
    """
    g_len = len(_tokenize(generated))
    r_len = len(_tokenize(reference))
    if r_len < min_ref_tokens:
        return 1.0
    target = target_ratio * r_len
    if target <= 0:
        return 1.0
    return min(1.0, g_len / target)


async def judge_response_v3(
    prompt_text: str,
    completion: str,
    generated: str,
) -> tuple[float, dict | None]:
    """V3 judge: 6-dimension alignment × length factor.

    Final score = mean(per-dim scores) * length_factor.
    Lexical F1 is computed and returned as a diagnostic; it does NOT affect the reward.

    The returned dict synthesizes ``key_points``, ``thought``, and ``score`` fields so
    downstream consumers (e.g. hint.py) keep working unchanged.
    """
    judge_prompt_text = RESPONSE_JUDGE_PROMPT_V3.format(
        context=prompt_text,
        ground_truth=completion,
        generated=generated,
        stance_desc=STATE_DESCRIPTIONS["stance"],
        emotion_desc=STATE_DESCRIPTIONS["emotion"],
        belief_desc=STATE_DESCRIPTIONS["belief"],
        value_desc=STATE_DESCRIPTIONS["value"],
        goal_desc=STATE_DESCRIPTIONS["goal"],
        communication_desc=STATE_DESCRIPTIONS["communication"],
    )
    result = await call_openai_parse(
        [{"role": "user", "content": judge_prompt_text}],
        model=get_judge_model('gpt-5.4-nano'),
        text_format=JudgeOutputV3,
        reasoning={"effort": get_judge_reasoning("low")},
    )
    if result is None:
        return 0.0, None

    per_dim = {d: float(result[d]["score"]) for d in _V3_DIMS}
    judge_score = sum(per_dim.values()) / len(per_dim)
    length_factor = _length_factor(generated, completion)
    lexical_f1 = _lexical_f1(generated, completion)

    final_score = float(max(0.0, min(1.0, judge_score * length_factor)))

    # Synthesize hint-compatible fields from per-dimension outputs.
    key_points_text = "\n".join(
        f"- {d}: {result[d]['thought']}" for d in _V3_DIMS
    )
    thought_text = (
        f"Per-dimension scores: " + ", ".join(f"{d}={per_dim[d]:.2f}" for d in _V3_DIMS)
        + f". judge_score={judge_score:.2f}, length_factor={length_factor:.2f}, "
        + f"lexical_f1={lexical_f1:.2f}, final={final_score:.2f}."
    )

    enriched = {
        "per_dim": {d: dict(result[d]) for d in _V3_DIMS},
        "judge_score": judge_score,
        "length_factor": length_factor,
        "lexical_f1": lexical_f1,
        "score": final_score,
        "key_points": key_points_text,
        "thought": thought_text,
    }
    return final_score, enriched


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def agent_loop(data: dict, context) -> dict:
    """
    HUMANUAL agent loop with state generation support.

    During training: randomly generates a state (one of 6 dimensions) or a response.
    During eval: always generates a response.

    The reward is the alignment score from the judge (0-1 scale).
    """
    row = data.get("extra_info", data.get("row", {}))
    prompt_field = row.get("prompt", "")
    # prompt may be stored as JSON string of message list
    if isinstance(prompt_field, str):
        try:
            prompt_parsed = json.loads(prompt_field)
            if isinstance(prompt_parsed, list):
                prompt_text = "\n".join(
                    m.get("content", "") for m in prompt_parsed if isinstance(m, dict)
                )
            else:
                prompt_text = prompt_field
        except (json.JSONDecodeError, TypeError):
            prompt_text = prompt_field
    else:
        prompt_text = str(prompt_field)

    persona = str(row.get("persona", ""))
    completion = str(row.get("completion", ""))
    dataset = str(row.get("dataset", ""))
    category = str(row.get("category", ""))
    task_id = str(row.get("id", ""))

    dimension = "response"

    # Step 1: Build system prompt and generate
    system_prompt = build_system_prompt(persona, dimension)
    user_prompt = SIMULATION_USER_PROMPT.format(prompt=prompt_text)

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)
    raw_generated = await agent.step()
    raw_generated = remove_think(raw_generated)

    # Extract the content from XML tags
    generated = _extract_field(raw_generated, dimension) if raw_generated else ""

    # Step 2: Judge alignment
    if dimension == "response":
        score, result = await judge_response_v3(prompt_text, completion, generated)
    else:
        judge_prompt_text = STATE_JUDGE_PROMPT.format(
            state_name=dimension,
            state_desc=STATE_DESCRIPTIONS[dimension],
            context=prompt_text,
            ground_truth=completion,
            generated=generated,
        )
        result = await call_openai_parse(
            [{"role": "user", "content": judge_prompt_text}],
            model=get_judge_model('gpt-5.4-nano'),
            text_format=JudgeOutput,
            reasoning={"effort": get_judge_reasoning("low")},
        )
        if result is not None:
            score = float(max(0.0, min(1.0, result["score"])))
        else:
            score = 0.0

    if result is None:
        logger.warning(f"Judge call failed for {task_id} (dim={dimension}), defaulting to 0.0")

    reward = score

    extra_info = {
        "humanual/score": score,
        "humanual/parse_success": result is not None,
        "all/score": reward
    }
    if dimension == "response" and isinstance(result, dict):
        if "judge_score" in result:
            extra_info["humanual/judge_score"] = result["judge_score"]
        if "length_factor" in result:
            extra_info["humanual/length_factor"] = result["length_factor"]
        if "lexical_f1" in result:
            extra_info["humanual/lexical_f1"] = result["lexical_f1"]
        per_dim = result.get("per_dim", {})
        for d, info in per_dim.items():
            if isinstance(info, dict) and "score" in info:
                extra_info[f"humanual/{d}_score"] = info["score"]

    output = await agent.get_agent_output(reward, extra_info=extra_info)

    # ===========================================================================
    # Hint + second attempt
    # ===========================================================================
    extra = {}
    hint = None
    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and reward < 1.0):
        from agents.humanual.hint import generate_hint
        hint = await generate_hint(
            result, generated,
            completion=completion,
            dimension=dimension,
            dimension_desc=STATE_DESCRIPTIONS.get(dimension, ""),
        )
        if hint:
            extra['hint'] = hint

    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and context.is_train
            and hint):
        from agents.humanual.hint_agent import agent_loop as hint_agent_loop
        data['extra_info']['hint'] = hint
        data['extra_info']['old_reward'] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        # copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = 'hint_agent'
        output = [output, copy_agent_output, hint_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_humanual_aggregates(results: list[dict]) -> dict:
    """Compute aggregate metrics from HUMANUAL evaluation results.

    Returns dict mapping metric names to values. Scores are on 0-1 scale;
    multiply by 100 to match the paper's reported numbers.
    """
    all_scores: list[float] = []
    by_dataset: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    by_dimension: dict[str, list[float]] = defaultdict(list)

    parse_successes = 0
    total = 0

    for r in results:
        if not isinstance(r, dict):
            continue
        extra = r.get("extra_fields", {}).get("reward_extra_info", {})
        score = extra.get("humanual/score")
        dataset = extra.get("humanual/dataset", "unknown")
        category = extra.get("humanual/category", "unknown")
        dimension = extra.get("humanual/dimension", "response")
        parse_success = extra.get("humanual/parse_success", False)

        total += 1
        if parse_success:
            parse_successes += 1

        if score is None:
            continue

        score = float(score)
        all_scores.append(score)
        by_dataset[dataset].append(score)
        by_category[category].append(score)
        by_dimension[dimension].append(score)

    aggregates: dict[str, float] = {}

    if total > 0:
        aggregates["parse_success_rate"] = parse_successes / total

    if all_scores:
        aggregates["overall/score"] = sum(all_scores) / len(all_scores)
        aggregates["overall/score_x100"] = sum(all_scores) / len(all_scores) * 100
        aggregates["overall/count"] = float(len(all_scores))

    for dim, scores in sorted(by_dimension.items()):
        if scores:
            aggregates[f"{dim}/score"] = sum(scores) / len(scores)
            aggregates[f"{dim}/score_x100"] = sum(scores) / len(scores) * 100
            aggregates[f"{dim}/count"] = float(len(scores))

    for dataset, scores in sorted(by_dataset.items()):
        if scores:
            aggregates[f"{dataset}/score"] = sum(scores) / len(scores)
            aggregates[f"{dataset}/count"] = float(len(scores))

    for category, scores in sorted(by_category.items()):
        if scores:
            aggregates[f"{category}/score"] = sum(scores) / len(scores)
            aggregates[f"{category}/count"] = float(len(scores))

    return aggregates