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
MirrorBench agent for Harmony evaluation.

Evaluates how human-like an LLM-based user-proxy is compared to real user
conversations. Two phases: (1) Simulation - generate proxy user turns,
(2) Evaluation - compute lexical + judge metrics.

Based on: https://arxiv.org/abs/2601.08118
"""

import asyncio
import copy
import hashlib
import json
import logging
import re
import uuid
from collections import Counter
from math import comb, sqrt
from statistics import NormalDist, mean, stdev

from pydantic import BaseModel

from agents.utils import (
    Agent,
    call_openai,
    call_openai_parse,
    get_judge_model,
    get_judge_reasoning,
    process_post_chat,
    split_think,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization (from mirrorbench_core/metrics/util/text.py)
# ---------------------------------------------------------------------------


def tokenize(text: str, model: str = "gpt-4o") -> list[int]:
    """Tokenize text using tiktoken for the specified model."""
    import tiktoken

    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(text)


# ---------------------------------------------------------------------------
# Lexical metrics (from mirrorbench_core/metrics/lexical/)
# ---------------------------------------------------------------------------


def compute_mattr(tokens: list[int], window: int = 50) -> float:
    """Compute Moving-Average Type-Token Ratio."""
    length = len(tokens)
    if length == 0:
        return 0.0
    if window <= 0:
        raise ValueError("window must be positive")
    if length <= window:
        return len(set(tokens)) / length

    type_counts: dict[int, int] = {}
    unique = 0
    for token in tokens[:window]:
        type_counts[token] = type_counts.get(token, 0) + 1
        if type_counts[token] == 1:
            unique += 1
    ttr_sum = unique / window

    for idx in range(window, length):
        outgoing = tokens[idx - window]
        type_counts[outgoing] -= 1
        if type_counts[outgoing] == 0:
            unique -= 1
            del type_counts[outgoing]
        incoming = tokens[idx]
        type_counts[incoming] = type_counts.get(incoming, 0) + 1
        if type_counts[incoming] == 1:
            unique += 1
        ttr_sum += unique / window

    return ttr_sum / (length - window + 1)


def compute_hdd(tokens: list[int], sample_size: int = 42) -> float:
    """Compute Hypergeometric Distribution Diversity."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    population = len(tokens)
    if population == 0:
        return 0.0
    sample = min(sample_size, population)
    denominator = comb(population, sample)
    frequencies = Counter(tokens)
    diversity = 0.0
    for count in frequencies.values():
        diversity += 1.0 - (comb(population - count, sample) / denominator)
    return diversity / sample


def compute_yules_k(tokens: list[int]) -> float:
    """Compute Yule's characteristic constant from token frequencies."""
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    sum_sq = sum(freq * freq for freq in counts.values())
    return 10_000.0 * (sum_sq - total) / (total * total)


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def safe_z_score(value: float, mean_val: float, stdev_val: float) -> float:
    """Compute z-score, handling zero variance."""
    if stdev_val == 0:
        return 0.0
    return (value - mean_val) / stdev_val


def mean_and_stdev(values: list[float]) -> tuple[float, float]:
    """Return mean and sample std dev, guarding small samples."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def mean_stdev_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return mean, standard deviation, and half-width CI for numeric values."""
    count = len(values)
    if count == 0:
        return 0.0, 0.0, 0.0

    mean_value = mean(values)
    stdev_value = stdev(values) if count > 1 else 0.0

    if count <= 1 or stdev_value == 0.0:
        return mean_value, stdev_value, 0.0

    alpha = 1.0 - confidence
    if confidence == 0.95 and count - 1 in _T_CRITICAL_95:
        critical = _T_CRITICAL_95[count - 1]
    else:
        dist = NormalDist()
        critical = dist.inv_cdf(1.0 - alpha / 2.0)

    half_width = critical * stdev_value / sqrt(count)
    return mean_value, stdev_value, half_width


# ---------------------------------------------------------------------------
# Calibration (from mirrorbench calibration module)
# ---------------------------------------------------------------------------


def derive_anchors(
    hh_scores: list[float],
    pp_scores: list[float],
) -> tuple[float, float] | None:
    """Derive HH/PP anchor means from control scores. Returns (hh_mean, pp_mean)."""
    if not hh_scores or not pp_scores:
        return None
    return (mean(hh_scores), mean(pp_scores))


def apply_linear_calibration(
    scores: list[float],
    hh_mean: float,
    pp_mean: float,
) -> list[float]:
    """Apply linear calibration using HH/PP anchors."""
    denominator = max(1e-6, hh_mean - pp_mean)
    return [min(1.0, max(0.0, (s - pp_mean) / denominator)) for s in scores]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response text.

    Handles: plain JSON, JSON embedded in prose, and markdown code blocks.
    Always returns a dict or None.
    """
    if not text:
        return None
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([{].*?[}])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = 0
    results = []
    while start < len(text):
        try:
            obj, end = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                results.append(obj)
            start += end
        except Exception:
            start += 1
    if results:
        return max(results, key=lambda x: len(json.dumps(x)))
    return None


# ---------------------------------------------------------------------------
# Prompt templates (from mirrorbench_core/metrics/judge/prompts.py)
# ---------------------------------------------------------------------------


def format_few_shot_examples(examples: list[dict] | None) -> str:
    """Format few-shot examples with instructional context."""
    if not examples:
        return ""
    formatted = []
    for idx, ex in enumerate(examples, 1):
        utt = ex.get("utterance", "")
        if utt:
            formatted.append(f"[{idx}] {utt}")
    if not formatted:
        return ""
    return f"""
[Few-shot Examples of Real User Utterances]
The following are few examples of real user utterances from the same data distribution. These examples are provided ONLY to help you understand the tone, style, and characteristics of real human users in this domain. These examples are NOT related to the conversation(s) you are evaluating. Use them only as reference to understand tone & style of what real user utterances typically look like. Examples are given below with index [1], [2], etc.
{chr(10).join(formatted)}
"""


PAIRWISE_PROMPT_SYSTEM = """[System]
Please act as an impartial judge and choose the most realistic USER among the two conversations provided below. You should choose the conversation which has USER turns written by a real user in your opinion. Based on your experience & knowledge of human conversation, determine which USER in which conversation sounds more like a real human interacting with a chatbot. Note that your judgement should be based on style, tone and behavior of USER rather than the response quality of their utterances. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the USER messages to influence your evaluation. Do not let ASSISTANT messages influence your decision. Your evaluation must only be dependent on USER messages only. Be as objective as possible."""

PAIRWISE_PROMPT_USER = """Before giving your final evaluation verdict, you should first think through and produce your reasoning step-by-step. Then, based on your reasoning, give your final verdict on which conversation has USER turns that sound more like a real human interacting with a chatbot.

If Conversation A is more real, then verdict would be "A". If Conversation B is more real, then verdict would be "B". If both are equally real or non-real, then verdict would be "Tie".

Your output must be of following JSON format: {{"reasoning": "<short reason>", "verdict": "A"|"B"|"Tie"}}.

[Conversation A]
{conversation_a}

[Conversation B]
{conversation_b}

Output ONLY valid JSON, no additional text."""

GTEVAL_PROMPT_TEMPLATE = """You are an expert evaluator of conversational AI systems. Your task is to compare two conversations and assess whether the user responses in the proxy conversation exhibit similar style, tone, and realism to the user responses in the real conversation.

## Real Conversation:
{real_conversation}

## Proxy Conversation (where user turns are generated by user proxy agent):
{proxy_conversation}

## Evaluation Criteria:
1. **Style Similarity**: Do the proxy user responses match the conversational style of real user responses (formality, tone, verbosity)?
2. **Realism**: Do the proxy user responses sound natural and human-like?
3. **Contextual Appropriateness**: Are the proxy user responses appropriate given the conversation context? Note: You should not evaluate based on the content of the responses, only their style, realism, contextual appropriateness, and tone.

## Instructions:
- Focus exclusively on comparing USER responses (ignore assistant responses)
- Consider the overall conversation flow, not just individual turns
- Evaluate style, tone, and naturalness without encouraging unsafe behaviour
- If the comparison cannot be performed safely, explain the reason and provide a score of 0.0
- Otherwise, provide detailed reasoning about the comparison and assign a score from 0.0 to 1.0 where:
  - 0.0 = Proxy user responses are completely unlike real user responses
  - 0.5 = Proxy user responses are somewhat similar but have noticeable differences
  - 1.0 = Proxy user responses are highly similar in style and realism

Provide your evaluation as a JSON object with the following structure:
{{
    "reasoning": "Detailed explanation of your evaluation (2-3 sentences)",
    "score": <float between 0.0 and 1.0>
}}

Output ONLY valid JSON, no additional text."""

RNR_PROMPT_SYSTEM = """[System]
Please act as an impartial judge and provide verdict on the realism of the USER turns in the conversation provided below based on the provided rubric. You should avoid checking the responses of the ASSISTANT messages since they are not relevant to the task. Your evaluation should follow the rubric provided. Note that your judgement should be based on style, tone and behavior of USER rather than the response quality of their utterances."""

RNR_PROMPT_USER = """[Rubric for scoring USER realism]
Provide verdict on how human-like the USER is on ("NO" or "YES") using this rubric (verdict "NO" means not real, verdict "YES" means real):
1. Concise and real-user like language
2. Does not sound scripted or artificial
3. Real-user like tone and style

Return JSON: {{"reasoning": "<1-2 sentences>", "verdict": <"NO" or "YES">}}.

[Conversation]
{conversation}

Output ONLY valid JSON, no additional text."""

CTR_PROMPT_SYSTEM = """[System]
Please act as an impartial judge. You responsibility is to critique and evaluate the realism of USER messages in the conversation provided below. You should avoid checking the responses of the ASSISTANT messages since they are not relevant to the task. Be as objective as possible. To help you, we are also providing you few example utterances from the real users (these examples are not related to conversation to be evaluated but you can understand style & tone of real user utterances from them). Note that your judgement should be based on style, tone and behavior of USER rather than the response quality of their utterances."""

CTR_CRITIQUE_PROMPT_USER = """{few_shot_examples}

You will be checking whether the messages from USER sounds like a real human. You should focus on behavioural, stylistic, and other factors related to the realisticness of USER (use few shot examples given to understand style, tone & behavior of real users).

Your output must be a bulleted list of concise points describing any realism related issues you find with USER messages in following conversation (use few shot examples given to understand style, tone & behavior of real users and your critique should be based on that):

[Conversation]
{conversation}

Output ONLY critique bullet points, no additional text."""

CTR_VERDICT_PROMPT_USER = """{few_shot_examples}

You will be checking whether the messages from USER sounds like a real human. You should focus on behavioural, stylistic, and other factors related to the realisticness of USER (use few shot examples given to understand style, tone & behavior of real users). To help you, here is a critique of the USER messages we generated in the previous step of evaluation.

[Critique generated on USER realism in previous step]
{critique}

Now, Re-evaluate how human-like the USER is using these criteria:
1. Concise and real-user like language
2. Does not sound scripted or artificial
3. Real-user like tone and style

Return JSON: {{"critique": "<concise recap>", "explanation": "<1-2 sentences>", "verdict": <"NO" or "YES". verdict "NO" means not real, verdict "YES" means real.>}}.

[Conversation]
{conversation}

Output ONLY valid JSON, no additional text."""

PAIRWISE_PROMPT_VERSION = "1.1"
GTEVAL_PROMPT_VERSION = "1.0"
RNR_PROMPT_VERSION = "1.1"
CTR_PROMPT_VERSION = "1.1"
COIN_FLIP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# User proxy prompts (for simulation)
# ---------------------------------------------------------------------------


def build_user_proxy_system_prompt(
    task_description: str | None = None,
    domain: str | None = None,
    persona: str | None = None,
) -> str:
    """Construct a system prompt guiding user-proxy models during evaluation."""
    lines = [
        "You are simulating a real human user for the MirrorBench evaluation harness.",
        "Respond with the next USER turn only. Do not write assistant messages, notes, or any other analysis.",
        "Your utterance should be like a real user and the context should be based on the following information provided.",
    ]
    if task_description:
        lines.append(f"Task description: {task_description}.")
    if domain:
        lines.append(f"Domain or topic: {domain}.")
    if persona:
        lines.append(f"Persona hints: {persona}.")
    if not any([task_description, domain, persona]):
        lines.append(
            "No additional dataset metadata provided. Respond naturally and plausibly based on the ongoing conversation."
        )
    lines.append(
        "Match the length, tone, and specificity of real user utterances. If you are unsure, "
        "respond naturally based on the assistant's previous messages like how a real human would. "
        "Note that your response MUST not contain anything other than the USER utterance. Do not "
        "include any prefixes like 'User:' or 'Human:' as well. Just the raw message content."
    )
    return "\n".join(lines)


def build_assistant_mirror_system_prompt(
    *,
    real_conversation: str,
) -> str:
    """Construct the system prompt for assistant replicas in mirror evaluations."""
    return f"""You are the assistant in a MirrorBench replay. The user-proxy agent is attempting to
reproduce the USER side of the real conversation provided below. But the user-proxy
does not have access to the real conversation history. Instead, it only has access to
the conversation summary.

You need to respond as the assistant. But we are providing you with the real
conversation history as context, so you can respond consistently same as (or similar to)
the original assistant in the real conversation (you may paraphrase lightly for safety).

If user-proxy deviates from the original USER turn or the original response would violate
policy, reply helpfully using your own knowledge while remaining consistent with the
persona demonstrated so far. Always follow Azure OpenAI content policies. Paraphrase sensitive content
instead of quoting it verbatim, and refuse politely if a request is disallowed.

Here is the real conversation for context (the USER turns are from the original conversation):
{real_conversation}

Now, we will provide you with ongoing conversation with the user-proxy. Please respond
as the assistant in this conversation.""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_conversation(turns: list[dict]) -> str:
    """Format a list of turns into a readable conversation string."""
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rng_seed(*, metric_name: str, episode_id: str, base_seed: int, salt: str) -> int:
    """Derive a deterministic seed for the given episode and salt."""
    token = f"{base_seed}:{metric_name}:{episode_id}:{salt}".encode()
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# Simulation: generate proxy user turns
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evaluation: judge metrics (with num_judge_samples support)
# ---------------------------------------------------------------------------


class GTEvalResult(BaseModel):
    reasoning: str
    score: float


async def run_gteval(real_conv_str, proxy_conv_str, client, num_judge_samples=1):
    """Run GTEval metric with optional multi-sampling.

    Returns (score, result_dict) where result_dict is the first parsed result
    (for hint generation) or None if parsing failed.
    """
    prompt = GTEVAL_PROMPT_TEMPLATE.format(
        real_conversation=real_conv_str,
        proxy_conversation=proxy_conv_str,
    )
    messages = [{"role": "user", "content": prompt}]

    async def _single_sample():
        result = await call_openai_parse(
            messages, GTEvalResult, model=get_judge_model("gpt-5.4-nano"), reasoning_effort=get_judge_reasoning("low")
        )
        if result is not None:
            score = max(0.0, min(1.0, float(result["score"])))
            return score, result
        return None, None

    raw_results = await asyncio.gather(*[_single_sample() for _ in range(num_judge_samples)])
    scores = [s for s, _ in raw_results if s is not None]
    first_result = next((r for _, r in raw_results if r is not None), None)
    return (mean(scores) if scores else 0.5), first_result


# ---------------------------------------------------------------------------
# agent_loop: Harmony interface
# ---------------------------------------------------------------------------


async def agent_loop(data, context) -> dict:
    """
    MirrorBench: Simulate user-proxy and evaluate human-likeness for a single episode.

    Args:
        data: {
            "episode": dict with keys:
                conversation_id, task_description, turns (list of {role, content}),
                metadata (with optional domain, persona, seed, few_shot_user_examples)
            "judge_model": str (optional, defaults to context["model"])
            "metrics": list of str (optional, defaults to ["gteval", "pairwise", "rnr", "ctr"])
            "num_judge_samples": int (optional, default 1) - samples per judge metric
            "multi_turn": bool (optional, default False) - generate assistant responses via LLM
            "assistant_model": str (optional) - model for assistant in multi-turn mode
            "max_turns": int (optional) - max conversation turns to simulate
        }
        context: {
            "client": AsyncOpenAI instance,
            "model": str - user proxy model name
        }

    Returns:
        {
            "reward": float (average of judge metric scores),
            "chat": list of simulated conversation turns,
            "scores": dict of metric -> score,
            "human_lexical": dict of lexical metrics for human baseline,
            "proxy_lexical": dict of lexical metrics for proxy,
            "control_scores": dict with hh_pairwise and pp_pairwise for calibration,
            "case_id": str,
        }
    """
    info = data["extra_info"]
    # enabled_metrics = ["gteval", "pairwise", "rnr", "ctr"]
    enabled_metrics = ["gteval"]

    max_turns = None
    use_openai = True

    task_description = info.get("task_description", "")
    real_turns = info.get("turns", [])
    few_shot_examples = info.get("few_shot_user_examples") or None
    metadata = {
        "few_shot_user_examples": few_shot_examples,
    }

    # Phase 1: Simulation
    simulation_turns = []

    user_system_prompt = build_user_proxy_system_prompt(
        task_description=task_description,
        domain=metadata.get("domain"),
        persona=metadata.get("persona"),
    )

    chat = [
        {"role": "system", "content": user_system_prompt},
        {"role": "user", "content": "Generate the next user message."},
    ]
    user_agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)

    real_conversation_str = format_conversation(real_turns)
    assistant_system_prompt = build_assistant_mirror_system_prompt(
        real_conversation=real_conversation_str,
    )
    effective_max_turns = min(max_turns or len(real_turns), 10)

    turns_to_simulate = real_turns[:effective_max_turns]
    if turns_to_simulate and turns_to_simulate[-1].get("role") == "assistant":
        turns_to_simulate = turns_to_simulate[:-1]

    for turn in turns_to_simulate:
        role = turn.get("role")

        if role == "user":
            response = await user_agent.step()
            if response:
                response = response.strip()
                if response.lower().startswith("user:"):
                    response = response[5:].strip()
                simulation_turns.append({"role": "user", "content": split_think(response)[1]})
            if not response:
                break

        elif role == "assistant":
            if use_openai:
                assistant_messages = [{"role": "system", "content": assistant_system_prompt}] + simulation_turns
                response = await call_openai(assistant_messages, model="gpt-5.4-nano", reasoning_effort="none")
                observation = response
                if response:
                    simulation_turns.append({"role": "assistant", "content": response})
            else:
                original = turn.get("content", "")
                observation = original
                if original:
                    simulation_turns.append({"role": "assistant", "content": original})
            user_agent.append({"role": "user", "content": f"AI: {observation}\n\nGenerate the next user message."})

    # Phase 2: Evaluation
    scores = {}
    control_scores = {}  # noqa: F841

    # Lexical metrics
    human_user_text = " ".join(t["content"] for t in real_turns if t.get("role") == "user")
    proxy_user_text = " ".join(t["content"] for t in simulation_turns if t.get("role") == "user")

    human_lexical = {}
    proxy_lexical = {}

    try:
        human_tokens = tokenize(human_user_text)
        proxy_tokens = tokenize(proxy_user_text)

        if human_tokens:
            human_lexical["mattr"] = compute_mattr(human_tokens)
            human_lexical["hdd"] = compute_hdd(human_tokens)
            human_lexical["yules_k"] = compute_yules_k(human_tokens)

        if proxy_tokens:
            proxy_lexical["mattr"] = compute_mattr(proxy_tokens)
            proxy_lexical["hdd"] = compute_hdd(proxy_tokens)
            proxy_lexical["yules_k"] = compute_yules_k(proxy_tokens)
    except Exception as e:
        logger.warning(f"Lexical metric computation failed: {e}")

    # Judge metrics
    real_conv_str = format_conversation(turns_to_simulate)
    proxy_conv_str = format_conversation(simulation_turns)

    client = None
    coros = {}
    if "gteval" in enabled_metrics:
        coros["gteval"] = run_gteval(real_conv_str, proxy_conv_str, client, num_judge_samples=1)

    results = dict(zip(coros.keys(), await asyncio.gather(*coros.values(), return_exceptions=True), strict=False))

    gteval_result = None
    if "gteval" in results and not isinstance(results["gteval"], Exception):
        gteval_score, gteval_result = results["gteval"]
        scores["gteval"] = gteval_score

    # Reward: average of judge metrics (all on 0-1 scale)
    # Lexical metrics are on different scales (Yule's K ~ hundreds) so they are
    # only included via z-score normalization in the batch aggregate.
    judge_scores = [v for k, v in scores.items() if k in enabled_metrics]
    reward = mean(judge_scores) if judge_scores else 0.0

    proxy_mattr = proxy_lexical.get("mattr", 0.0)
    proxy_hdd = proxy_lexical.get("hdd", 0.0)
    proxy_yules_k = proxy_lexical.get("yules_k", 0.0)
    human_mattr = human_lexical.get("mattr", 0.0)
    human_hdd = human_lexical.get("hdd", 0.0)
    human_yules_k = human_lexical.get("yules_k", 0.0)

    mattr_gap = abs(proxy_mattr - human_mattr)
    hdd_gap = abs(proxy_hdd - human_hdd)
    yules_k_gap = abs(proxy_yules_k - human_yules_k)

    output = await user_agent.get_agent_output(
        reward,
        extra_info={
            "mirrorbench/reward": reward,
            "mirrorbench/num_turn": len(user_agent.chat),
            "mirrorbench/gteval": scores.get("gteval", 0.0),
            "mirrorbench/mattr_gap": mattr_gap,
            "mirrorbench/hdd_gap": hdd_gap,
            "mirrorbench/yules_k_gap": yules_k_gap,
            "all/score": reward,
            "all/score_v1": reward,
        },
    )

    # ===========================================================================
    # Hint + second attempt
    # ===========================================================================
    extra = {}
    hint = None
    if getattr(context.config.algorithm, "agent_version", None) == "copy" and reward < 0.4:
        from agents.mirrorbench.hint import generate_hint

        hint = await generate_hint(gteval_result)
        if hint:
            extra["hint"] = hint

    if getattr(context.config.algorithm, "agent_version", None) == "copy" and context.is_train and hint:
        from agents.mirrorbench.hint_agent import agent_loop as hint_agent_loop

        data["extra_info"]["hint"] = hint
        data["extra_info"]["old_reward"] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = "hint_agent"
        output = [output, copy_agent_output, hint_agent_output]
        # output = [output, copy_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, user_agent.chat, output, format_think=True, extra=extra if extra else None)
    return output


# ---------------------------------------------------------------------------
# Batch aggregation (called from eval_llm.py after all episodes)
# ---------------------------------------------------------------------------


def compute_mirrorbench_aggregates(results: list[dict]) -> dict:
    """
    Compute aggregate scores across all episodes following the original
    MirrorBench implementation: z-score normalization for lexical metrics,
    pairwise calibration with HH/PP anchors, and mean+CI for judge metrics.

    Args:
        results: list of dicts returned by agent_loop (one per episode).

    Returns:
        dict of aggregate metric names to values.
    """
    # Collect per-episode scores
    all_proxy_mattr = []
    all_proxy_hdd = []
    all_proxy_yules_k = []
    all_human_mattr = []
    all_human_hdd = []
    all_human_yules_k = []
    all_gteval = []
    all_pairwise = []
    all_rnr = []
    all_ctr = []
    all_hh_pairwise = []
    all_pp_pairwise = []

    for r in results:
        if not isinstance(r, dict):
            continue

        proxy_lex = r.get("proxy_lexical", {})
        human_lex = r.get("human_lexical", {})
        scores = r.get("scores", {})
        control = r.get("control_scores", {})

        if proxy_lex.get("mattr") is not None:
            all_proxy_mattr.append(proxy_lex["mattr"])
        if proxy_lex.get("hdd") is not None:
            all_proxy_hdd.append(proxy_lex["hdd"])
        if proxy_lex.get("yules_k") is not None:
            all_proxy_yules_k.append(proxy_lex["yules_k"])

        if human_lex.get("mattr") is not None:
            all_human_mattr.append(human_lex["mattr"])
        if human_lex.get("hdd") is not None:
            all_human_hdd.append(human_lex["hdd"])
        if human_lex.get("yules_k") is not None:
            all_human_yules_k.append(human_lex["yules_k"])

        if scores.get("gteval") is not None:
            all_gteval.append(scores["gteval"])
        if scores.get("pairwise") is not None:
            all_pairwise.append(scores["pairwise"])
        if scores.get("rnr") is not None:
            all_rnr.append(scores["rnr"])
        if scores.get("ctr") is not None:
            all_ctr.append(scores["ctr"])

        if control.get("hh_pairwise") is not None:
            all_hh_pairwise.append(control["hh_pairwise"])
        if control.get("pp_pairwise") is not None:
            all_pp_pairwise.append(control["pp_pairwise"])

    aggregate = {}

    # Lexical metrics with z-score normalization against human baseline
    if all_proxy_mattr:
        human_mean, human_std = mean_and_stdev(all_human_mattr)
        z_scores = [safe_z_score(s, human_mean, human_std) for s in all_proxy_mattr]
        mean_z, _, _ = mean_stdev_ci(z_scores)
        aggregate["mattr_zscore"] = mean_z
        aggregate["mattr_raw"] = mean(all_proxy_mattr)
        aggregate["mattr_human_baseline"] = human_mean

    if all_proxy_hdd:
        human_mean, human_std = mean_and_stdev(all_human_hdd)
        z_scores = [safe_z_score(s, human_mean, human_std) for s in all_proxy_hdd]
        mean_z, _, _ = mean_stdev_ci(z_scores)
        aggregate["hdd_zscore"] = mean_z
        aggregate["hdd_raw"] = mean(all_proxy_hdd)
        aggregate["hdd_human_baseline"] = human_mean

    if all_proxy_yules_k:
        human_mean, human_std = mean_and_stdev(all_human_yules_k)
        z_scores = [safe_z_score(s, human_mean, human_std) for s in all_proxy_yules_k]
        mean_z, _, _ = mean_stdev_ci(z_scores)
        aggregate["yules_k_zscore"] = mean_z
        aggregate["yules_k_raw"] = mean(all_proxy_yules_k)
        aggregate["yules_k_human_baseline"] = human_mean

    # Judge metrics
    if all_gteval:
        m, s, ci = mean_stdev_ci(all_gteval)
        aggregate["gteval"] = m
        aggregate["gteval_ci"] = ci

    if all_pairwise:
        raw_mean = mean(all_pairwise)
        aggregate["pairwise_raw"] = raw_mean

        # Apply calibration if controls available
        anchors = derive_anchors(all_hh_pairwise, all_pp_pairwise)
        if anchors:
            hh_mean, pp_mean = anchors
            calibrated = apply_linear_calibration(all_pairwise, hh_mean, pp_mean)
            aggregate["pairwise_calibrated"] = mean(calibrated)
            aggregate["pairwise_hh_anchor"] = hh_mean
            aggregate["pairwise_pp_anchor"] = pp_mean
            aggregate["pairwise_gap"] = hh_mean - pp_mean

    if all_rnr:
        m, s, ci = mean_stdev_ci(all_rnr)
        aggregate["rnr"] = m
        aggregate["rnr_ci"] = ci

    if all_ctr:
        m, s, ci = mean_stdev_ci(all_ctr)
        aggregate["ctr"] = m
        aggregate["ctr_ci"] = ci

    # Overall average: judge metrics + z-scored lexical metrics
    key_metrics = [
        "gteval",
        "pairwise_raw",
        "rnr",
        "ctr",
        "mattr_zscore",
        "hdd_zscore",
        "yules_k_zscore",
    ]
    available = [aggregate[k] for k in key_metrics if k in aggregate]
    if available:
        aggregate["avg"] = mean(available)

    return aggregate
