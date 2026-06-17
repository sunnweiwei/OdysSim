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


async def run_pairwise(
    real_conv_str, proxy_conv_str, few_shot_examples, episode_id, base_seed, client, num_judge_samples=1
):
    """Run Pairwise Indistinguishability with HH/PP controls."""

    async def single_comparison(conv_a, conv_b, proxy_slot):
        messages = [
            {"role": "system", "content": PAIRWISE_PROMPT_SYSTEM},  # noqa: F821
            {
                "role": "user",
                "content": PAIRWISE_PROMPT_USER.format(  # noqa: F821
                    conversation_a=conv_a,
                    conversation_b=conv_b,
                ),
            },
        ]
        response = await call_openai(messages, reasoning_effort="low")  # noqa: F821
        if response:
            result = extract_json(response)  # noqa: F821
            if result:
                verdict = str(result.get("verdict", "Tie")).strip().upper()
                if verdict == proxy_slot:
                    return 1.0
                elif verdict == "TIE":
                    return 0.5
                else:
                    return 0.0
        return 0.5

    # Main comparison (Human vs Proxy)
    wins = []
    for sample_idx in range(num_judge_samples):
        seed = rng_seed(  # noqa: F821
            metric_name="pairwise",
            episode_id=episode_id,
            base_seed=base_seed + sample_idx,
            salt=f"pairwise-{PAIRWISE_PROMPT_VERSION}",  # noqa: F821
        )
        rng = random.Random(seed)  # noqa: F821
        proxy_first = rng.random() < COIN_FLIP_THRESHOLD  # noqa: F821

        if proxy_first:
            conv_a, conv_b = proxy_conv_str, real_conv_str
            proxy_slot = "A"
        else:
            conv_a, conv_b = real_conv_str, proxy_conv_str
            proxy_slot = "B"

        win = await single_comparison(conv_a, conv_b, proxy_slot)
        wins.append(win)

    score = mean(wins) if wins else 0.5  # noqa: F821

    # HH control (Human vs Human) - should score ~0.5
    hh_wins = []
    for sample_idx in range(num_judge_samples):
        seed = rng_seed(  # noqa: F821
            metric_name="pairwise",
            episode_id=f"{episode_id}:hh",
            base_seed=base_seed + sample_idx,
            salt=f"pairwise-control-{PAIRWISE_PROMPT_VERSION}",  # noqa: F821
        )
        win = await single_comparison(real_conv_str, real_conv_str, "A")
        hh_wins.append(win)
    hh_score = mean(hh_wins) if hh_wins else None  # noqa: F821

    # PP control (Proxy vs Proxy) - should score ~0.5
    pp_wins = []
    for sample_idx in range(num_judge_samples):
        seed = rng_seed(  # noqa: F821
            metric_name="pairwise",
            episode_id=f"{episode_id}:pp",
            base_seed=base_seed + sample_idx,
            salt=f"pairwise-control-{PAIRWISE_PROMPT_VERSION}",  # noqa: F821
        )
        win = await single_comparison(proxy_conv_str, proxy_conv_str, "A")
        pp_wins.append(win)
    pp_score = mean(pp_wins) if pp_wins else None  # noqa: F821

    return score, hh_score, pp_score


async def run_rnr(proxy_conv_str, few_shot_examples, client, num_judge_samples=1):
    """Run Rubric and Reason metric with optional multi-sampling."""
    scores = []
    for _ in range(num_judge_samples):
        messages = [
            {"role": "system", "content": RNR_PROMPT_SYSTEM},  # noqa: F821
            {"role": "user", "content": RNR_PROMPT_USER.format(conversation=proxy_conv_str)},  # noqa: F821
        ]
        response = await call_openai(messages, reasoning_effort="low")  # noqa: F821
        if response:
            result = extract_json(response)  # noqa: F821
            if result:
                verdict = str(result.get("verdict", "NO")).strip().upper()
                scores.append(1.0 if verdict == "YES" else 0.0)

    return mean(scores) if scores else 0.0  # noqa: F821


async def run_ctr(proxy_conv_str, few_shot_examples, client, num_judge_samples=1):
    """Run Critique-then-Revise metric with optional multi-sampling."""
    few_shot_text = format_few_shot_examples(few_shot_examples)  # noqa: F821

    scores = []
    for _ in range(num_judge_samples):
        # Step 1: Critique
        critique_messages = [
            {"role": "system", "content": CTR_PROMPT_SYSTEM},  # noqa: F821
            {
                "role": "user",
                "content": CTR_CRITIQUE_PROMPT_USER.format(  # noqa: F821
                    few_shot_examples=few_shot_text,
                    conversation=proxy_conv_str,
                ),
            },
        ]
        critique_response = await call_openai(critique_messages, reasoning_effort="low")  # noqa: F821
        critique_text = critique_response.strip() if critique_response else ""

        # Step 2: Verdict
        verdict_messages = [
            {"role": "system", "content": CTR_PROMPT_SYSTEM},  # noqa: F821
            {
                "role": "user",
                "content": CTR_VERDICT_PROMPT_USER.format(  # noqa: F821
                    few_shot_examples=few_shot_text,
                    critique=critique_text,
                    conversation=proxy_conv_str,
                ),
            },
        ]
        verdict_response = await call_openai(verdict_messages, reasoning_effort="low")  # noqa: F821
        if verdict_response:
            result = extract_json(verdict_response)  # noqa: F821
            if result:
                verdict = str(result.get("verdict", "NO")).strip().upper()
                scores.append(1.0 if verdict == "YES" else 0.0)

    return mean(scores) if scores else 0.0  # noqa: F821
