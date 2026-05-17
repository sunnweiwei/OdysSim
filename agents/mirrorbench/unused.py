async def run_pairwise(real_conv_str, proxy_conv_str, few_shot_examples,
                       episode_id, base_seed, client,
                       num_judge_samples=1):
    """Run Pairwise Indistinguishability with HH/PP controls."""

    async def single_comparison(conv_a, conv_b, proxy_slot):
        messages = [
            {"role": "system", "content": PAIRWISE_PROMPT_SYSTEM},
            {"role": "user", "content": PAIRWISE_PROMPT_USER.format(
                conversation_a=conv_a,
                conversation_b=conv_b,
            )},
        ]
        response = await call_openai(messages, reasoning_effort='low')
        if response:
            result = extract_json(response)
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
        seed = rng_seed(
            metric_name="pairwise", episode_id=episode_id,
            base_seed=base_seed + sample_idx,
            salt=f"pairwise-{PAIRWISE_PROMPT_VERSION}",
        )
        rng = random.Random(seed)
        proxy_first = rng.random() < COIN_FLIP_THRESHOLD

        if proxy_first:
            conv_a, conv_b = proxy_conv_str, real_conv_str
            proxy_slot = "A"
        else:
            conv_a, conv_b = real_conv_str, proxy_conv_str
            proxy_slot = "B"

        win = await single_comparison(conv_a, conv_b, proxy_slot)
        wins.append(win)

    score = mean(wins) if wins else 0.5

    # HH control (Human vs Human) - should score ~0.5
    hh_wins = []
    for sample_idx in range(num_judge_samples):
        seed = rng_seed(
            metric_name="pairwise", episode_id=f"{episode_id}:hh",
            base_seed=base_seed + sample_idx,
            salt=f"pairwise-control-{PAIRWISE_PROMPT_VERSION}",
        )
        win = await single_comparison(real_conv_str, real_conv_str, "A")
        hh_wins.append(win)
    hh_score = mean(hh_wins) if hh_wins else None

    # PP control (Proxy vs Proxy) - should score ~0.5
    pp_wins = []
    for sample_idx in range(num_judge_samples):
        seed = rng_seed(
            metric_name="pairwise", episode_id=f"{episode_id}:pp",
            base_seed=base_seed + sample_idx,
            salt=f"pairwise-control-{PAIRWISE_PROMPT_VERSION}",
        )
        win = await single_comparison(proxy_conv_str, proxy_conv_str, "A")
        pp_wins.append(win)
    pp_score = mean(pp_wins) if pp_wins else None

    return score, hh_score, pp_score


async def run_rnr(proxy_conv_str, few_shot_examples, client,
                  num_judge_samples=1):
    """Run Rubric and Reason metric with optional multi-sampling."""
    scores = []
    for _ in range(num_judge_samples):
        messages = [
            {"role": "system", "content": RNR_PROMPT_SYSTEM},
            {"role": "user", "content": RNR_PROMPT_USER.format(conversation=proxy_conv_str)},
        ]
        response = await call_openai(messages, reasoning_effort='low')
        if response:
            result = extract_json(response)
            if result:
                verdict = str(result.get("verdict", "NO")).strip().upper()
                scores.append(1.0 if verdict == "YES" else 0.0)

    return mean(scores) if scores else 0.0


async def run_ctr(proxy_conv_str, few_shot_examples, client,
                  num_judge_samples=1):
    """Run Critique-then-Revise metric with optional multi-sampling."""
    few_shot_text = format_few_shot_examples(few_shot_examples)

    scores = []
    for _ in range(num_judge_samples):
        # Step 1: Critique
        critique_messages = [
            {"role": "system", "content": CTR_PROMPT_SYSTEM},
            {"role": "user", "content": CTR_CRITIQUE_PROMPT_USER.format(
                few_shot_examples=few_shot_text,
                conversation=proxy_conv_str,
            )},
        ]
        critique_response = await call_openai(critique_messages, reasoning_effort='low')
        critique_text = critique_response.strip() if critique_response else ""

        # Step 2: Verdict
        verdict_messages = [
            {"role": "system", "content": CTR_PROMPT_SYSTEM},
            {"role": "user", "content": CTR_VERDICT_PROMPT_USER.format(
                few_shot_examples=few_shot_text,
                critique=critique_text,
                conversation=proxy_conv_str,
            )},
        ]
        verdict_response = await call_openai(verdict_messages, reasoning_effort='low')
        if verdict_response:
            result = extract_json(verdict_response)
            if result:
                verdict = str(result.get("verdict", "NO")).strip().upper()
                scores.append(1.0 if verdict == "YES" else 0.0)

    return mean(scores) if scores else 0.0
