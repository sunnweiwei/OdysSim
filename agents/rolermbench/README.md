# RoleRMBench

Harmony adaptation of RoleRMBench as a dual-mode LLM-as-judge benchmark using paper/repo-aligned prompts.

## What It Evaluates

- One shared role-play context
- Two candidate final assistant replies
- Human preference label: `preferred_response` vs `dispreferred_response`

The judge runs both:

- a pairwise binary-choice prompt with swapped-order debiasing
- a per-response rating prompt

Each path yields:

- `1.0` when both directions consistently prefer the human-labeled response
- `0.0` when both directions consistently prefer the dispreferred response
- `0.5` for ties, inconsistent decisions, or parse failures

The row-level reward is the pairwise score, used for progress/checkpoint reporting. The paper-style benchmark score is computed at aggregate time: pairwise and rating are averaged separately, then `accuracy_overall_rolermbench = max(accuracy_overall_pairwise, accuracy_overall_rating)`. `accuracy_overall_row_max` is reported only as a diagnostic for the older, more optimistic per-row max behavior.

The prompt text and conversation-history formatting are aligned with the RoleRMBench paper/repo. One intentional difference remains: parse failures use deterministic `0.5` instead of the repo's random fallback.

## Dataset

After the Hugging Face PR is merged, prepare the parquet data with:

```bash
python prepare_dataset.py --data rolermbench --save_path ~/data/rolermbench/test.parquet
```

The expected source file is:

```text
https://huggingface.co/datasets/Jerry999/user-sim-eval/resolve/main/rolermbench/rolermbench_eval.jsonl
```

## Reference Comparison

Paper reference: Table 2 in RoleRMBench reports `GPT-5-mini-2025-08-07 = 69.30` on the full benchmark.

Quick local sanity check on a 50-row random sample (`seed=42`) from `rolermbench_eval.jsonl` with `gpt-5-mini-2025-08-07`, after switching the main metric from per-row max to aggregate-level best setting:

- `accuracy_overall_pairwise = 72.00`
- `accuracy_overall_rating = 68.00`
- `accuracy_overall_rolermbench = 72.00`
- `accuracy_macro_best_setting = 73.40`

Compare against the paper using `accuracy_overall_rolermbench`, not the row-level average reward.
