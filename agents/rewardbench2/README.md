# RewardBench 2

Harmony implementation of the RewardBench 2 generative evaluation path.

- non-`Ties` subsets: single-turn 4-way completion ranking
- `Ties`: per-candidate ratings plus the paper's aggregate calibration score

Paper: https://arxiv.org/abs/2506.01937
Official code: https://github.com/allenai/reward-bench
Official dataset: https://huggingface.co/datasets/allenai/reward-bench-2

The local dataset loader is registered as `rewardbench2` in `prepare_dataset.py`.
It expects the processed JSONL hosted at:

```text
https://huggingface.co/datasets/Jerry999/user-sim-eval/resolve/main/rewardbench2/rewardbench2_eval.jsonl
```

Non-`Ties` rows return one agent-loop output. `Ties` rows query each candidate in
a fresh one-turn rating context, matching the official implementation; because
Harmony training outputs are tied to generated tokens, the loop returns one
output per candidate with the same row-level reward metadata.

