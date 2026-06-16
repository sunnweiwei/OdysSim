# Tau-USI

## Scoring: USI that matches AgentArena

USI is a **distribution-level** metric, so evaluation has two stages and only the
first is per-task:

1. **Rollout (one task at a time, preserved):** `agent.py::agent_loop` simulates
   the user for a single TauBench task and the harness writes a
   `*_task_results.json` (`{"results": [{instance_id, conversation, survey,
   reward, ...}, ...]}`). The RL reward in `reward.py` (`compute_distributional_
   reward`, an EMA moment-matching proxy) is unrelated to the USI score below.
2. **Scoring (aggregation):** `usi_metric.py` is a faithful, numpy-only port of
   AgentArena's `annotation_analysis/analyze_interaction.compute_all_with_variance`
   — Dice-Sørensen D1–D4 over pooled feature rows vs the 3 human batches,
   difficulty-binned ECE, survey-based Eval, and `USI = mean(D1..D4, [Eval,]
   (1−ECE)·100)`. Feature extraction is the **single shared definition** in
   `utils.extract_conversation_features`. Verified **bit-identical** (max abs
   diff `0.0`) to AgentArena across 33 models × 7 metrics.

ECE difficulty bins are **fixed to AgentArena's 31 published baselines**
(`usi_metric.PUBLISHED_BASELINES`, `difficulty_files=baselines`) so a new model is
scored on the same yardstick as the leaderboard.

### Data setup (gitignored mirror)

The human annotations + baselines live on the AgentArena box (not in git). Mirror
them into `data/tau_usi/` (or set `$TAU_USI_DATA_DIR`):

```bash
rsync -az aws-ec2-usrsim:'AgentArena/annotation_analysis/data/tau_bench_tasks_unified.json' data/tau_usi/
rsync -az aws-ec2-usrsim:'AgentArena/annotation_analysis/data/survey_data/'  data/tau_usi/survey_data/
rsync -az aws-ec2-usrsim:'AgentArena/annotation_analysis/data/eval_results/' data/tau_usi/eval_results/
```

### Run

```bash
# Score a one-by-one eval into the USI table (+ writes <label>_aggregate_metrics.json)
python -m agents.tau_usi.score_eval results/v6_task_results.json --label osim-8b-v6 --print-all
# Optional: derive an Eval (survey-agreement) term from the eval's own surveys
python -m agents.tau_usi.score_eval results/v6_task_results.json --label osim-8b-v6 --emit-survey-comparable
```

Regression tests (synthetic always run; golden test skips without the data
mirror): `pytest agents/tau_usi/tests/test_usi_metric.py`.

## Replication (2026-03-05)

Run setup (concise): `tau_usi`, agent=`gpt-5.2`, domains=`retail,airline`, tasks=`165`, workers=`50`, timeout=`300s`, checkpoint-resume enabled.

Replicated results (this repo runs):

| Model | D1 Conv. | D2 Info. | D3 Clarif. | D4 React. | Eval | ECE | USI |
|------|------|------|------|------|------|------|------|
| gpt-5.2 agent + gpt-5-mini user-sim (reasoning effort=high) | 38.7±6.9 | 76.9±0.7 | 76.1±3.9 | 60.6±3.6 | 73.5±0.5 | 0.267±0.035 | 66.5±1.3 |
| gpt-5.2 agent + gemini-2.0-flash user-sim | 54.6±2.1 | 88.0±1.2 | 73.4±3.5 | 70.1±2.6 | 73.7±0.8 | 0.168±0.013 | 73.8±1.4 |

Artifacts:
- `results/tau_usi/20260305_102037_model_gpt-5.2__usersim_gpt-5-mini_task_results.json`
- `results/tau_usi/20260305_102037_model_gpt-5.2__usersim_gpt-5-mini_aggregate_metrics.json`
- `results/tau_usi/20260305_135903_model_gpt-5.2__usersim_gemini-2.0-flash_task_results.json`
- `results/tau_usi/20260305_135903_model_gpt-5.2__usersim_gemini-2.0-flash_aggregate_metrics.json`

## Reference Numbers

| Model | D1 Conv. | D2 Info. | D3 Clarif. | D4 React. | Eval | ECE | USI |
|------|------|------|------|------|------|------|------|
| Human (inter-ann.) | 87.4±6.8 | 97.9±0.9 | 88.0±1.3 | 93.5±2.5 | 97.4±5.0 | 0.069±0.022 | 92.9±0.9 |
| Gemini-2.0-Flash | 51.6±1.6 | 88.9±1.1 | 68.2±2.1 | 76.9±3.7 | 73.7±0.8 | 0.196±0.020 | 73.3±0.4 |
| GPT-5.1 | 47.3±6.9 | 77.4±0.6 | 73.3±2.0 | 88.1±2.6 | 72.1±1.5 | 0.331±0.030 | 70.9±0.6 |
| GPT-5 | 49.7±5.6 | 73.7±0.7 | 73.2±2.3 | 73.4±3.3 | 74.5±1.1 | 0.210±0.019 | 70.6±1.2 |
| GPT-5-mini | 39.4±5.9 | 74.4±0.7 | 83.1±2.3 | 68.7±1.6 | 73.5±0.5 | 0.174±0.019 | 70.3±0.9 |
