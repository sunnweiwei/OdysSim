# Tau-USI

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
