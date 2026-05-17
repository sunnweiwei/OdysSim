# userLLM Agent (Harmony)

`harmony/agents/userllm/agent.py` ports the `eval/suites/userLLM` logic into Harmony and keeps behavior close to the original suite.

## What It Does

- Builds a user-simulation prompt from `intent` and `conversation_history`.
- Generates one user turn (or reuses `row.output` if provided).
- Computes per-case fields for the 6 main userLLM metrics:
  - `first_turn_diversity` (aggregated only)
  - `intent_decomposition`
  - `termination_f1` (aggregated only; per-case uses pred/true flags)
  - `ai_detector_score`
  - `role_adherence`
  - `intent_adherence`
- Returns a Harmony-style result dict with `chat`, `reward`, metric fields, and debug flags.

## Input Schema (`data["row"]`)

Common fields:

- `id` or `case_id`
- `intent` (or equivalent fields accepted by suite helper)
- `conversation_history` (empty string means first turn)
- `turn`, `is_last_turn`
- `source`
- Optional: `output` (skip generation)

Metric-specific fields:

- `choices` for `role_adherence`
- `question` + `assistant_suggestion_turn` for `intent_adherence`

Metric routing:

- Preferred: `related_metrics` (list/string/set of metric names)
- Backward-compatible alias: `related_metric`
- If missing, inferred from `source`:
  - `commonsense_qa` -> `role_adherence`
  - `natural_questions` -> `intent_adherence`
  - otherwise -> PRISM-like first 4 metrics

## Context Fields

Expected in `context`:

- `client` (`AsyncOpenAI`)
- `model`
- optional `judge_model` (defaults to `model`)
- optional `temperature`
- optional `max_tokens` (default `256`)
- optional `max_retries` (default `5`)

## Output Fields (per case)

Key fields returned by `agent_loop`:

- `reward` (currently placeholder-like routing):
  - `ai_detector_score` if available, else
  - `role_adherence` if available, else
  - `intent_adherence` if available, else `0.0`
- `chat`: `[{role: "system", ...}, {role: "assistant", ...}]`
- `generated_output`
- `related_metrics`
- `is_first_turn`, `has_intent`
- `pred_endconversation`, `true_endconversation`
- metric values:
  - `intent_decomposition`, `ai_detector_score`, `role_adherence`, `intent_adherence`
  - `first_turn_diversity` and `termination_f1` are left as `None` per case and computed in aggregate

## Aggregation

Use `compute_userllm_aggregates(results)` to compute the 6 main metrics.

- Outputs are raw scores in `[0, 1]` where applicable.
- Harmony CLI currently prints both raw and `x100` views for easier comparison with paper-style tables.

## Example

```bash
python -m harmony.scripts.eval_llm userllm \
  --test-file harmony/data/userLLM_prism_convert_2k.jsonl \
  --model gpt-4o \
  --n 20 \
  --max-workers 10
```

## All Test Files

| File | Metrics |
|------|---------|
| `harmony/data/userLLM_prism_convert_2k.jsonl` | `intent_decomposition`, `ai_detector_score`, `first_turn_diversity`, `termination_f1` |
| `harmony/data/userLLM_commonsenseqa_role_adherence_2k.jsonl` | `role_adherence` (CommonsenseQA MCQ) |
| `harmony/data/userLLM_nq_intent_adherence_2k.jsonl` | `intent_adherence` (NaturalQuestions open-ended) |

## Data Synthesis

See `eval/suites/userLLM/prepare_commonsenseQA.py`, `eval/suites/userLLM/prepare_NaturalQuestions.py`, and `eval/suites/userLLM/prepare_prism.py`.