# Ditto

Recipe for **Ditto** — a human simulator trained with reinforcement learning
from **verbal feedback**. See the [paper](http://arxiv.org/abs/2605.20506).

## Data

Two HuggingFace datasets:

| Split | Dataset |
|---|---|
| Train | [`sunweiwei/sim-rl-data`](https://huggingface.co/datasets/sunweiwei/sim-rl-data) |
| Eval  | [`sunweiwei/sim-eval-data`](https://huggingface.co/datasets/sunweiwei/sim-eval-data) |

Download into `$DATA_DIR` (default `data/`):

```bash
huggingface-cli download sunweiwei/sim-rl-data   --repo-type dataset --local-dir data/sim_rl_data
huggingface-cli download sunweiwei/sim-eval-data --repo-type dataset --local-dir data/sim_eval_data
```

Each task has its own train/val parquet — the HF filenames differ per task
(e.g. `sotopia_clean_rl.parquet` + `sotopia_hard_val.parquet`). `run_rl.sh`
contains the full TASK → (train, val) mapping. Supported tasks:

```
sotopia, coser, lifechoices, userllm, mirrorbench, fantom, hitom, paratomi,
mistakes, twinvoice, social_r1, behaviorchain, sim_math, sim_doc,
humanual_{book,chat,email,news,opinion,politics},
alignx, socsci210, humanllm
```

## RL training (per-task)

```bash
# Train Ditto on a single task
bash recipe/ditto/run_rl.sh sotopia

# Or via env var
TASK=coser bash recipe/ditto/run_rl.sh

# Override paths / model / experiment name
ACTOR_MODEL_PATH=Qwen3-8B-Instruct \
EXPERIMENT_NAME=ditto-rl-sotopia \
DATA_DIR=data \
bash recipe/ditto/run_rl.sh sotopia
```

Override with `TRAIN_FILES=` / `VAL_FILES=` to point at custom parquets.

Hyperparameters (LoRA, GRPO clip, batch sizes, etc.) match the top-level
`run_rl.sh`; edit `run_rl.sh` in this directory to change them.

## Evaluation (full 27-task suite)

`eval.sh` runs `val_only` across the full SOUL eval suite. It supports two
modes — evaluate any local checkpoint you trained, or evaluate a remote
API model (Claude, GPT, or any OpenAI-compatible endpoint), or any
open-source model on HF, all against the same task suite.

### HugAgent (evaluation only)

Set `INCLUDE_HUGAGENT=1` to add `hugagent_bsi` (356 items) and
`hugagent_bdu` (1,386 items) to the default 27-task suite. These files are
distributed separately in the gated, CC BY-NC 4.0 dataset
[`social-atoms/hugagent`](https://huggingface.co/datasets/social-atoms/hugagent).
Accept the access terms on that page, log in to Hugging Face, and download:

```bash
huggingface-cli login
huggingface-cli download social-atoms/hugagent \
  odyssim/hugagent_bsi_val.parquet odyssim/hugagent_bdu_val.parquet \
  --repo-type dataset --revision data-v1.1 --local-dir data/hugagent

INCLUDE_HUGAGENT=1 bash recipe/ditto/eval.sh local
# Or, after configuring the API model:
INCLUDE_HUGAGENT=1 bash recipe/ditto/eval.sh api
```

By default the files are read from `$DATA_DIR/hugagent/odyssim`
(`data/hugagent/odyssim` when `DATA_DIR` is unset). To use an existing download,
set `HUGAGENT_DATA_DIR` to the directory containing both parquet files. The
script checks that both exist before starting evaluation. Without
`INCLUDE_HUGAGENT=1`, the original 27-task suite requires no HugAgent download.
Data version `data-v1.1` (2026-10-28) corrects 54 `hugagent_bdu` zoning items
whose gold answer was 0; a download made before that date must be repeated.

The released rows have `split=test` and must not be used for training. Scores
are BSI exact-match accuracy and BDU tolerance accuracy (within 1 on a 5-point
scale or 2 on a 10-point scale), averaged over items in each task. Answers must
be inside `<answer>...</answer>` tags; unparsed responses receive zero. These
task scores are not the paper's composite ATI score.

### Local checkpoints (via vLLM)

```bash
# Default: sunweiwei/Ditto-8B
bash recipe/ditto/eval.sh local

# Your own trained checkpoint
ACTOR_MODEL_PATH=outputs/ditto-rl-sotopia/global_step_200 \
EXPERIMENT_NAME=ditto-rl-sotopia-eval \
bash recipe/ditto/eval.sh local

# Any open-source HF model
ACTOR_MODEL_PATH=Qwen/Qwen3-8B-Instruct      bash recipe/ditto/eval.sh local
ACTOR_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct bash recipe/ditto/eval.sh local
ACTOR_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3 bash recipe/ditto/eval.sh local
```

### API models (via OpenAI-compatible endpoint)

Export `OPENAI_AGENT_MODEL` / `OPENAI_AGENT_BASE_URL` / `OPENAI_AGENT_API_KEY`
at the call site, then run `eval.sh api`. Any provider that exposes an
OpenAI-compatible Chat Completions endpoint works.

```bash
# OpenAI
OPENAI_AGENT_MODEL=gpt-5.4-mini \
OPENAI_AGENT_BASE_URL=https://api.openai.com/v1/ \
OPENAI_AGENT_API_KEY=$OPENAI_API_KEY \
bash recipe/ditto/eval.sh api

# Anthropic (Claude)
OPENAI_AGENT_MODEL=claude-opus-4-7 \
OPENAI_AGENT_BASE_URL=https://api.anthropic.com/v1/ \
OPENAI_AGENT_API_KEY=$ANTHROPIC_API_KEY \
OPENAI_AGENT_REASONING_EFFORT=low \
bash recipe/ditto/eval.sh api

# Google (Gemini, OpenAI-compatible endpoint)
OPENAI_AGENT_MODEL=gemini-3.1-pro-preview \
OPENAI_AGENT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
OPENAI_AGENT_API_KEY=$GEMINI_API_KEY \
bash recipe/ditto/eval.sh api

# Local vLLM / SGLang server (OpenAI-compatible)
OPENAI_AGENT_MODEL=Qwen3-8B-Instruct \
OPENAI_AGENT_BASE_URL=http://localhost:8000/v1/ \
OPENAI_AGENT_API_KEY=EMPTY \
bash recipe/ditto/eval.sh api
```
