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