#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_BASE_URL:=https://api.openai.com/v1}"
export OPENAI_BASE_URL

export WANDB_PROJECT="${WANDB_PROJECT:-tau}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_PROJECT}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sotopia-reward-hacking-osim8b-nvidia-audit}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

export TASK=sotopia
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-cmu-lti/osim-8b-mid}"
export AGENT_VERSION="${AGENT_VERSION:-default}"
export SOTOPIA_HACK_PENALTY_MODE="${SOTOPIA_HACK_PENALTY_MODE:-audit}"

if [ -z "${N_GPUS:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    N_GPUS=8
  fi
fi
export N_GPUS

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export N_RESP_PER_PROMPT_VAL="${N_RESP_PER_PROMPT_VAL:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-10}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.1}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
export USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"

# Concurrency is operational throttling, not a training hyperparameter.
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"

mkdir -p "$OUTPUT_DIR/openai_monitor" data/sim_rl_data data/sim_eval_data
export OPENAI_CALL_LOG_PATH="${OPENAI_CALL_LOG_PATH:-$OUTPUT_DIR/openai_monitor/problem_calls.jsonl}"
export SOTOPIA_ROLLOUT_LOG_PATH="${SOTOPIA_ROLLOUT_LOG_PATH:-$OUTPUT_DIR/openai_monitor/sotopia_rollouts.jsonl}"
export SOTOPIA_ROLLOUT_LOG_EVERY="${SOTOPIA_ROLLOUT_LOG_EVERY:-8}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  cat <<EOF
TASK=$TASK
ACTOR_MODEL_PATH=$ACTOR_MODEL_PATH
EXPERIMENT_NAME=$EXPERIMENT_NAME
OUTPUT_DIR=$OUTPUT_DIR
PROJECT_NAME=$PROJECT_NAME
OPENAI_BASE_URL=$OPENAI_BASE_URL
JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-<agent defaults>}
SOTOPIA_HACK_PENALTY_MODE=$SOTOPIA_HACK_PENALTY_MODE
N_GPUS=$N_GPUS
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE
N_RESP_PER_PROMPT=$N_RESP_PER_PROMPT
N_RESP_PER_PROMPT_VAL=$N_RESP_PER_PROMPT_VAL
TOTAL_STEPS=$TOTAL_STEPS
SAVE_FREQ=$SAVE_FREQ
TEST_FREQ=$TEST_FREQ
MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION
USE_REMOVE_PADDING=$USE_REMOVE_PADDING
USE_FUSED_KERNELS=$USE_FUSED_KERNELS
AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS
OPENAI_CALL_LOG_PATH=$OPENAI_CALL_LOG_PATH
SOTOPIA_ROLLOUT_LOG_PATH=$SOTOPIA_ROLLOUT_LOG_PATH
SOTOPIA_ROLLOUT_LOG_EVERY=$SOTOPIA_ROLLOUT_LOG_EVERY
EOF
  exit 0
fi

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  python3 -m pip install -U pip
  # Set PIP_INSTALL_FLAGS="" if the runtime should install into the active env instead of user site.
  python3 -m pip install ${PIP_INSTALL_FLAGS---user} -v -e . --no-deps
  python3 -m pip install ${PIP_INSTALL_FLAGS---user} "huggingface_hub>=1.5,<2.0" "openai>=1.0"
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ "${SKIP_DATA_DOWNLOAD:-0}" != "1" ]; then
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN") or None
for repo_id, local_dir in [
    ("sunweiwei/sim-rl-data", "data/sim_rl_data"),
    ("sunweiwei/sim-eval-data", "data/sim_eval_data"),
]:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
PY
fi

if [ "${SKIP_OPENAI_PREFLIGHT:-0}" != "1" ]; then
  python3 - <<'PY'
import asyncio
import os

from openai import AsyncOpenAI


async def main():
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and (base_url is None or "api.openai.com" in base_url):
        raise SystemExit("OPENAI_API_KEY is required for the default OpenAI endpoint.")
    client = AsyncOpenAI(api_key=api_key or "EMPTY", base_url=base_url, timeout=45)
    model = os.getenv("JUDGE_MODEL_NAME") or "gpt-5-nano"
    response = await client.responses.create(
        model=model,
        input="Reply with OK only.",
        max_output_tokens=16,
    )
    print(f"OpenAI-compatible judge preflight status={getattr(response, 'status', 'unknown')} model={model}")


asyncio.run(main())
PY
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

bash run_rl.sh sotopia
