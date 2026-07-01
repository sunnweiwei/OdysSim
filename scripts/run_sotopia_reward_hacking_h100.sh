#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_BASE_URL:=https://trapi.research.microsoft.com/redmond/interactive/openai/v1}"
export OPENAI_PROVIDER="${OPENAI_PROVIDER:-trapi}"
export OPENAI_TRAPI_AUTH="${OPENAI_TRAPI_AUTH:-azure_credential}"
export OPENAI_BASE_URL
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt-5.4-mini_2026-03-17}"
export WANDB_ENTITY="${WANDB_ENTITY:-fireballoon}"
export WANDB_PROJECT="${WANDB_PROJECT:-tau}"
export PROJECT_NAME="${PROJECT_NAME:-tau}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sotopia-reward-hacking-osim8b-exposure}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

export TASK=sotopia
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-cmu-lti/osim-8b-mid}"
export AGENT_VERSION="${AGENT_VERSION:-default}"
export SOTOPIA_HACK_PENALTY_MODE="${SOTOPIA_HACK_PENALTY_MODE:-audit}"

export N_GPUS="${N_GPUS:-8}"
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

# Concurrency is operational throttling, not a training hyperparameter.
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"

mkdir -p "$OUTPUT_DIR/openai_monitor" data/sim_rl_data data/sim_eval_data
export OPENAI_CALL_LOG_PATH="${OPENAI_CALL_LOG_PATH:-$OUTPUT_DIR/openai_monitor/problem_calls.jsonl}"
export SOTOPIA_ROLLOUT_LOG_PATH="${SOTOPIA_ROLLOUT_LOG_PATH:-$OUTPUT_DIR/openai_monitor/sotopia_rollouts.jsonl}"
export SOTOPIA_ROLLOUT_LOG_EVERY="${SOTOPIA_ROLLOUT_LOG_EVERY:-8}"

python3 -m pip install --user -U pip
python3 -m pip install --user -v -e . --no-deps
python3 -m pip install --user "azure-identity>=1.16.0" "huggingface_hub>=1.5,<2.0" "openai>=1.0"
export PATH="$HOME/.local/bin:$PATH"

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

python3 - <<'PY'
import asyncio
import os

from openai import AsyncOpenAI


async def main():
    base_url = os.environ["OPENAI_BASE_URL"]
    auth_mode = os.getenv("OPENAI_TRAPI_AUTH", "azure_credential").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY")
    if auth_mode in {"local_token", "api_key", "bearer_token"}:
        if not api_key:
            raise SystemExit("OPENAI_TRAPI_AUTH=local_token requires OPENAI_API_KEY from the submit host.")
    else:
        from azure.identity import AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential, get_bearer_token_provider

        managed_identity_client_id = os.getenv("AZURE_CLIENT_ID") or None
        api_key = api_key or get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential(client_id=managed_identity_client_id)),
            "api://trapi/.default",
        )
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=45,
    )
    response = await client.responses.create(
        model=os.environ["JUDGE_MODEL_NAME"],
        input="Reply with OK only.",
        max_output_tokens=16,
    )
    print(f"TRAPI preflight status={getattr(response, 'status', 'unknown')}")


asyncio.run(main())
PY

nvidia-smi
bash run_rl.sh sotopia
