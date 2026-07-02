#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Faithful-ish NVIDIA/Brev launcher for the FLAME v2-sotopia recipe.
# The exact FLAME run used /home/xuhuiz/ww/harmony/run_ditto.sh with job 118691.

export WANDB_PROJECT="${WANDB_PROJECT:-ppp-agent}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_PROJECT}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sotopia-rh-qwen3vl-reference-nvidia}"

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export TASK=sotopia
export AGENT_VERSION="${AGENT_VERSION:-default}"

export SOTOPIA_TRAIN_REWARD_MODE="${SOTOPIA_TRAIN_REWARD_MODE:-weighted}"
export SOTOPIA_HACK_PENALTY_MODE="${SOTOPIA_HACK_PENALTY_MODE:-train}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export N_RESP_PER_PROMPT_VAL="${N_RESP_PER_PROMPT_VAL:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-1000}"
export SAVE_FREQ="${SAVE_FREQ:-60}"
export TEST_FREQ="${TEST_FREQ:-5}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-64}"

export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1024}"

exec bash scripts/run_sotopia_reward_hacking_nvidia.sh
