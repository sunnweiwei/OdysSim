#!/bin/bash
# Eval-only run across the full SOUL eval suite (27 tasks).
#
# Usage:
#   bash recipe/ditto/eval.sh local   # eval a local checkpoint via vLLM
#   bash recipe/ditto/eval.sh api     # eval an API model (Claude / GPT / ...)

MODE="${1:-${MODE:-local}}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ditto-eval-$MODE}"

data_dir="${DATA_DIR:-data}"
eval_dir="$data_dir/sim_eval_data"
val_files="[\
$eval_dir/sotopia_hard_val.parquet,\
$eval_dir/coser_val.parquet,\
$eval_dir/lifechoices_val.parquet,\
$eval_dir/userllm_val.parquet,\
$eval_dir/mirrorbench_val.parquet,\
$eval_dir/fantom_val.parquet,\
$eval_dir/hitom_val.parquet,\
$eval_dir/paratomi_val.parquet,\
$eval_dir/mistakes_val.parquet,\
$eval_dir/twinvoice_val.parquet,\
$eval_dir/social_r1_val.parquet,\
$eval_dir/behaviorchain_val.parquet,\
$eval_dir/sim_math_val.parquet,\
$eval_dir/sim_doc_val.parquet,\
$eval_dir/humanual_book_val.parquet,\
$eval_dir/humanual_chat_val.parquet,\
$eval_dir/humanual_email_val.parquet,\
$eval_dir/humanual_news_val.parquet,\
$eval_dir/humanual_opinion_val.parquet,\
$eval_dir/humanual_politics_val.parquet,\
$eval_dir/alignx_demo_val.parquet,\
$eval_dir/alignx_pair_val.parquet,\
$eval_dir/alignx_ugc_val.parquet,\
$eval_dir/alignx_arbitrary_val.parquet,\
$eval_dir/alignx_history16_val.parquet,\
$eval_dir/socsci210_val.parquet,\
$eval_dir/humanllm_val.parquet]"

# ── Mode: pick which model to evaluate ────────────────────────────────────────
case "$MODE" in
  local)
    # Evaluate a local checkpoint (HF hub id or local path) via vLLM.
    default_agent_loop="agent_hub"
    actor_model_path="${ACTOR_MODEL_PATH:-sunweiwei/Ditto-8B}"
    ;;
  api)
    # Evaluate a remote API model via the OpenAI-compatible agent loop.
    # Caller is expected to export OPENAI_AGENT_MODEL / BASE_URL / API_KEY
    # (see README for OpenAI / Claude / Gemini / ... examples).
    default_agent_loop="openai_agent"
    # actor_model_path is still required for tokenizer; keep a small local model.
    actor_model_path="${ACTOR_MODEL_PATH:-Qwen3-8B-Instruct}"
    ;;
  *) echo "Unknown MODE: $MODE (expected: local | api)" >&2; exit 1 ;;
esac

# ── Hyperparameters ───────────────────────────────────────────────────────────
max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 8))

n_resp_per_prompt_val=1
infer_tp=1
max_concurrent_rollouts=512

n_gpus=8

# ── Setup ─────────────────────────────────────────────────────────────────────
export HF_HOME=$OUTPUT_DIR/hf_cache
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p $OUTPUT_DIR/hf_cache $OUTPUT_DIR/wandb $OUTPUT_DIR/$EXPERIMENT_NAME

# ── Eval ──────────────────────────────────────────────────────────────────────
export LOGGING_LEVEL=ERROR
NCCL_DEBUG=WARN python3 train_ppo.py \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=$default_agent_loop \
  data.train_files="$val_files" \
  data.val_files="$val_files" \
  data.train_batch_size=64 \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$actor_model_path \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length + 1024)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  +actor_rollout_ref.rollout.agent.max_concurrent_rollouts=$max_concurrent_rollouts \
  algorithm.use_kl_in_reward=False \
  trainer.n_gpus_per_node=$n_gpus \
  trainer.nnodes=1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=ditto \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
  trainer.total_training_steps=1