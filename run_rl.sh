#!/bin/bash
# Ditto RL training — one task at a time.
#
# Data:
#   train: https://huggingface.co/datasets/sunweiwei/sim-rl-data
#   eval : https://huggingface.co/datasets/sunweiwei/sim-eval-data
#
# Usage:
#   bash run_rl.sh sotopia
#   TASK=coser bash run_rl.sh
#
# Each TASK trains independently on its own train/val parquet.

set -e

# ── Task ──────────────────────────────────────────────────────────────────────
TASK="${1:-${TASK:-sotopia}}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ditto-rl-${TASK}}"

# Layout assumed after `huggingface-cli download sunweiwei/sim-rl-data` and
# `sunweiwei/sim-eval-data` into $DATA_DIR.
data_dir="${DATA_DIR:-data}"
rl_dir="$data_dir/sim_rl_data"
val_dir="$data_dir/sim_eval_data"

# TASK → (train file, val file) — matches the HF dataset filenames.
case "$TASK" in
  sotopia)            train_rel=sotopia_clean_rl.parquet;        val_rel=sotopia_hard_val.parquet ;;
  coser)              train_rel=coser_rl_train.parquet;          val_rel=coser_val.parquet ;;
  lifechoices)        train_rel=lifechoices_hard_rl.parquet;     val_rel=lifechoices_val.parquet ;;
  userllm)            train_rel=userllm_rl_train.parquet;        val_rel=userllm_val.parquet ;;
  mirrorbench)        train_rel=mirrorbench_rl_train.parquet;    val_rel=mirrorbench_val.parquet ;;
  fantom)             train_rel=fantom_rl_train.parquet;         val_rel=fantom_val.parquet ;;
  hitom)              train_rel=hitom_rl_train.parquet;          val_rel=hitom_val.parquet ;;
  paratomi)           train_rel=paratomi_rl_train.parquet;       val_rel=paratomi_val.parquet ;;
  mistakes)           train_rel=mistakes_rl_train.parquet;       val_rel=mistakes_val.parquet ;;
  twinvoice)          train_rel=twinvoice_rl_train.parquet;      val_rel=twinvoice_val.parquet ;;
  social_r1)          train_rel=social_r1_rl.parquet;            val_rel=social_r1_val.parquet ;;
  behaviorchain)      train_rel=behaviorchain_rl_train.parquet;  val_rel=behaviorchain_val.parquet ;;
  sim_math)           train_rel=sim_math_rl.parquet;             val_rel=sim_math_val.parquet ;;
  sim_doc)            train_rel=sim_doc_rl.parquet;              val_rel=sim_doc_val.parquet ;;
  humanual_book)      train_rel=humanual_rl_book.parquet;        val_rel=humanual_book_val.parquet ;;
  humanual_chat)      train_rel=humanual_rl_chat.parquet;        val_rel=humanual_chat_val.parquet ;;
  humanual_email)     train_rel=humanual_rl_email.parquet;       val_rel=humanual_email_val.parquet ;;
  humanual_news)      train_rel=humanual_rl_news.parquet;        val_rel=humanual_news_val.parquet ;;
  humanual_opinion)   train_rel=humanual_rl_opinion.parquet;     val_rel=humanual_opinion_val.parquet ;;
  humanual_politics)  train_rel=humanual_rl_politics.parquet;    val_rel=humanual_politics_val.parquet ;;
  alignx)             train_rel=alignx_rl_8k.parquet;            val_rel=alignx_demo_val.parquet ;;
  socsci210)          train_rel=socsci210_rl_2k.parquet;         val_rel=socsci210_val.parquet ;;
  humanllm)           train_rel=humanllm_rl_train.parquet;       val_rel=humanllm_val.parquet ;;
  *) echo "Unknown TASK: $TASK" >&2; exit 1 ;;
esac

train_files="${TRAIN_FILES:-$rl_dir/$train_rel}"
val_files="${VAL_FILES:-$val_dir/$val_rel}"

actor_model_path="${ACTOR_MODEL_PATH:-Qwen3-8B-Instruct}"
project_name="${PROJECT_NAME:-${WANDB_PROJECT:-ditto}}"

# ── Hyperparameters ───────────────────────────────────────────────────────────
default_agent_loop="agent_hub"
# agent_version: "copy" = Ditto (verbal feedback), "default" = vanilla GRPO.
agent_version="${AGENT_VERSION:-default}"

loss_mode="${LOSS_MODE:-vanilla}"
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.28}"

actor_lr="${ACTOR_LR:-5e-6}"
lr_warmup_steps_ratio="${LR_WARMUP_STEPS_RATIO:-0.1}"

max_prompt_length="${MAX_PROMPT_LENGTH:-$((1024 * 8))}"
max_response_length="${MAX_RESPONSE_LENGTH:-$((1024 * 8))}"
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 8))

usp_size="${USP_SIZE:-1}"
train_batch_size="${TRAIN_BATCH_SIZE:-64}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
n_resp_per_prompt_val="${N_RESP_PER_PROMPT_VAL:-1}"
infer_tp="${INFER_TP:-1}"

lora_rank="${LORA_RANK:-32}"
lora_alpha="${LORA_ALPHA:-64}"

n_gpus="${N_GPUS:-8}"
total_steps="${TOTAL_STEPS:-200}"
save_freq="${SAVE_FREQ:-50}"
test_freq="${TEST_FREQ:-5}"
resume_mode="${RESUME_MODE:-auto}"
train_max_samples="${TRAIN_MAX_SAMPLES:--1}"
val_max_samples="${VAL_MAX_SAMPLES:--1}"
val_before_train="${VAL_BEFORE_TRAIN:-True}"
attn_implementation="${ATTN_IMPLEMENTATION:-flash_attention_2}"
use_remove_padding="${USE_REMOVE_PADDING:-True}"
use_fused_kernels="${USE_FUSED_KERNELS:-True}"
use_torch_compile="${USE_TORCH_COMPILE:-True}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-null}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-1024}"

# ── Setup ─────────────────────────────────────────────────────────────────────
export HF_HOME=$OUTPUT_DIR/hf_cache
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p "$OUTPUT_DIR/hf_cache" "$OUTPUT_DIR/wandb" "$OUTPUT_DIR/$EXPERIMENT_NAME"

# ── Train ─────────────────────────────────────────────────────────────────────
export TURNOFF_THINK=1
export LOGGING_LEVEL=ERROR
NCCL_DEBUG=WARN python3 train_ppo.py \
  hydra.run.dir=$OUTPUT_DIR/hydra \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=$default_agent_loop \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-64} \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.train_batch_size=$train_batch_size \
  data.train_max_samples=$train_max_samples \
  data.val_max_samples=$val_max_samples \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$actor_model_path \
  +actor_rollout_ref.model.override_config.attn_implementation=$attn_implementation \
  actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
  actor_rollout_ref.model.lora_rank=$lora_rank \
  actor_rollout_ref.model.lora_alpha=$lora_alpha \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$lr_warmup_steps_ratio \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_torch_compile=$use_torch_compile \
  actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$usp_size \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
  actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length + 1024)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
  actor_rollout_ref.rollout.n=$n_resp_per_prompt \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  algorithm.use_kl_in_reward=False \
  +algorithm.agent_version=$agent_version \
  trainer.n_gpus_per_node=$n_gpus \
  trainer.nnodes=1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=$project_name \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.val_before_train=$val_before_train \
  trainer.save_freq=$save_freq \
  trainer.resume_mode=$resume_mode \
  trainer.max_actor_ckpt_to_keep=10 \
  trainer.max_critic_ckpt_to_keep=10 \
  trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
  trainer.test_freq=$test_freq \
  trainer.total_training_steps=$total_steps \
  trainer.total_epochs=10000
