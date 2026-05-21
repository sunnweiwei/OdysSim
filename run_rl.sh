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

# ── Hyperparameters ───────────────────────────────────────────────────────────
default_agent_loop="agent_hub"
# agent_version: "copy" = Ditto (verbal feedback), "default" = vanilla GRPO.
agent_version="default"

loss_mode="vanilla"
clip_ratio_low=0.2
clip_ratio_high=0.28

actor_lr=5e-6

max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 8))
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 8))

usp_size=1
train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=8
n_resp_per_prompt_val=1
infer_tp=1

lora_rank=32
lora_alpha=64

n_gpus=8
total_steps=200
save_freq=50
resume_mode=auto

# ── Setup ─────────────────────────────────────────────────────────────────────
export HF_HOME=$OUTPUT_DIR/hf_cache
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p $OUTPUT_DIR/hf_cache $OUTPUT_DIR/wandb $OUTPUT_DIR/$EXPERIMENT_NAME

# ── Train ─────────────────────────────────────────────────────────────────────
export TURNOFF_THINK=1
export LOGGING_LEVEL=ERROR
NCCL_DEBUG=WARN python3 train_ppo.py \
  hydra.run.dir=$OUTPUT_DIR/hydra \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=$default_agent_loop \
  actor_rollout_ref.rollout.agent.num_workers=64 \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.train_batch_size=$train_batch_size \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$actor_model_path \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.lora_rank=$lora_rank \
  actor_rollout_ref.model.lora_alpha=$lora_alpha \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$usp_size \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length + 1024)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.n=$n_resp_per_prompt \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  algorithm.use_kl_in_reward=False \
  +algorithm.agent_version=$agent_version \
  trainer.n_gpus_per_node=$n_gpus \
  trainer.nnodes=1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=ditto \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.val_before_train=True \
  trainer.save_freq=$save_freq \
  trainer.resume_mode=$resume_mode \
  trainer.max_actor_ckpt_to_keep=10 \
  trainer.max_critic_ckpt_to_keep=10 \
  trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
  trainer.test_freq=5 \
  trainer.total_training_steps=$total_steps \
  trainer.total_epochs=10000