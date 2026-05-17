#!/bin/bash
# Demo RL training entry. Edit data paths / hyperparameters for your setup.

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rl-demo}"

data_dir="${DATA_DIR:-data}"
train_files="$data_dir/train.parquet"
val_files="$data_dir/val.parquet"

actor_model_path="${ACTOR_MODEL_PATH:-Qwen3-VL-8B-Instruct}"

# ── Hyperparameters ───────────────────────────────────────────────────────────
default_agent_loop="agent_hub"
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
  trainer.project_name=harmony \
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