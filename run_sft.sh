#!/bin/bash
# Demo SFT training entry. Edit data paths / hyperparameters for your setup.

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sft-demo}"

data_dir="${DATA_DIR:-data}"
train_files="$data_dir/sft_train.parquet"
val_files="$data_dir/sft_val.parquet"
rl_test_files="$data_dir/val.parquet"  # optional generative eval via RL rollout

actor_model_path="${ACTOR_MODEL_PATH:-Qwen3-VL-8B-Instruct}"

actor_lr=1e-5
actor_lr_warmup_steps=50
max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 8))
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 2))

train_batch_size=1024
ppo_mini_batch_size=256
usp_size=1
infer_tp=1
total_training_steps=500

# ── Setup ─────────────────────────────────────────────────────────────────────
export HF_HOME=$OUTPUT_DIR/hf_cache
export WANDB_DIR=$OUTPUT_DIR/wandb
mkdir -p $OUTPUT_DIR/hf_cache $OUTPUT_DIR/wandb $OUTPUT_DIR/$EXPERIMENT_NAME

# ── Train ─────────────────────────────────────────────────────────────────────
export TURNOFF_THINK=1
export LOGGING_LEVEL=ERROR

HYDRA_ARGS=(
  # SFT data / loss path
  "data.train_files=$train_files"
  "data.val_files=$val_files"
  "data.train_batch_size=$train_batch_size"
  "data.max_prompt_length=$max_prompt_length"
  "data.max_response_length=$max_response_length"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "+data.lazy_load=False"
  "data.dataloader_num_workers=4"
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"

  # Shared actor training config
  "actor_rollout_ref.model.path=$actor_model_path"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.model.use_fused_kernels=True"
  "actor_rollout_ref.actor.optim.lr=$actor_lr"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=cosine"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=$actor_lr_warmup_steps"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.policy_loss.loss_mode=sft"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$usp_size"
  "actor_rollout_ref.actor.checkpoint.save_contents=[\"model\",\"extra\"]"

  # RL rollout worker is still used for weight sync and optional generative eval
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.7"
  "actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length + 1024))"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))"
  "actor_rollout_ref.rollout.max_num_seqs=1024"
  "actor_rollout_ref.rollout.n=1"

  # Trainer / logging / checkpointing
  "trainer.n_gpus_per_node=8"
  "trainer.nnodes=1"
  "trainer.logger=[\"console\",\"wandb\"]"
  "trainer.project_name=harmony"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME"
  "trainer.total_training_steps=$total_training_steps"
  "trainer.val_before_train=False"
  "trainer.test_freq=50"
  "trainer.save_freq=50"
  "trainer.max_actor_ckpt_to_keep=10"

  # Optional RL-style generative eval (drop these if not needed)
  "+trainer.rl_test_freq=25"
  "+data.rl_test_files=$rl_test_files"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml"
  "actor_rollout_ref.rollout.agent.default_agent_loop=agent_hub"
)

NCCL_DEBUG=WARN python3 train_sft.py "${HYDRA_ARGS[@]}"