import uuid

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tracking import Tracking

from sft.dataset import SFTDataset, sft_collate_fn


class _MainProcessCollate:
    """Wrap a StatefulDataLoader so the collate_fn (padding) runs in the main
    process rather than the worker. Workers share only unpadded per-sample
    tensors (actual sequence length), which is much smaller than a fully padded
    batch, avoiding shared-memory exhaustion with large batch sizes."""

    def __init__(self, dataloader, collate_fn):
        self._dl = dataloader
        self._collate_fn = collate_fn
        # last-iter timing (seconds): time spent waiting on workers, time spent collating
        self.last_fetch_s = 0.0
        self.last_collate_s = 0.0

    def __iter__(self):
        import queue, threading, time

        q: queue.Queue = queue.Queue(maxsize=2)
        DONE = object()

        def producer():
            try:
                for samples in self._dl:
                    t0 = time.perf_counter()
                    batch = self._collate_fn(samples)
                    q.put((batch, time.perf_counter() - t0))
            except BaseException as e:
                q.put(e)
            else:
                q.put(DONE)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            t0 = time.perf_counter()
            item = q.get()
            self.last_fetch_s = time.perf_counter() - t0
            if item is DONE:
                return
            if isinstance(item, BaseException):
                raise item
            batch, self.last_collate_s = item
            yield batch

    def __len__(self):
        return len(self._dl)

    def state_dict(self):
        return self._dl.state_dict()

    def load_state_dict(self, state_dict):
        self._dl.load_state_dict(state_dict)


class SFTRayTrainer(RayPPOTrainer):
    """
    SFT trainer built on top of RayPPOTrainer.

    Reuses all verl infrastructure (FSDP/Megatron, vLLM weight sync, sequence
    parallel, multi-node, checkpointing) and only overrides:
      - _create_dataloader  → SFT dataset for train/val; RL dataset for rl eval
      - fit                 → simplified loop (no rollout, no reward, no advantage)
      - _sft_val            → fast loss eval on val_files
    """

    # ── Dataloader ──────────────────────────────────────────────────────────────

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from functools import partial
        config = self.config
        tokenizer = self.tokenizer
        lazy = config.data.get("lazy_load", False)
        seed = config.data.get("seed", 42)
        num_workers = config.data.get("dataloader_num_workers", 4)
        train_batch_size = config.data.train_batch_size

        collate = partial(
            sft_collate_fn,
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
        )

        # ── SFT train dataset ──
        self.train_dataset = SFTDataset(
            config.data.train_files, tokenizer, config=config, lazy=lazy, seed=seed
        )
        self.train_dataloader = _MainProcessCollate(
            StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                collate_fn=list,   # workers return raw sample lists — no padding
                num_workers=num_workers,
                drop_last=True,
            ),
            collate,
        )

        # ── SFT val dataset (fast loss eval, test_freq) ──
        val_files = config.data.get("val_files")
        if val_files:
            self.sft_val_dataset = SFTDataset(
                val_files, tokenizer, config=config, lazy=lazy, seed=0
            )
            val_batch_size = config.data.get("val_batch_size") or train_batch_size
            self.sft_val_dataloader = _MainProcessCollate(
                StatefulDataLoader(
                    dataset=self.sft_val_dataset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    collate_fn=list,
                    num_workers=num_workers,
                    drop_last=False,
                ),
                collate,
            )
        else:
            self.sft_val_dataloader = None

        # ── Posttrain dataset (N-step LR-fixed warmup before each rl eval) ──
        posttrain_files = config.data.get("posttrain_files")
        if posttrain_files:
            self.posttrain_dataset = SFTDataset(
                posttrain_files, tokenizer, config=config, lazy=lazy, seed=seed
            )
            posttrain_batch_size = config.data.get("posttrain_batch_size", None) or train_batch_size
            self.posttrain_dataloader = _MainProcessCollate(
                StatefulDataLoader(
                    dataset=self.posttrain_dataset,
                    batch_size=posttrain_batch_size,
                    shuffle=True,
                    collate_fn=list,
                    num_workers=num_workers,
                    drop_last=True,
                ),
                collate,
            )
            self._posttrain_iter = None
        else:
            self.posttrain_dataloader = None
            self._posttrain_iter = None

        # ── RL val dataset (generative eval, rl_test_freq) ──
        # parent _validate() uses self.val_dataloader, so wire it here.
        rl_test_files = config.data.get("rl_test_files")
        if rl_test_files:
            from verl.trainer.main_ppo import create_rl_dataset
            from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

            rl_val_dataset = create_rl_dataset(
                rl_test_files,
                config.data,
                tokenizer,
                self.processor,
                is_train=False,
            )
            self.val_dataset = rl_val_dataset
            val_batch_size = config.data.get("val_batch_size") or len(rl_val_dataset)
            self.val_dataloader = StatefulDataLoader(
                dataset=rl_val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                collate_fn=rl_collate_fn,
                num_workers=num_workers,
                drop_last=False,
            )
        else:
            self.val_dataset = None
            self.val_dataloader = None

        # total_training_steps used by parent helpers (_save_checkpoint, etc.)
        total = len(self.train_dataloader) * config.trainer.total_epochs
        if config.trainer.get("total_training_steps") is not None:
            total = config.trainer.total_training_steps
        self.total_training_steps = total

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        print(f"[SFT] train batches/epoch: {len(self.train_dataloader)}, "
              f"total steps: {self.total_training_steps}")

    # ── Training loop ───────────────────────────────────────────────────────────

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self._logger = logger

        self.global_steps = 0
        self._total_tokens = 0
        self._total_response_tokens = 0
        self._load_checkpoint()

        test_freq = self.config.trainer.get("test_freq", -1)
        rl_test_freq = self.config.trainer.get("rl_test_freq", -1)
        save_freq = self.config.trainer.get("save_freq", -1)

        # optional: RL val before training (mirrors val_before_train in RL)
        if self.config.trainer.get("val_before_train", False) and self.val_dataloader is not None:
            if self.posttrain_dataloader is not None:
                val_metrics = self._posttrain_and_eval()
            else:
                val_metrics = self._validate()
            logger.log(val_metrics, step=self.global_steps)

        # optional: SFT val before training (fast loss eval on sft_val_dataloader)
        if self.config.trainer.get("val_before_train", False) and self.sft_val_dataloader is not None:
            val_metrics = self._sft_val()
            logger.log(val_metrics, step=self.global_steps)

        # val_only: run validation once, skip training loop
        if self.config.trainer.get("val_only", False):
            return

        pbar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="SFT",
        )

        import time

        while self.global_steps < self.total_training_steps:
            t_iter_start = time.perf_counter()
            for batch_dict in self.train_dataloader:
                data_s = time.perf_counter() - t_iter_start

                self.global_steps += 1
                is_last = self.global_steps >= self.total_training_steps

                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch.meta_info["global_token_num"] = (
                    batch.batch["attention_mask"].sum(dim=-1).tolist()
                )

                # ── actor update (SFT loss) ──
                t0 = time.perf_counter()
                actor_output = self._update_actor(batch)
                update_s = time.perf_counter() - t0

                metrics = {}
                if hasattr(actor_output, "meta_info") and "metrics" in actor_output.meta_info:
                    from verl.utils.metric import reduce_metrics
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                tokens = int(batch.batch["attention_mask"].sum())
                response_tokens = int(batch.batch["response_mask"].sum())
                self._total_tokens += tokens
                self._total_response_tokens += response_tokens
                metrics["train/tokens"] = tokens
                metrics["train/response_tokens"] = response_tokens
                metrics["train/total_tokens"] = self._total_tokens
                metrics["train/total_response_tokens"] = self._total_response_tokens

                step_s = time.perf_counter() - t_iter_start
                metrics["timing/step_s"] = step_s
                metrics["timing/update_s"] = update_s
                metrics["timing/data_s"] = data_s
                metrics["timing/gpu_busy"] = update_s / max(step_s, 1e-6)

                print(
                    f"[step {self.global_steps:>5}] step={step_s:6.2f}s | "
                    f"update={update_s:6.2f}s | data={data_s:5.2f}s | "
                    f"busy={update_s / max(step_s, 1e-6):.0%}",
                    flush=True,
                )
                logger.log(metrics, step=self.global_steps)
                pbar.update(1)
                t_iter_start = time.perf_counter()

                # ── SFT val: loss on val_files ──
                if (
                    self.sft_val_dataloader is not None
                    and test_freq > 0
                    and (is_last or self.global_steps % test_freq == 0)
                ):
                    val_metrics = self._sft_val()
                    logger.log(val_metrics, step=self.global_steps)

                # ── RL val: generative eval via vLLM + agent loop ──
                if (
                    self.val_dataloader is not None
                    and self.val_reward_fn is not None
                    and rl_test_freq > 0
                    and (is_last or self.global_steps % rl_test_freq == 0)
                ):
                    if self.posttrain_dataloader is not None:
                        val_metrics = self._posttrain_and_eval()
                    else:
                        val_metrics = self._validate()
                    logger.log(val_metrics, step=self.global_steps)

                # ── checkpoint ──
                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_checkpoint()

                if is_last:
                    self._save_checkpoint()
                    return

    # ── Posttrain-before-eval ──────────────────────────────────────────────────

    def _posttrain_and_eval(self) -> dict:
        """Snapshot → N posttrain steps at fixed LR → RL eval → restore snapshot.

        Lets a midtraining SFT run evaluate the agent-loop reward as if the model
        had been posttrained, without actually polluting midtrain state.
        """
        import os
        snap_path = os.path.join(self.config.trainer.posttrain_snapshot_dir, "snapshot")
        n_steps = int(self.config.trainer.posttrain_steps)
        lr = float(self.config.trainer.posttrain_lr)
        mbs_override = self.config.trainer.get("posttrain_ppo_mini_batch_size", None)
        saved_gs = self.global_steps

        # Optional: also evaluate the raw midtrain ckpt (pre-posttrain) and
        # surface metrics under "mid-val-core/" / "mid-val-aux/".
        eval_raw = bool(self.config.trainer.get("posttrain_eval_raw_midtrain", False))
        if eval_raw:
            raw_metrics = self._validate()
            raw_metrics = {
                k.replace("val-", "mid-val-", 1) if k.startswith("val-") else k: v
                for k, v in raw_metrics.items()
            }
            # Log immediately so the dashboard updates without waiting for the
            # subsequent posttrain + posttrained-eval to finish.
            if getattr(self, "_logger", None) is not None:
                self._logger.log(raw_metrics, step=self.global_steps)

        self.actor_rollout_wg.save_snapshot(snap_path)
        try:
            self.actor_rollout_wg.set_actor_lr(lr)
            if mbs_override is not None:
                self.actor_rollout_wg.set_actor_ppo_mini_batch_size(int(mbs_override))
            pt_pbar = tqdm(total=n_steps, desc=f"posttrain@gs{saved_gs}")
            for _ in range(n_steps):
                if self._posttrain_iter is None:
                    self._posttrain_iter = iter(self.posttrain_dataloader)
                try:
                    batch_dict = next(self._posttrain_iter)
                except StopIteration:
                    self._posttrain_iter = iter(self.posttrain_dataloader)
                    batch_dict = next(self._posttrain_iter)

                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object,
                )
                batch.meta_info["global_token_num"] = (
                    batch.batch["attention_mask"].sum(dim=-1).tolist()
                )
                self._update_actor(batch)
                pt_pbar.update(1)
            pt_pbar.close()

            val_metrics = self._validate()
        finally:
            if mbs_override is not None:
                self.actor_rollout_wg.restore_actor_ppo_mini_batch_size()
            self.actor_rollout_wg.load_snapshot(snap_path)
            self.global_steps = saved_gs

        return val_metrics

    # ── SFT val ─────────────────────────────────────────────────────────────────

    def _sft_val(self) -> dict:
        """Compute CE loss on val_files. No generation — fast.

        Reports both overall val/sft_loss and per-dataset val/sft_loss/<data_source>.
        """
        from collections import defaultdict
        total_loss = 0.0
        total_tokens = 0
        per_src_loss = defaultdict(float)
        per_src_tokens = defaultdict(float)

        for batch_dict in self.sft_val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                dtype=object,
            )

            # forward pass only — _compute_old_log_prob handles legacy/new engine
            log_prob_output, _ = self._compute_old_log_prob(batch)
            log_probs = log_prob_output.batch["old_log_probs"]  # (bsz, resp_len)
            mask = batch.batch["response_mask"]                  # (bsz, resp_len)

            batch_loss = -(log_probs * mask).sum().item()
            batch_tokens = mask.sum().item()
            total_loss += batch_loss
            total_tokens += batch_tokens

            # Per-source breakdown.
            sources = batch.non_tensor_batch.get("data_source")
            if sources is not None:
                # per-row loss and token counts (keep unary minus on the tensor, not the list)
                row_loss = (-(log_probs * mask).sum(dim=-1)).detach().cpu().tolist()
                row_tokens = mask.sum(dim=-1).detach().cpu().tolist()
                for src, l, t in zip(sources, row_loss, row_tokens):
                    per_src_loss[str(src)] += float(l)
                    per_src_tokens[str(src)] += float(t)

        loss = total_loss / max(total_tokens, 1)
        metrics = {"val/sft_loss": loss}
        for src in sorted(per_src_loss):
            if per_src_tokens[src] > 0:
                metrics[f"val/sft_loss/{src}"] = per_src_loss[src] / per_src_tokens[src]
        return metrics
