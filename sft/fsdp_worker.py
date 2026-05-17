import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


class SFTAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """SFT worker: keeps params and optimizer on GPU during training,
    offloads only when vLLM needs memory for eval.

    param_offload / optimizer_offload in fsdp_config control whether to offload
    during vLLM eval — NOT during SFT training steps. Training always keeps
    everything on GPU.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        # Save the user's intended offload settings (controlling vLLM eval only),
        # then disable them so the base class never offloads during training.
        self._sft_offload_param = self._is_offload_param
        self._sft_offload_optimizer = self._is_offload_optimizer
        self._is_offload_param = False
        self._is_offload_optimizer = False
        # Base class already offloaded to CPU during init if flags were True —
        # reload now so training starts with everything on GPU.
        if self._is_actor:
            if self._sft_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            if self._sft_offload_optimizer:
                load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

            # Dedicated snapshot manager for posttrain-before-eval. Always saves
            # model+optimizer+extra (scheduler+RNG), independent of the real
            # checkpoint's save_contents. Written to a path outside default_local_dir
            # so _load_checkpoint's latest-ckpt discovery never sees it.
            self.snapshot_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=OmegaConf.create({
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"],
                }),
            )

    def _load_optimizer_to_gpu(self):
        """Reload optimizer states from CPU after vLLM eval offloaded them."""
        if not getattr(self, "_optimizer_offloaded", False):
            return
        device_id = get_device_id()
        for state in self.actor_optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor) and value.device.type != "cuda":
                    state[key] = value.to(device_id)
        torch.cuda.synchronize()
        self._optimizer_offloaded = False

    async def rollout_mode(self):
        if self._is_actor:
            if self._sft_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                self._optimizer_offloaded = True
        await super().rollout_mode()
        # After vLLM has the weights, offload FSDP params to free GPU memory.
        if self._is_actor and self._sft_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            self._params_offloaded = True

    async def trainer_mode(self):
        # Reload params before training resumes.
        if self._is_actor and getattr(self, "_params_offloaded", False):
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            self._params_offloaded = False
        await super().trainer_mode()
        self._load_optimizer_to_gpu()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        self._load_optimizer_to_gpu()
        return super().update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_snapshot(self, path):
        self.snapshot_manager.save_checkpoint(
            local_path=path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_snapshot(self, path):
        self.snapshot_manager.load_checkpoint(local_path=path, hdfs_path=None, del_local_after_load=False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_ppo_mini_batch_size(self, bsz):
        # Accept the raw (pre-normalization) user-facing value, mirror the same
        # normalization fsdp_workers.py applies at init.
        bsz = int(bsz) * self.config.rollout.n
        bsz = bsz // (self.device_mesh.size() // self.ulysses_sequence_parallel_size)
        assert bsz > 0, f"normalized posttrain ppo_mini_batch_size {bsz} must be > 0"
        self.config.actor.ppo_mini_batch_size = bsz

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_lr(self, lr):
        # Override base_lrs so verl's scheduler (which re-derives lr from
        # base_lrs * lambda(step) every step) yields `lr` until snapshot restore.
        lr = float(lr)
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.base_lrs = [lr] * len(self.actor_lr_scheduler.base_lrs)
        for g in self.actor_optimizer.param_groups:
            g["lr"] = lr


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_ppo_mini_batch_size(self, unnormalized_mbs):
        if not self._is_actor:
            return
        rollout_n = getattr(self.config.rollout, "n", 1) if hasattr(self.config, "rollout") else 1
        ws = self.device_mesh.size()
        sp = self.ulysses_sequence_parallel_size
        new_normalized = (int(unnormalized_mbs) * rollout_n) // (ws // sp)
        if not hasattr(self, "_saved_ppo_mini_batch_size"):
            self._saved_ppo_mini_batch_size = self.actor.config.ppo_mini_batch_size
        self.actor.config.ppo_mini_batch_size = new_normalized

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_actor_ppo_mini_batch_size(self):
        if not self._is_actor:
            return
        if hasattr(self, "_saved_ppo_mini_batch_size"):
            self.actor.config.ppo_mini_batch_size = self._saved_ppo_mini_batch_size
            delattr(self, "_saved_ppo_mini_batch_size")
