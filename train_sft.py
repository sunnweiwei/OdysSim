# Copyright 2025 Individual Contributor: OdysSim Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SFT trainer entry point. Mirrors train_ppo.py but uses SFTRayTrainer.

Usage: see run_sft.sh
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import auto_set_device


class SFTTaskRunner(TaskRunner):
    """
    Slimmed-down TaskRunner for SFT:
      - sets up actor+rollout worker (FSDP + vLLM)
      - skips critic and reward model workers
      - uses SFTRayTrainer
    """

    def run(self, config):
        from pprint import pprint

        from sft.trainer import SFTRayTrainer
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        print(f"SFTTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # actor + vLLM rollout worker (same as RL — needed for weight sync + RL eval)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from sft.fsdp_worker import SFTAsyncActorRolloutRefWorker

            actor_rollout_cls = SFTAsyncActorRolloutRefWorker
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        # no critic, no reward model for SFT

        resource_pool_manager = self.init_resource_pool_mgr(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # val_reward_fn is used for RL generative eval (_validate).
        # Only loaded when rl_test_files is set; otherwise None disables RL eval.
        rl_test_files = config.data.get("rl_test_files")
        if rl_test_files:
            val_reward_fn = load_reward_manager(
                config,
                tokenizer,
                num_examine=1,
                **config.reward_model.get("reward_kwargs", {}),
            )
        else:
            val_reward_fn = None

        trainer = SFTRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=None,
            val_reward_fn=val_reward_fn,
            # datasets are built inside SFTRayTrainer._create_dataloader
            train_dataset=None,
            val_dataset=None,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)

    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_cls = ray.remote(num_cpus=1)(SFTTaskRunner)
    runner = task_runner_cls.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
