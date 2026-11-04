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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn


class _Rollouts:
    def __init__(self, include_gen_uid, outputs_per_item):
        self.include_gen_uid = include_gen_uid
        self.outputs_per_item = outputs_per_item

    def generate_sequences(self, batch):
        indices = np.repeat(np.arange(len(batch)), self.outputs_per_item)
        rows = batch.non_tensor_batch["extra_info"][indices]
        keys = set().union(*(row["metrics"].keys() for row in rows))
        fields = {key: np.array([row["metrics"].get(key) for row in rows], dtype=object) for key in keys}
        fields["uid"] = batch.non_tensor_batch["uid"][indices]
        if self.include_gen_uid:
            fields["gen_uid"] = batch.non_tensor_batch["gen_uid"][indices]
        return DataProto.from_dict(
            tensors={
                "prompts": torch.ones(len(rows), 1, dtype=torch.long),
                "responses": torch.ones(len(rows), 1, dtype=torch.long),
                "rm_scores": torch.tensor([[row["score"]] for row in rows]),
            },
            non_tensors=fields,
            meta_info={"reward_extra_keys": list(keys)},
        )


def _validate(batch_size, include_gen_uid, outputs_per_item):
    rows = []
    for index, (source, score, metrics) in enumerate(
        [
            ("bsi", 1.0, {"bsi/quality": 0.0, "all/optional": 0.0}),
            ("bdu", 0.0, {"bdu/quality": 8.0}),
            ("bsi", 0.0, {"bsi/quality": 2.0, "all/optional": 2.0}),
        ]
    ):
        rows.append(
            {
                "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                "data_source": source,
                "extra_info": {"index": index, "score": score, "metrics": metrics},
            }
        )
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1, "do_sample": False}, "agent": {"num_workers": 1}}},
            "reward_model": {"enable": False},
            "trainer": {"log_val_generations": 0, "validation_data_dir": "unused"},
        }
    )
    trainer.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, decode=lambda *args, **kwargs: "text")
    trainer.val_dataloader = DataLoader(rows, batch_size=batch_size, collate_fn=collate_fn)
    trainer.global_steps = 0
    trainer.async_rollout_mode = True
    trainer.async_rollout_manager = _Rollouts(include_gen_uid, outputs_per_item)
    trainer.val_reward_fn = None
    dumped = {}
    trainer._dump_generations = lambda **kwargs: dumped.update(kwargs)
    return trainer._validate(), dumped["reward_extra_infos_dict"]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("include_gen_uid,outputs_per_item", [(False, 1), (True, 1), (True, 2)])
def test_sparse_metrics_align_across_validation_batches(batch_size, include_gen_uid, outputs_per_item):
    """Early, late and reappearing metrics retain alignment, including aggregated multi-agent outputs."""
    metrics, dumped = _validate(batch_size, include_gen_uid, outputs_per_item)
    assert dumped["reward"] == [1.0, 0.0, 0.0]
    assert dumped["bsi/quality"] == [0.0, None, 2.0]
    assert dumped["bdu/quality"] == [None, 8.0, None]
    assert dumped["all/optional"] == [0.0, None, 2.0]
    assert metrics["val-core/bsi/reward/mean@1"] == 0.5
    assert metrics["val-core/bdu/reward/mean@1"] == 0.0
    assert metrics["val-aux/bsi/bsi/quality/mean@1"] == 1.0
    assert metrics["val-aux/bdu/bdu/quality/mean@1"] == 8.0
    assert metrics["val-core/all/optional"] == 1.0
    assert not any("bsi/bdu/quality" in key or "bdu/bsi/quality" in key for key in metrics)
