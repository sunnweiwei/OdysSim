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

import asyncio
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from agents.hugagent.agent import agent_loop
from verl.utils.dataset.rl_dataset import RLHFDataset


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant: "
        return self.encode(text) if tokenize else text


class _LLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    async def create_completion(self, input_ids, **kwargs):
        self.calls += 1
        ids = list(self.answer.encode())
        return {
            "choices": [
                {
                    "message": {
                        "content": self.answer,
                        "raw_output_ids": ids,
                        "response_log_probs": [0.0] * len(ids),
                    }
                }
            ]
        }


def _row(task):
    return {
        "item_id": f"{task}-test-0001",
        "split": "test",
        "task_type": task,
        "task_question": "Predict this person's response.",
        "demographics": {},
        "context_qas": [],
        "answer_options": {"A": "Yes", "B": "No"},
        "answer": "A",
        "question_type": "opinion",
        "scale": [1, 10],
        "user_answer": 5,
    }


def _context(answer, is_train=False):
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        llm_client=_LLM(answer),
        is_train=is_train,
        global_step=0,
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"prompt_length": 8192, "response_length": 256}},
                "trainer": {"experiment_name": "hugagent-test"},
            }
        ),
    )


@pytest.mark.parametrize("task,answer", [("belief_attribution", "A"), ("belief_update", "5")])
@pytest.mark.parametrize("index", ["missing", None, 0, 7])
def test_direct_agent_handles_published_metadata(monkeypatch, task, answer, index):
    """Real agent output and logging work with missing/null indices and preserve explicit indices."""
    row = _row(task)
    if index != "missing":
        row["index"] = index
    titles = []

    async def post_chat(messages, folder, title):
        titles.append(title)

    monkeypatch.setattr("agents.utils.post_chat", post_chat)
    context = _context(f"<answer>{answer}</answer>")
    output = asyncio.run(agent_loop({"data_source": "hugagent", "extra_info": row}, context))
    assert output.reward_score == 1.0
    assert output.extra_fields["reward_extra_info"]["all/score"] == 1.0
    assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs)
    expected_index = row["item_id"] if index in ("missing", None) else index
    assert titles == [f"val-hugagent-{expected_index}-(1.0)"]
    assert row.get("index") == (None if index == "missing" else index)


def test_test_split_is_rejected_before_inference():
    """Metadata normalization does not bypass the released test set's training guard."""
    context = _context("<answer>A</answer>", is_train=True)
    with pytest.raises(ValueError, match="evaluation only"):
        asyncio.run(agent_loop({"extra_info": _row("belief_attribution")}, context))
    assert context.llm_client.calls == 0


def test_dataset_assigns_unique_nested_and_top_level_indices(tmp_path):
    """Parquet null/missing indices get dataset positions while index=0 is preserved."""
    paths = []
    for name, extras in [
        ("unindexed", [{"item_id": "one"}, {"item_id": "two"}]),
        ("indexed", [{"item_id": "three", "index": 0}, {"item_id": "four", "index": None}]),
    ]:
        path = tmp_path / f"{name}.parquet"
        rows = [
            {"prompt": [{"role": "user", "content": "x"}], "data_source": "hugagent", "extra_info": extra}
            for extra in extras
        ]
        pq.write_table(pa.Table.from_pylist(rows), path)
        paths.append(str(path))
    dataset = RLHFDataset(paths, _Tokenizer(), OmegaConf.create({"filter_overlong_prompts": False}))
    assert [dataset[i]["index"] for i in range(4)] == [0, 1, 0, 3]
    assert [dataset[i]["extra_info"]["index"] for i in range(4)] == [0, 1, 0, 3]
