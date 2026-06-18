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
MMToM agent for Harmony evaluation.

Evaluates multi-modal Theory-of-Mind (ToM) reasoning using the MMToM-QA
dataset. Single-phase: prompt LLM with context+question (binary a/b choice),
extract answer, compare to ground truth.

Based on: "MMToM-QA: Multimodal Theory of Mind Question Answering"
(Jin et al., 2024). The dataset probes whether agents can reason about a
character's beliefs about object locations in simulated apartments.

Each question encodes the full apartment state + character actions and asks
which belief the character holds (a or b). The data `question` field contains
both the context narrative and the actual question with answer choices.

Scoring: exact match of extracted answer letter (a/b) vs. correct answer.
"""

import logging
import re

from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)

PROMPT_PREFIX = """Imagine that you are an observer in the scenario. Assume that the characters can perceive every scene in their location but not scenes occurring elsewhere. If something is being moved, that means it is not in its original location anymore. You should carefully analyze the character's actions and beliefs based on their observations. Provide your reasoning within the <reasoning></reasoning> tag. For the answer, use <answer>(put your answer here)</answer> and only include the letter (a or b) of your chosen option.

"""


def build_prompt(row: dict) -> str:
    """Build the prompt for a single MMToM question."""
    # The data's `question` field contains the full context + question + answer choices
    return PROMPT_PREFIX + row["question"]


def extract_answer(response: str) -> str:
    """Extract letter answer strictly from <answer>X</answer> tag. Returns '' if not found."""
    if not response:
        return ""
    match = re.search(r"<answer>\s*([ab])\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def compute_mmtom_aggregates(results: list[dict]) -> dict:
    """Compute accuracy overall, by question_type, and by episode."""
    by_type: dict[str, list[float]] = {}
    by_episode: dict[str, list[float]] = {}
    all_rewards: list[float] = []

    for r in results:
        if not isinstance(r, dict):
            continue
        reward = r.get("reward", 0.0)
        all_rewards.append(reward)
        qt = str(r.get("question_type", "unknown"))
        by_type.setdefault(qt, []).append(reward)
        ep = str(r.get("episode", "unknown"))
        by_episode.setdefault(ep, []).append(reward)

    aggregates = {}
    if all_rewards:
        aggregates["accuracy_overall"] = sum(all_rewards) / len(all_rewards)
    for qt, rewards in sorted(by_type.items()):
        aggregates[f"accuracy_type_{qt}"] = sum(rewards) / len(rewards)
    for ep, rewards in sorted(by_episode.items(), key=lambda x: (len(x[0]), x[0])):
        aggregates[f"accuracy_episode_{ep}"] = sum(rewards) / len(rewards)
    return aggregates


async def agent_loop(data, context):
    """
    MMToM: Evaluate ToM belief reasoning on a single binary question.

    Args:
        data: {
            "row": dict with keys:
                question      - full context + question + answer choices string
                answer        - correct answer letter ('a' or 'b')
                question_type - float type identifier (e.g., 1.3)
                test          - list of test component names
                episode       - episode id (int)
                start_time    - int
                end_time      - int
                answer_list   - list of valid answer letters
        }
        context: {
            "client": AsyncOpenAI instance,
            "model": str model name
        }

    Returns:
        {
            "reward": float (1.0 if correct, 0.0 otherwise),
            "chat": list of message dicts,
            "predicted": str extracted answer letter,
            "correct": str ground truth letter,
            "is_correct": bool,
            "question_type": float,
            "episode": int,
        }
    """
    row = data["extra_info"]

    prompt = build_prompt(row)
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    predicted = remove_think(response)
    predicted = extract_answer(predicted)
    correct_answer = str(row.get("answer", "")).strip().lower()
    is_correct = predicted == correct_answer
    reward = 1.0 if is_correct else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "all/score": reward,
            "mmtom/reward": reward,
            "mmtom/response_length": len(response.split()) if response else 0,
        },
    )
    await process_post_chat(data, context, agent.chat, output)
    return output

    return {
        "reward": reward,
        "chat": chat,
        "predicted": predicted,
        "correct": correct_answer,
        "is_correct": is_correct,
        "question_type": row.get("question_type", ""),
        "episode": row.get("episode", ""),
    }
