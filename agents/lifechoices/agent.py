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
LifeChoices agent for Harmony evaluation.

Evaluates LLMs' role-playing capabilities in making persona-driven life choices
for literary characters. Single-phase: prompt LLM with character profile + MCQ,
extract choice, compare to ground truth.

Based on: "Character is Destiny: Can Large Language Models Simulate
Persona-Driven Decisions in Role-Playing?" (Xu et al., 2024)
"""

import copy
import re
import uuid

from agents.utils import Agent, process_post_chat, remove_think

PROMPT_TEMPLATE = """Please play the role of {character_name} based on the Profile and make your life choice under the Scenario regarding Question. Return the option letter (A, B, C, or D) that your character should most appropriately choose in the current scenario. The Profile consists of Description and Memory, where Description is an overall description of the character, and Memory consists of specific events the character has experienced.

# Inputs:
1. Profile:
1.1. Description
{character_name}

1.2. Memory
{input_text}

2. Scenario:
{scenario}

3. Question:
{question}

4. Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

# Outputs:
Think step by step about what {character_name} would choose, then output your final answer in the format:
<answer>X</answer>
where X is exactly one letter: A, B, C, or D."""


def create_prompt(character_data: dict) -> str:
    """Create structured prompt for the character decision task."""
    mcq = character_data["Multiple Choice Question"]
    options = mcq["Options"]
    return PROMPT_TEMPLATE.format(
        character_name=character_data["character_name"],
        input_text=character_data["input_text"],
        scenario=mcq["Scenario"],
        question=mcq["Question"],
        option_a=options[0] if len(options) > 0 else "",
        option_b=options[1] if len(options) > 1 else "",
        option_c=options[2] if len(options) > 2 else "",
        option_d=options[3] if len(options) > 3 else "",
    )


def extract_choice(response: str) -> str | None:
    """Extract the choice letter (A, B, C, or D) from the LLM response."""
    if not response:
        return None

    response = response.strip().upper()

    if response in ["A", "B", "C", "D"]:
        return response

    patterns = [
        r"<answer>\s*([ABCD])\s*</answer>",
        r"^([ABCD])\s*[.):,]",
        r"choice[:\s]+([ABCD])",
        r"answer[:\s]+([ABCD])",
        r"\(([ABCD])\)",
        r"^([ABCD])$",
        r"([ABCD])\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    for char in response:
        if char in "ABCD":
            return char

    return None


async def agent_loop(data: dict, context):
    """
    LifeChoices: Evaluate a single character's persona-driven decision.

    Args:
        data: {
            "character_data": dict with keys: character_name, input_text,
                              book, Multiple Choice Question (with Scenario,
                              Question, Options, Correct Answer)
        }
        context: {
            "client": AsyncOpenAI instance,
            "model": str model name
        }

    Returns:
        {"reward": float, "chat": list, "predicted": str, "correct": str}
    """
    character_data = data["extra_info"]

    prompt = create_prompt(character_data)
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=True)
    response = await agent.step()

    content = remove_think(response)
    predicted = extract_choice(content)

    correct_answer = character_data.get("Multiple Choice Question", {}).get("Correct Answer", "")
    if correct_answer:
        correct_answer = correct_answer.strip().upper()
        if len(correct_answer) > 1:
            correct_answer = correct_answer[0]

    is_correct = predicted == correct_answer if predicted else False
    reward = 1.0 if is_correct else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "lifechoices/reward": reward,
            "lifechoices/response_length": len(response.split()) if response else 0,
            "all/score": reward,
            "all/score_v1": reward,
        },
    )

    # ===========================================================================
    # Hint + second attempt (mirrors sotopia copy-agent pattern)
    # Only generate a hint when the model answered incorrectly.
    # ===========================================================================
    extra = {}
    hint = None
    if getattr(context.config.algorithm, "agent_version", None) == "copy" and not is_correct:
        from agents.lifechoices.hint import generate_hint

        hint = await generate_hint(character_data, content)
        if hint:
            extra["hint"] = hint

    if getattr(context.config.algorithm, "agent_version", None) == "copy" and context.is_train and hint:
        from agents.lifechoices.hint_agent import agent_loop as hint_agent_loop

        data["extra_info"]["hint"] = hint
        data["extra_info"]["old_reward"] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = "hint_agent"
        output = [output, copy_agent_output, hint_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output

    return {
        "reward": reward,
        "chat": chat,
        "predicted": predicted,
        "correct": correct_answer,
        "character_name": character_data.get("character_name"),
        "book": character_data.get("book"),
    }
