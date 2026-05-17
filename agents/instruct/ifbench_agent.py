"""
IFBench agent for Harmony evaluation.

Evaluates instruction-following using the IFBench benchmark (294 instances).
Each instance contains a prompt with embedded constraint descriptions and a list
of constraint IDs + parameters. The model responds to the prompt, and the reward
is computed as the fraction of constraints satisfied (instruction-level accuracy).

Based on: https://github.com/allenai/IFBench

Dependencies (nltk, emoji, syllapy, langdetect, immutabledict) are optional —
if not installed, evaluate_strict returns all-False and reward is 0.0.
"""

import json
import logging

from agents.utils import Agent, process_post_chat, remove_think

logger = logging.getLogger(__name__)

try:
    from agents.instruct.instructions_registry import INSTRUCTION_DICT as _IFBENCH_DICT
    from agents.instruct.verifiable_instructions import INSTRUCTION_DICT as _VI_DICT
    INSTRUCTION_DICT = {**_IFBENCH_DICT, **_VI_DICT}
except ImportError as e:
    logger.warning("IFBench evaluation dependencies not available (%s). Rewards will be 0.", e)
    INSTRUCTION_DICT = None


def evaluate_strict(response: str, instruction_id_list: list, kwargs: list, prompt: str) -> tuple[bool, list[bool]]:
    """
    Evaluate response against all constraints (strict mode).
    Returns (follow_all, follow_list). All-False if dependencies are missing.
    """
    if INSTRUCTION_DICT is None:
        return False, [False] * len(instruction_id_list)

    is_following_list = []
    for index, instruction_id in enumerate(instruction_id_list):
        if instruction_id not in INSTRUCTION_DICT:
            logger.warning("Unknown instruction_id: %s", instruction_id)
            is_following_list.append(False)
            continue

        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        clean_kwargs = {k: v for k, v in (kwargs[index] or {}).items() if v is not None}
        try:
            instruction.build_description(**clean_kwargs)
        except AttributeError:
            # Some Nemotron kwargs store single-element lists where strings are expected
            # (e.g. keyword1: ["help"] instead of "help"). Unwrap and retry.
            normalized = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in clean_kwargs.items()}
            instruction.build_description(**normalized)

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        if response and response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return all(is_following_list), is_following_list


async def agent_loop(data, context):
    """
    IFBench: Evaluate instruction-following on a single instance.

    data["extra_info"] contains:
        prompt              - user prompt with embedded constraint descriptions
        instruction_id_list - list of constraint type IDs, e.g. ["count:keywords_multiple"]
        kwargs              - list of dicts with constraint parameters (one per instruction)
        key                 - instance index (0-293)

    Reward = fraction of constraints satisfied (partial credit for RL training).
    A reward of 1.0 means all constraints were followed.
    """
    row = data["extra_info"]
    prompt = row.get("task_prompt", "")
    instruction_id_list = json.loads(row.get("instruction_id_list", "[]"))
    kwargs = json.loads(row.get("kwargs", "[]"))

    chat = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]

    agent = Agent(
        context.llm_client, chat, context.tokenizer, context.config,
        prompt_turn=2, enable_think=True,
    )
    response = await agent.step()

    content = remove_think(response)

    follow_all, follow_list = evaluate_strict(content, instruction_id_list, kwargs, prompt)

    n = len(follow_list)
    reward = sum(follow_list) / n if n > 0 else 0.0

    output = await agent.get_agent_output(reward, extra_info={
        "all/score": reward,
        "ifbench/reward": reward,
        "ifbench/follow_all": float(follow_all),
        "ifbench/num_constraints": float(n),
        "ifbench/constraints_followed": float(sum(follow_list)),
    })

    await process_post_chat(data, context, agent.chat, output)
    return output
