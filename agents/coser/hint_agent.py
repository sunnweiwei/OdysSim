"""
CoSER hint agent — measures improvement from reflection hints.

Runs a fresh simulation where all character agents receive:
  1. The reference conversation as storyline guidance
  2. A coaching hint from the prior run

Returns new reward with reward_delta tracked in extra_info.
"""

import asyncio
import copy
import json
import logging
import random
import time
import uuid

from agents.utils import Agent, process_post_chat, remove_think, truncate_text
from agents.coser.hint import get_teacher_character_prompt
from agents.coser.prompt import (
    ENVIRONMENT, NSP,
    get_environment_prompt, get_nsp_prompt,
    remove_inner_thoughts, add_speaker_name, extract_nsp,
)
from agents.coser.coser_agent import SimpleAgent, evaluate_simulation

logger = logging.getLogger(__name__)


async def agent_loop(data: dict, context):
    """
    CoSER hint agent: re-run the simulation with hint-augmented character prompts.

    Expected extra_info fields (same as coser agent, plus):
      - hint: str — coaching brief from the prior run
      - old_reward: float — reward from the prior run
      - old_eval_result: dict — per-dimension scores from the prior run
    """
    item = data["extra_info"]
    circumstance = item["circumstance"]
    if isinstance(circumstance, str):
        circumstance = json.loads(circumstance)

    hint = item.get("hint", "")
    old_reward = float(item.get("old_reward", 0.0))
    wo_thought = item.get("wo_thought", False)
    continue_from = item.get("continue_from", 0)

    book_title = circumstance["book"]
    plot = circumstance["plot"]
    conversation = circumstance
    speaking_characters_w_env = conversation["speaking_characters_w_env"].copy()
    if ENVIRONMENT not in speaking_characters_w_env:
        speaking_characters_w_env.append(ENVIRONMENT)
    major_characters = conversation["major_characters"]
    character_profiles = circumstance["character_profiles"]
    plot_characters = [c["name"] for c in plot["key_characters"]]

    # Build enhanced character profiles (same as main agent)
    involved_character_profiles = {}
    for character in speaking_characters_w_env:
        if character == ENVIRONMENT:
            continue
        profile = character_profiles.get(character, "")
        if character in plot_characters:
            char_info = next((c for c in plot["key_characters"] if c.get("name", "") == character), {})
            if "description" in char_info:
                profile = char_info["description"].strip("\n") + "\n\n" + profile.strip("\n")
        profile = profile.strip(" \n")
        if profile:
            involved_character_profiles[character] = profile

    # Create agents — character agents get teacher prompts; NSP & ENVIRONMENT are unchanged
    init_msg = {"role": "user", "content": "===Conversation Start===\n\n"}
    character_agents = {}
    for character in speaking_characters_w_env + [NSP]:
        if character == NSP:
            agent = SimpleAgent(get_nsp_prompt(speaking_characters_w_env, conversation["scenario"]))
            agent.append(init_msg)
        elif character == ENVIRONMENT:
            agent = SimpleAgent(get_environment_prompt(major_characters, conversation["scenario"]))
            agent.append(init_msg)
        else:
            motivation = next(
                (c.get("motivation", "") or c.get("thought", "")
                 for c in conversation["key_characters"] if c.get("name") == character), ""
            )
            system_prompt = get_teacher_character_prompt(
                book_title=book_title,
                character=character,
                character_profile=truncate_text(involved_character_profiles.get(character, ""), 1000),
                background=truncate_text(plot.get("summary", ""), 1000),
                scenario=truncate_text(conversation.get("scenario", ""), 1000),
                motivation=truncate_text(motivation, 1000),
                wo_thought=wo_thought,
                other_character_profiles={
                    k: truncate_text(v, 1000) for k, v in involved_character_profiles.items()
                },
                hint=hint,
            )
            agent = Agent(
                context.llm_client,
                [{"role": "system", "content": system_prompt}, init_msg],
                context.tokenizer, context.config, prompt_turn=2,
            )
        character_agents[character] = agent

    # Run conversation (identical logic to main agent)
    dialogues = conversation.get("dialogues", [])
    max_rounds = 15
    agent_conversations = []
    current_speaker = speaking_characters_w_env[0]
    generated_speakers = set()

    for i_round in range(max_rounds):
        if current_speaker == "<END CHAT>":
            break

        speaker_agent = character_agents[current_speaker]
        if i_round < continue_from:
            response = dialogues[i_round]["message"] if i_round < len(dialogues) else ""
            speaker_agent.append({"role": "assistant", "content": response})
        else:
            response = await speaker_agent.step()
            if not response:
                break
            if current_speaker != ENVIRONMENT:
                generated_speakers.add(current_speaker)

        clean = remove_think(response, remove_unclosed=True)
        if not clean:
            break
        agent_conversations.append({"role": current_speaker, "content": clean})
        broadcast = add_speaker_name(remove_inner_thoughts(clean), current_speaker)
        for name, agent in character_agents.items():
            if name != current_speaker:
                agent.append({"role": "user", "content": broadcast})

        nsp_agent = character_agents[NSP]
        if i_round < continue_from:
            nsp_raw = dialogues[i_round + 1]["character"] if i_round < len(dialogues) - 1 else "<END CHAT>"
        else:
            nsp_raw = await nsp_agent.step()
            if not nsp_raw:
                break
        nsp_parsed = extract_nsp(nsp_raw) or (nsp_raw.split(":")[0].strip() if ":" in nsp_raw else nsp_raw)

        if nsp_parsed == "<END CHAT>" and i_round >= 5:
            current_speaker = "<END CHAT>"
        elif nsp_parsed in speaking_characters_w_env and nsp_parsed != current_speaker:
            current_speaker = nsp_parsed
        else:
            candidates = (
                (set(major_characters + [ENVIRONMENT]) - {current_speaker})
                or set(speaking_characters_w_env) - {current_speaker}
            )
            current_speaker = random.choice(list(candidates))
            nsp_agent.messages[-1]["content"] = nsp_parsed
            nsp_agent.append({"role": "user", "content": nsp_agent.messages[0]["content"]})
        agent_conversations.append({"role": NSP, "content": nsp_parsed})

    # Evaluate
    _eval_start = time.monotonic()
    new_reward, new_eval_result, _ = await evaluate_simulation(
        agent_conversations, circumstance, involved_character_profiles,
        continue_from=continue_from,
        structured_output=True,
    )
    eval_time = time.monotonic() - _eval_start

    reward_delta = new_reward - old_reward
    extra_info = {
        "coser-hint/reward": new_reward,
        "coser-hint/reward_delta": reward_delta,
        "coser-hint/delta_positive": int(reward_delta > 0),
        "coser-hint/eval_time": eval_time,
        **{
            f"coser-hint/{dim}": new_eval_result[dim]["score"]
            for dim in ("Storyline Consistency", "Character Fidelity", "Anthropomorphism", "Storyline Quality")
            if dim in new_eval_result
        },
    }

    outputs = []
    for name in generated_speakers:
        if character_agents[name].think_format_correct() == 0 and context.is_train:
            use_reward = new_reward / 2
        else:
            use_reward = new_reward
        out = await character_agents[name].get_agent_output(use_reward, extra_info=extra_info)
        if any(out.response_mask):
            outputs.append(out)

    return outputs