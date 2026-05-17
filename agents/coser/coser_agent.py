"""
CoSER agent for Harmony evaluation.

Character-Oriented Simulation and Evaluation for Role-playing.
Runs GCA (Given-Circumstance Acting) simulation with multiple LLM agents
(character, environment, NSP), then evaluates the simulation quality using
LLM-based judging across 4 dimensions.

Based on: https://arxiv.org/abs/2403.06411
"""

import asyncio
import copy
import json
import logging
import random
import re
import uuid
from pydantic import BaseModel, Field, ConfigDict
from agents.utils import Agent, call_openai, call_openai_parse, process_post_chat, remove_think, get_judge_model, get_judge_reasoning
from agents.coser.prompt import (
    ENVIRONMENT, NSP, SPECIAL_CHARACTERS,
    CRITIC_TEMPLATE, COMBINED_CRITIC_TEMPLATE, DIMENSION_DETAILS, DIMENSIONS,  # noqa: F401
    get_character_prompt, get_environment_prompt, get_nsp_prompt,
    remove_inner_thoughts, add_speaker_name, extract_nsp,
    calculate_bleu_rouge, extract_json_from_text
)


# ---------------------------------------------------------------------------
# Simple agent class for managing conversation state
# ---------------------------------------------------------------------------

class SimpleAgent:
    """Lightweight agent that manages conversation history and LLM calls."""

    def __init__(self, system_prompt):
        self.messages = [{"role": 'system', "content": system_prompt}]

    async def step(self):
        """Generate a response and append it to history."""
        response = await call_openai(self.messages, model='gpt-5.4-nano', reasoning_effort='none')
        if not response or response.startswith("Error after"):
            return None
        self.append({'role': 'assistant', 'content': response})
        return response

    def append(self, turn):
        """Append a message to history, merging consecutive same-role messages."""
        role = turn['role']
        message = turn['content']
        if message:
            if self.messages and self.messages[-1]["role"] == role:
                self.messages[-1]["content"] += "\n\n" + message
            else:
                self.messages.append({"role": role, "content": message})


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM judge output
# ---------------------------------------------------------------------------

class DimEval(BaseModel):
    reasoning: str
    score: int

class CoserEvalResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    storyline_consistency: DimEval = Field(alias="Storyline Consistency")
    anthropomorphism: DimEval = Field(alias="Anthropomorphism")
    character_fidelity: DimEval = Field(alias="Character Fidelity")
    storyline_quality: DimEval = Field(alias="Storyline Quality")


# ---------------------------------------------------------------------------
# Evaluation: judge a single simulation
# ---------------------------------------------------------------------------

async def evaluate_simulation(
    simulation, circumstance, involved_character_profiles,
    continue_from=0,
    judge_mode="combined",
    structured_output=False,
):
    """
    Evaluate a simulation against ground truth using LLM-based judging.

    Returns per-dimension scores, BLEU/ROUGE-L, and an overall reward.
    """
    # Clean simulation (remove NSP messages and inner thoughts)
    sim_clean = [m for m in simulation if m["role"] != NSP]
    sim_clean = [
        m if m["role"] == ENVIRONMENT else {**m, "content": remove_inner_thoughts(m["content"])}
        for m in sim_clean
    ]

    reference = circumstance["dialogues"]
    reference = [
        m if m["character"] == ENVIRONMENT else {**m, "message": remove_inner_thoughts(m["message"])}
        for m in reference
    ]

    simulation_str = "\n\n".join([f"{m['role']}: {m['content']}".strip("\n") for m in sim_clean])
    reference_str = "\n\n".join([f"{m['character']}: {m['message']}".strip("\n") for m in reference])

    book_title = circumstance["book"]
    scenario_str = circumstance["scenario"]
    character_profile_str = "\n\n".join([
        f"### {char}\n\n{profile.strip()}"
        for char, profile in involved_character_profiles.items()
    ])
    major_characters = circumstance["major_characters"]

    # Additional instructions when using continue_from
    additional_instructions = ""
    if continue_from > 0:
        additional_instructions = (
            f"Please note that the first {continue_from} messages in the simulated "
            "conversation are the same as the reference. Focus your evaluation only "
            "on the content after these messages."
        )

    def validate_dim(parsed, dim):
        """Validate that a judge response has score (1-10) and reasoning."""
        if not isinstance(parsed, dict) or dim not in parsed:
            return False
        entry = parsed[dim]
        if not isinstance(entry, dict):
            return False
        score = entry.get("score")
        if score is None or not isinstance(score, int | float):
            return False
        return 1 <= score <= 10

    def score_dim(parsed, dim):
        """Convert a LLM score (1-10) to a 0-100 scale."""
        if not parsed or dim not in parsed:
            return 0.0, ""
        entry = parsed[dim]
        raw_score = entry.get("score", 0)
        reasoning = entry.get("reasoning", "")
        clamped = max(1, min(int(raw_score), 10))
        return round((clamped - 1) * 100.0 / 9.0, 1), reasoning

    eval_result = {}
    max_retries = 2

    if judge_mode == "combined":
        # Build combined dimension sections with focus, criteria, and anchors
        dim_parts = []
        for dim in DIMENSIONS:
            details = DIMENSION_DETAILS[dim]
            criteria = details["criteria"]
            if "{major_characters}" in criteria:
                criteria = criteria.replace("{major_characters}", ", ".join(major_characters))
            dim_parts.append(
                f"### {dim}\n**Focus:** {details['focus']}\n\n"
                f"**What to evaluate:**\n{criteria}\n\n"
                f"**Score anchors:**\n{details['anchors']}"
            )
        all_dimension_sections = "\n\n".join(dim_parts)

        critic_prompt = COMBINED_CRITIC_TEMPLATE.format(
            book=book_title,
            major_characters=", ".join(major_characters),
            additional_instructions=additional_instructions,
            plot_summary=circumstance["plot"]["summary"],
            scenario=scenario_str,
            character_profiles=character_profile_str,
            original_conversation=reference_str,
            all_dimension_sections=all_dimension_sections,
        )
        judge_messages = [
            {"role": "system", "content": critic_prompt},
            {"role": "user", "content": simulation_str},
        ]

        parsed = None
        try:
            async with asyncio.timeout(300):
                for attempt in range(max_retries):
                    try:
                        if structured_output:
                            parsed = await call_openai_parse(judge_messages, CoserEvalResult, model=get_judge_model('gpt-5.4-mini'), reasoning_effort=get_judge_reasoning('low'))
                            if parsed and all(validate_dim(parsed, dim) for dim in DIMENSIONS):
                                break
                            parsed = None
                        else:
                            response = await call_openai(judge_messages, model=get_judge_model('gpt-5.4-mini'), reasoning_effort=get_judge_reasoning('low'))
                            if response:
                                parsed = await extract_json_from_text(response)
                                if parsed and all(validate_dim(parsed, dim) for dim in DIMENSIONS):
                                    break
                                parsed = None
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"LLM judge attempt {attempt + 1} failed: {e}")
        except asyncio.TimeoutError:
            logging.getLogger(__name__).warning("LLM judge timed out after 300s")

        for dim in DIMENSIONS:
            dim_score, reasoning = score_dim(parsed, dim)
            eval_result[dim] = {"score": dim_score, "reasoning": reasoning}

    else:
        # Per-dimension mode: one API call per dimension
        for dim in DIMENSIONS:
            details = DIMENSION_DETAILS[dim]
            criteria = details["criteria"]
            if "{major_characters}" in criteria:
                criteria = criteria.replace("{major_characters}", ", ".join(major_characters))

            critic_prompt = CRITIC_TEMPLATE.format(
                book=book_title,
                major_characters=", ".join(major_characters),
                dimension_name=dim,
                dimension_focus=details["focus"],
                additional_instructions=additional_instructions,
                plot_summary=circumstance["plot"]["summary"],
                scenario=scenario_str,
                character_profiles=character_profile_str,
                original_conversation=reference_str,
                dimension_criteria=criteria,
                dimension_anchors=details["anchors"],
            )
            judge_messages = [
                {"role": "system", "content": critic_prompt},
                {"role": "user", "content": simulation_str},
            ]

            parsed = None
            for attempt in range(max_retries + 1):
                response = await call_openai(judge_messages)
                if response:
                    parsed = await extract_json_from_text(response)
                    if parsed and validate_dim(parsed, dim):
                        break
                    parsed = None
                if attempt >= max_retries:
                    break

            dim_score, dim_reasoning = score_dim(parsed, dim)
            eval_result[dim] = {"score": dim_score, "reasoning": dim_reasoning}

    # BLEU and ROUGE-L metrics
    bleu, rouge_l = calculate_bleu_rouge(reference, sim_clean, continue_from=continue_from)
    eval_result["bleu"] = bleu
    eval_result["rouge_l"] = rouge_l

    # Overall reward: average of dimension scores, normalized to 0-1
    avg_score = sum(eval_result[d]["score"] for d in DIMENSIONS) / len(DIMENSIONS)
    reward = avg_score / 100.0

    return reward, eval_result, parsed


# ---------------------------------------------------------------------------
# agent_loop: Harmony interface
# ---------------------------------------------------------------------------

async def agent_loop(data, context):
    """
    CoSER: Simulate and evaluate a single GCA circumstance.

    Args:
        data: {
            "circumstance": dict - a single test case from CoSER dataset with keys:
                book, plot, i_c, character_profiles, dialogues,
                speaking_characters_w_env, major_characters, key_characters, scenario
            "env_model": str (optional, defaults to context["model"])
            "nsp_model": str (optional, defaults to context["model"])
            "judge_model": str (optional, defaults to context["model"])
            "wo_thought": bool (optional, default False)
            "continue_from": int (optional, default 0) - use ground truth for first N rounds
        }
        context: {
            "client": AsyncOpenAI instance,
            "model": str - actor model name
        }

    Returns:
        {
            "reward": float (0-1, average of 4 dimension scores),
            "chat": list of conversation messages,
            "scores": dict of per-dimension scores,
            "bleu": float,
            "rouge_l": float,
            "eval_detail": dict,
            "book": str,
            "case_id": str,
        }
    """
    item = data['extra_info']
    circumstance = item["circumstance"]
    if isinstance(circumstance, str):
        circumstance = json.loads(circumstance)
    wo_thought = item.get("wo_thought", False)
    continue_from = item.get("continue_from", 0)

    # Phase 1: Simulation
    book_title = circumstance["book"]
    plot = circumstance["plot"]
    conversation = circumstance

    # Identify characters
    plot_characters = [c["name"] for c in plot["key_characters"]]
    speaking_characters_w_env = conversation["speaking_characters_w_env"].copy()
    if ENVIRONMENT not in speaking_characters_w_env:
        speaking_characters_w_env.append(ENVIRONMENT)
    major_characters = conversation["major_characters"]
    character_profiles = circumstance["character_profiles"]

    # Build enhanced character profiles
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

    # Create agents — NSP & ENVIRONMENT use OpenAI (SimpleAgent); characters use the RL model (Agent)
    init_msg = {'role': 'user', 'content': '===Conversation Start===\n\n'}
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
            system_prompt = get_character_prompt(
                book_title, character,
                involved_character_profiles.get(character, ""),
                plot["summary"], conversation["scenario"], motivation,
                thoughtless=wo_thought,
                other_character_profiles=involved_character_profiles,
            )
            system_prompt = system_prompt.strip('\n')
            system_prompt += (
                '\n\nSpeak concisely as humans, instead of being verbose. '
                'Limit your response to 60 words.\n\nAlways respond in English.'
            )
            agent = Agent(
                context.llm_client,
                [{'role': 'system', 'content': system_prompt}, init_msg],
                context.tokenizer, context.config, prompt_turn=2,
            )
        character_agents[character] = agent

    # Run conversation
    dialogues = conversation.get("dialogues", [])
    max_rounds = 10
    agent_conversations = []
    current_speaker = speaking_characters_w_env[0]
    generated_speakers = set()

    for i_round in range(max_rounds):
        if current_speaker == "<END CHAT>":
            break

        # --- Speaker turn ---
        speaker_agent = character_agents[current_speaker]
        if i_round < continue_from:
            response = dialogues[i_round]["message"] if i_round < len(dialogues) else ""
            speaker_agent.append({'role': 'assistant', 'content': response})
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
                agent.append({'role': 'user', 'content': broadcast})

        # --- NSP turn: determine next speaker ---
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
            valid = True
        elif nsp_parsed in speaking_characters_w_env and nsp_parsed != current_speaker:
            current_speaker = nsp_parsed
            valid = True
        else:
            candidates = (
                (set(major_characters + [ENVIRONMENT]) - {current_speaker})
                or set(speaking_characters_w_env) - {current_speaker}
            )
            current_speaker = random.choice(list(candidates))
            valid = False
        agent_conversations.append({"role": NSP, "content": nsp_parsed})
        if not valid:
            # step() appended nsp_raw; fix to nsp_parsed and re-inject system prompt
            nsp_agent.messages[-1]["content"] = nsp_parsed
            nsp_agent.append({'role': 'user', 'content': nsp_agent.messages[0]["content"]})

    # Phase 2: Evaluation
    import time
    _eval_start = time.monotonic()
    reward, eval_result, parsed_judge = await evaluate_simulation(
        agent_conversations, circumstance, involved_character_profiles,
        continue_from=continue_from,
        structured_output=True,
    )
    eval_time = time.monotonic() - _eval_start

    extra_info = {
        "coser/reward": reward,
        "coser/bleu": eval_result.get("bleu", 0.0),
        "coser/rouge_l": eval_result.get("rouge_l", 0.0),
        "coser/eval_time": eval_time,
        **{f"coser/{dim}": eval_result[dim]["score"] for dim in DIMENSIONS if dim in eval_result},
    }
    extra_info['all/score'] = reward
    extra_info['all/score_v1'] = reward
    outputs = []
    for name in generated_speakers:
        if character_agents[name].think_format_correct() == 0 and context.is_train:
            use_reward = reward / 2
        else:
            use_reward = reward
        out = await character_agents[name].get_agent_output(use_reward, extra_info=extra_info)
        if any(out.response_mask):
            outputs.append(out)
    # outputs = outputs[:1]
    # ===========================================================================
    # Hint + second attempt
    # ===========================================================================
    hint = None
    # Build student conversation string for hint generation
    student_conv_str = "\n\n".join(
        f"{m['role']}: {m['content']}"
        for m in agent_conversations if m["role"] != NSP
    )
    # hint only for low-score rollout
    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and reward <= 0.2
            and outputs and context.global_step < 30):
        from agents.coser.hint import generate_hint
        hint = await generate_hint(
            circumstance=circumstance,
            involved_character_profiles=involved_character_profiles,
            eval_result=eval_result,
            parsed_judge=parsed_judge,
            student_conversation=student_conv_str,
        )
        if hint:
            if parsed_judge is None:
                parsed_judge = {}
            parsed_judge['hint'] = hint

    if (getattr(context.config.algorithm, 'agent_version', None) == 'copy'
            and context.is_train
            and hint
            and outputs):
        from agents.coser.hint_agent import agent_loop as hint_agent_loop
        data['extra_info']['hint'] = hint
        data['extra_info']['old_reward'] = reward

        hint_outputs = await hint_agent_loop(data, context)

        if hint_outputs:
            # Pair each hint output with a copy carrying the original prompt_ids
            extra_outputs = []
            for base_out, hint_out in zip(outputs, hint_outputs):
                copy_out = copy.deepcopy(hint_out)
                copy_out.prompt_ids = copy.deepcopy(base_out.prompt_ids)
                copy_out.extra_fields["gen_uid"] = str(uuid.uuid4())
                hint_out.extra_fields["agent_role"] = 'hint_agent'
                extra_outputs.extend([copy_out])
            # outputs = outputs + extra_outputs[:2]
    # ===========================================================================

    if outputs:
        first_agent = character_agents[next(iter(generated_speakers))]
        await process_post_chat(data, context, first_agent.chat, outputs[0], format_think=True, extra=parsed_judge)
    return outputs
