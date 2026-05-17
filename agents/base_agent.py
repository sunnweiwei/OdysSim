import logging
import os
from typing import Any, Union

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from agents.utils import CallLLM, CallAPI, TaskContext

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_agent_loop(data_source: str):
    """Map data_source name to its agent_loop function.
    Data source names match the keys registered in prepare_dataset.py.
    """
    from agents.math.math_agent import agent_loop as math_agent_loop
    from agents.fantom.agent import agent_loop as fantom_agent_loop
    from agents.hitom.agent import agent_loop as hitom_agent_loop
    from agents.lifechoices.agent import agent_loop as lifechoices_agent_loop
    from agents.mirrorbench.agent import agent_loop as mirrorbench_agent_loop
    from agents.mistakes.agent import agent_loop as mistakes_agent_loop
    from agents.mmtom.agent import agent_loop as mmtom_agent_loop
    from agents.paratomi.agent import agent_loop as paratomi_agent_loop
    from agents.userllm.agent import agent_loop as userllm_agent_loop
    from agents.coser.coser_agent import agent_loop as coser_agent_loop
    from agents.twinvoice.agent import agent_loop as twinvoice_agent_loop
    from agents.sotopia.agent import agent_loop as sotopia_agent_loop
    from agents.sotopia.hint_agent import agent_loop as sotopia_hint_agent_loop
    from agents.humanual.agent import agent_loop as humanual_agent_loop
    from agents.instruct.ifbench_agent import agent_loop as ifbench_agent_loop
    from agents.social_r1.agent import agent_loop as social_r1_agent_loop
    from agents.sim_arena.agent_doc import agent_loop as sim_arena_doc_agent_loop
    from agents.sim_arena.agent_math import agent_loop as sim_arena_math_agent_loop
    from agents.tombench.agent import agent_loop as tombench_agent_loop
    from agents.behavior_chain.agent import agent_loop as behavior_chain_agent_loop
    from agents.alignx.agent import agent_loop as alignx_agent_loop
    from agents.humanllm.agent import agent_loop as humanllm_agent_loop
    from agents.socsci210.agent import agent_loop as socsci210_agent_loop

    _ROUTES = {
        "fantom": fantom_agent_loop,
        "hitom": hitom_agent_loop,
        "lifechoices": lifechoices_agent_loop,
        "mirrorbench": mirrorbench_agent_loop,
        "mistakes": mistakes_agent_loop,
        "mmtom": mmtom_agent_loop,
        "paratomi": paratomi_agent_loop,
        "userllm": userllm_agent_loop,
        "dapo": math_agent_loop,
        "aime": math_agent_loop,
        "coser": coser_agent_loop,
        "twinvoice": twinvoice_agent_loop,
        "sotopia_hint": sotopia_hint_agent_loop,
        'sotopia': sotopia_agent_loop,
        'humanual': humanual_agent_loop,
        'humanual-book': humanual_agent_loop,
        'humanual-chat': humanual_agent_loop,
        'humanual-email': humanual_agent_loop,
        'humanual-news': humanual_agent_loop,
        'humanual-opinion': humanual_agent_loop,
        'humanual-politics': humanual_agent_loop,
        'ifbench': ifbench_agent_loop,
        'social_r1': social_r1_agent_loop,
        'sim_arena_math': sim_arena_math_agent_loop,
        'sim_arena_doc': sim_arena_doc_agent_loop,
        'tombench': tombench_agent_loop,
        'behavior_chain': behavior_chain_agent_loop,
        'alignx': alignx_agent_loop,
        'alignx_demo': alignx_agent_loop,
        'alignx_pair': alignx_agent_loop,
        'alignx_ugc': alignx_agent_loop,
        'alignx_arbitrary': alignx_agent_loop,
        'alignx_history16': alignx_agent_loop,
        'humanllm': humanllm_agent_loop,
        'socsci210': socsci210_agent_loop
    }
    for key, fn in _ROUTES.items():
        if key == data_source:
            return fn
    return math_agent_loop  # default fallback


@register("agent_hub")
class AgentHubLoop(AgentLoopBase):
    _class_initialized = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_class(config=self.config, tokenizer=self.tokenizer, processor=self.processor)

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing AgentHubLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

    async def run(
            self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        item = dict(kwargs)

        llm_client = CallLLM(
            url=self.server_manager,
            tokenizer=self.tokenizer,
            config=self.config.actor_rollout_ref.rollout,
            loop=self.loop,
        )
        context = TaskContext(
            config=self.config,
            global_step=kwargs.get('global_step', 0),
            llm_client=llm_client,
            is_train=kwargs.get('is_train', True),
            tokenizer=self.tokenizer,
        )

        data_source = item["data_source"]
        agent_loop_fn = _get_agent_loop(data_source)
        return await agent_loop_fn(item, context)


@register("openai_agent")
class OpenAIAgentLoop(AgentLoopBase):
    _class_initialized = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_class(config=self.config, tokenizer=self.tokenizer, processor=self.processor)

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing OpenAIAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

    async def run(
            self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        item = dict(kwargs)

        model_name = os.getenv("OPENAI_AGENT_MODEL", "gpt-5-nano")
        llm_client = CallAPI(
            url=model_name,
            tokenizer=self.tokenizer,
            config=self.config.actor_rollout_ref.rollout,
        )
        context = TaskContext(
            config=self.config,
            global_step=kwargs.get('global_step', 0),
            llm_client=llm_client,
            is_train=kwargs.get('is_train', True),
            tokenizer=self.tokenizer,
        )

        data_source = item["data_source"]
        agent_loop_fn = _get_agent_loop(data_source)
        return await agent_loop_fn(item, context)
