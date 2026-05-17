import os
import time
import copy
import uuid
import logging
from unittest.mock import patch
from itertools import groupby
import re, unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
import asyncio
import json
import torch
from pydantic import BaseModel
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from typing import Any, Optional
import asyncio
from openai import AsyncOpenAI

async def post_chat(messages, folder, title):
    # Placeholder: original implementation posted validation rollouts to an
    # internal chat-viewer service for inspection. No-op by default; wire up
    # your own logging here if you want to mirror that workflow.
    return

async def process_post_chat(data, context, chat, output, format_think=False, extra=None):
    if not context.is_train:
        import datetime
        step = context.global_step
        experiment_name = context.config.trainer.experiment_name
        now = datetime.datetime.now()
        date_tag = f"{now.month}.{now.day}.{now.hour}"
        index = data['extra_info']['index']
        data_source = data.get("data_source", "unknown")
        extra_info = output.extra_fields.get("reward_extra_info") or {}
        result_rows = "\n".join(f"| {k} | {v} |" for k, v in extra_info.items())

        if not extra:
            extra = ""
        elif isinstance(extra, dict):
            extra = json.dumps(extra, indent=4)
        else:
            extra = str(extra)

        canvas = (
            f"<|canvas|>"
            f"## System Prompt\n"
            f"{chat[0]['content']}\n"
            f"## Meta\n"
            f"| Key | Value |\n|---|---|\n"
            f"| data | {data_source} |\n"
            f"| index | {index} |\n"
            f"| step | {step} |\n"
            f"| date | {date_tag} |\n"
            f"## Result\n"
            f"| Key | Value |\n|---|---|\n"
            f"{result_rows}\n"
            f"## Extra\n"
            f"{extra}"
            f"<|/canvas|>"
        )
        reward = output.reward_score
        messages = chat[1:]
        if format_think:
            processed = []
            for m in messages:
                if m["role"] == "assistant":
                    think, content = split_think(m["content"])
                    new_content = f"<|think|>{think}<|/think|>{content}" if think else content
                    processed.append({**m, "content": new_content})
                else:
                    processed.append(m)
            messages = processed
        first_assistant = next((m for m in messages if m['role'] == 'assistant'), None)
        if first_assistant:
            first_assistant['content'] += canvas
        await post_chat(messages, folder=f"{experiment_name}-{step}-({date_tag})", title=f"val-{data_source}-{index}-({reward})")

_openai_client: AsyncOpenAI | None = None
_fallback_client: AsyncOpenAI | None = None
_gpt54_client: AsyncOpenAI | None = None


OPENAI_TIMEOUT = 120  # seconds per API call

def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", None), timeout=OPENAI_TIMEOUT)
    return _openai_client


def _get_fallback_client() -> AsyncOpenAI:
    """Fallback to official OpenAI API (e.g. when Azure content filter blocks the request)."""
    global _fallback_client
    if _fallback_client is None:
        _fallback_client = AsyncOpenAI(api_key=os.getenv("FALLBACK_OPENAI_API_KEY"), base_url="https://api.openai.com/v1", timeout=OPENAI_TIMEOUT)
    return _fallback_client


def _get_gpt54_client() -> AsyncOpenAI:
    """Optional secondary client (e.g. an Azure deployment for 'gpt-5.4').
    Configure via GPT54_OPENAI_API_KEY and GPT54_OPENAI_BASE_URL."""
    global _gpt54_client
    if _gpt54_client is None:
        _gpt54_client = AsyncOpenAI(
            api_key=os.getenv("GPT54_OPENAI_API_KEY", ""),
            base_url=os.getenv("GPT54_OPENAI_BASE_URL", None),
            timeout=OPENAI_TIMEOUT,
        )
    return _gpt54_client


def _is_content_filter_error(e: Exception) -> bool:
    return "content_filter" in str(e)


def _is_server_error(e: Exception) -> bool:
    return "Error code: 500" in str(e) or "server_error" in str(e)


def _is_json_truncation_error(e: Exception) -> bool:
    msg = str(e)
    return "json_invalid" in msg or "EOF while parsing" in msg


def _resolve_client_and_model(model: str, use_fallback: bool = False) -> tuple[AsyncOpenAI, str]:
    if use_fallback:
        return _get_fallback_client(), model
    if model == 'gpt-5.4':
        return _get_gpt54_client(), 'gpt-5.4-2'
    return _get_openai_client(), model


def get_judge_model(default: str) -> str:
    """Return JUDGE_MODEL_NAME env override if set, else the default."""
    return os.environ.get("JUDGE_MODEL_NAME") or default


def get_judge_reasoning(default):
    """Return JUDGE_MODEL_REASONING env override if set, else the default."""
    val = os.environ.get("JUDGE_MODEL_REASONING")
    return val if val else default


async def call_openai(messages, model='gpt-5-nano', max_retries=3, response_format=None, reasoning_effort='low'):
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]
    use_fallback = True
    for attempt in range(max_retries):
        client, cur_model = _resolve_client_and_model(model, use_fallback)
        try:
            kwargs = dict(model=cur_model, messages=messages)
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = await client.chat.completions.create(**kwargs)
            if not response.choices[-1].message.content:
                continue
            return response.choices[-1].message.content or ''
        except Exception as e:
            if (_is_content_filter_error(e) or _is_server_error(e)) and not use_fallback:
                print(f"<<<USE FALLBACK>>> ({e})")
                use_fallback = True
                continue
            if attempt == max_retries - 1:
                print(f"[CALL OPENAI] Error after {max_retries} attempts: {str(e)}")
                return f"Error after {max_retries} attempts: {str(e)}"
            await asyncio.sleep(1 * (attempt + 1))
    return ""


async def call_openai_parse(messages, text_format, model='gpt-5-nano', max_retries=3, reasoning_effort='low', **kwargs):
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]
    if 'reasoning' not in kwargs and reasoning_effort is not None:
        kwargs['reasoning'] = {"effort": reasoning_effort}
    use_fallback = True
    parse_failures = 0
    for attempt in range(max_retries + 1):
        client, cur_model = _resolve_client_and_model(model, use_fallback)
        try:
            response = await client.responses.parse(
                model=cur_model, input=messages, text_format=text_format, **kwargs,
            )
            if response.output_parsed is not None:
                return response.output_parsed.model_dump(by_alias=True)
            parse_failures += 1
            if parse_failures >= 2 and not use_fallback:
                print("<<<USE FALLBACK>>> (output_parsed is None)")
                use_fallback = True
                continue
        except Exception as e:
            logging.getLogger(__name__).warning(f"[call_openai_parse] attempt {attempt + 1} failed: {e}")
            if (_is_content_filter_error(e) or _is_json_truncation_error(e)) and not use_fallback:
                parse_failures += 1
                if parse_failures >= 2:
                    print(f"<<<USE FALLBACK>>> ({e})")
                    use_fallback = True
                    continue
        if attempt >= max_retries:
            break
        await asyncio.sleep(1)
    return None


async def editlens_score(text: str, base_url: str | None = None, timeout: float = 10.0, max_retries: int = 2) -> float | None:
    # base_url should point to an OpenAI-compatible server hosting an "editlens-llama" model.
    # Configure via EDITLENS_BASE_URL or pass explicitly. Returns None if not configured.
    base_url = base_url or os.getenv("EDITLENS_BASE_URL")
    if not base_url:
        return None
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=0)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            completion = await client.chat.completions.create(
                model="editlens-llama",
                max_tokens=20,
                temperature=0.0,
                messages=[{"role": "user", "content": text}],
            )
            return float(json.loads(completion.choices[0].message.content)["score_pred"])
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    logging.getLogger(__name__).warning(f"[editlens_score] API call failed after {max_retries + 1} attempts, returning None: {last_err}")
    return None


def remove_think(text: str, remove_unclosed: bool = False) -> str:
    """Remove thinking blocks from model response. Supports <think>...</think> and <seed:think>...</seed:think>."""
    if not text:
        return text
    text = re.sub(r'<seed:think>.*?</seed:think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if remove_unclosed:
        text = re.sub(r'<seed:think>.*$', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def split_think(text: str) -> tuple[str, str]:
    if not text:
        return '', ''
    think_parts = []
    for pattern in (r'<seed:think>(.*?)</seed:think>', r'<think>(.*?)</think>'):
        for m in re.finditer(pattern, text, flags=re.DOTALL):
            think_parts.append(m.group(1).strip())
    content = remove_think(text)
    return '\n'.join(think_parts), content


def decode_conversation(input_ids: list[int], tokenizer) -> tuple[list[dict[str, str]], str]:
    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)
    pattern = re.compile(
        re.escape(tokenizer.bos_token)
        + r'(system|user|assistant|tool)\n'
        + r'(.*?)'
        + r'(?=' + re.escape(tokenizer.eos_token) + r')',
        re.DOTALL,
    )
    matches = pattern.findall(decoded_str)
    conversation = [{'role': role, 'content': content} for role, content in matches]
    return conversation, decoded_str

def truncate_text(text: str, n_words: int) -> str:
    if not text:
        return text
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= n_words:
        return text
    end = matches[n_words - 1].end()
    return text[:end] + " [...]"


def truncate_text_left(text: str, n_words: int) -> str:
    """Keep the last n_words words (drop from the left). For conversation history."""
    if not text:
        return text
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= n_words:
        return text
    start = matches[-n_words].start()
    return "[...] " + text[start:]


def truncate_turns_for_reference(
    turns: list[dict],
    content_key: str = "content",
    max_words_per_turn: int = 500,
) -> list[dict]:
    """Truncate turn contents for use as references in teacher prompts."""
    return [
        {**t, content_key: truncate_text(t.get(content_key, ""), max_words_per_turn)}
        for t in turns
    ]


class LLMClass:
    async def create_completion(self, input_ids, **kwargs):
        raise NotImplemented

class CallLLM(LLMClass):  # Call LLM in Verl RL env
    def __init__(self, url, tokenizer, config, loop, **kwargs):
        self.server_manager = url
        self.tokenizer = tokenizer
        self.config = config
        self.loop = loop

    async def _create_completion(self, input_ids, **kwargs):
        from uuid import uuid4

        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_len = min(max_len, self.config.prompt_length + self.config.response_length)
        max_new_tokens = max_len - len(input_ids)
        # This is used to avoid repetitive generation.
        if 'max_new_tokens' in kwargs:
            max_new_tokens = min(max_new_tokens, kwargs['max_new_tokens'])

        if max_new_tokens < 10:
            print(f"[DEBUG] max_new_tokens {max_new_tokens}, skip rollout")
            return None

        uid = kwargs.pop('uid', None) or uuid4().hex

        sampling_params = kwargs.pop('sampling_params', None) or {}
        sampling_params = {
            'temperature': sampling_params.get('temperature', getattr(self.config, 'temperature', 1.0)),
            'top_p': sampling_params.get('top_p', getattr(self.config, 'top_p', 1.0)),
            'top_k': sampling_params.get('top_k', getattr(self.config, 'top_k', -1)),
            'repetition_penalty': sampling_params.get('repetition_penalty', getattr(self.config, 'repetition_penalty', 1.0)),
            'logprobs': sampling_params.get('logprobs', getattr(self.config, 'calculate_log_probs', True)),
            'max_tokens': max_new_tokens,
        }

        output = await self.server_manager.generate(
            request_id=uid,
            prompt_ids=input_ids,
            sampling_params=sampling_params,
            image_data=None,
        )

        if output is None or len(output.token_ids) == 0:
            return None

        response_text = await self.loop.run_in_executor( None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True))

        return {
            "choices": [{
                "message": {
                    "content": response_text,
                    "raw_output_ids": output.token_ids,
                    "response_log_probs": output.log_probs if hasattr(output, 'log_probs') else [0.0] * len(
                        output.token_ids),
                    "routed_experts": output.routed_experts if hasattr(output, 'routed_experts') else None,
                    "extra_data": {"input_ids": input_ids},
                    "metrics": {}
                }
            }]
        }

    async def create_completion(self, input_ids, **kwargs):
        completion = await self._create_completion(input_ids, **kwargs)
        return completion


class CallAPI(LLMClass):  # Call external API (OpenAI-compatible)
    def __init__(self, url, tokenizer, config, **kwargs):
        self.tokenizer = tokenizer
        self.config = config
        self.model = url
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_AGENT_API_KEY"),
            base_url=os.getenv("OPENAI_AGENT_BASE_URL", None),
        )

    async def create_completion(self, input_ids, **kwargs):
        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_tokens = max_len - len(input_ids)

        kwargs.pop('uid', None)
        kwargs.pop('sampling_params', None)

        if 'max_new_tokens' in kwargs:
            max_tokens = min(max_tokens, kwargs.pop('max_new_tokens'))

        if max_tokens < 10:
            return None

        messages = kwargs.pop('messages', None)
        if messages is None:
            messages = decode_conversation(input_ids, self.tokenizer)[0]

        reasoning_effort = os.getenv("OPENAI_AGENT_REASONING_EFFORT", None)
        extra_kwargs = {}
        if reasoning_effort:
            extra_kwargs["reasoning_effort"] = reasoning_effort

        # Qwen3.6-Plus on Together is streaming-only; force stream and aggregate.
        needs_stream = "Qwen3.6-Plus" in self.model

        for attempt in range(10):
            try:
                if needs_stream:
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                        **extra_kwargs,
                    )
                    text = ""
                    usage = None
                    async for chunk in stream:
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                text += delta.content

                    class _U:
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                    response = type("R", (), {"usage": _U()})()
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        **extra_kwargs,
                    )
                    text = response.choices[0].message.content or ""

                text_ids = self.tokenizer.encode(text, add_special_tokens=False)

                return {
                    "choices": [{
                        "message": {
                            "content": text,
                            "raw_output_ids": text_ids,
                            "response_log_probs": [0.0] * len(text_ids),
                            "routed_experts": None,
                            "extra_data": {"input_ids": input_ids},
                            "metrics": {"usage": {
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else len(text_ids),
                                "total_tokens": response.usage.total_tokens if response.usage else 0,
                            }},
                        }
                    }]
                }
            except Exception as e:
                if attempt == 9:
                    print(f"[CallAPI ERROR] Failed after 10 attempts: {e}")
                    return None
                wait_time = min(2 ** attempt, 60)
                print(f"[CallAPI] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        return None


def truncate_prompt(chat, prompt_length, tokenizer, prompt_turn, apply_chat_template=None):
    _apply = apply_chat_template or tokenizer.apply_chat_template
    exceed_len = len(_apply(chat[:prompt_turn], tokenize=True)) + 8 - prompt_length
    _cut_idx = 0
    while exceed_len > 0:  # truncate long user prompt
        print('[PROMPT] now exceed', exceed_len, 'work on cut turn', _cut_idx)
        chat[_cut_idx]['content'] = tokenizer.decode(
            tokenizer.encode(chat[_cut_idx]['content'], add_special_tokens=False)[
                exceed_len + 4:], add_special_tokens=False)
        exceed_len = len(_apply(chat[:prompt_turn], tokenize=True)) + 8 - prompt_length
        _cut_idx = _cut_idx + 1
        if _cut_idx >= prompt_turn:
            break
    return chat


class AgentContext:
    # Manage context of an agent
    def __init__(self, chat, tokenizer, config, prompt_turn=2, enable_think=None, reserve_length=0):
        self.tokenizer = tokenizer
        self.config = config
        self.init_len = len(chat)
        self.prompt_turn = prompt_turn
        self.reserve_length = reserve_length

        if enable_think is None:
            enable_think = os.getenv("TURNOFF_THINK", "1").lower() in ("0", "false", "no")
        self.enable_think = enable_think
        _orig = tokenizer.apply_chat_template
        if enable_think:
            self._apply_chat_template = _orig
        else:
            def _apply_chat_template(*args, **kwargs):
                for extra in [{"thinking_budget": 0}, {"enable_thinking": False}, {}]:
                    try:
                        return _orig(*args, **{**extra, **kwargs})
                    except TypeError:
                        continue
            self._apply_chat_template = _apply_chat_template

        # Support both inference and training config styles
        if hasattr(config, 'actor_rollout_ref'):
            # Training config (VERL)
            self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
            self.response_length = config.actor_rollout_ref.rollout.response_length
        else:
            # Inference config
            self.prompt_length = config.prompt_length
            self.response_length = config.response_length

        self.context_uid = str(uuid.uuid4())

        self.chat = copy.deepcopy([turn for turn in chat])
        self.chat = truncate_prompt(self.chat, self.prompt_length, tokenizer, prompt_turn, self._apply_chat_template)
        self.chat_completions = [None for _ in range(len(self.chat))]
        self.chat_ids = [self.get_turn_context(i) for i in range(len(self.chat))]
        self.log_probs = [[0.0] * len(turn) for turn in self.chat_ids]
        self.token_mask = [[False] * len(turn) for turn in self.chat_ids]
        self.additional_info = [None for _ in self.chat_ids]
        self.generation_prompt = None
        self.metrics = None
        self.prompt_ids_len = len(sum(self.chat_ids[:prompt_turn], []))

    def get_turn_context(self, i):
        tokens = self._apply_chat_template(self.chat[:i + 1], add_generation_prompt=False, tokenize=True)
        prev = self._apply_chat_template(self.chat[:i], add_generation_prompt=False,
                                         tokenize=True) if i > 0 else []
        turn_tokens = tokens[len(prev):]
        return turn_tokens

    def get_generation_prompt(self):
        if self.generation_prompt is None:
            tokens = self._apply_chat_template(self.chat, add_generation_prompt=False, tokenize=True)
            add_tokens = self._apply_chat_template(self.chat, add_generation_prompt=True, tokenize=True)
            self.generation_prompt = add_tokens[len(tokens):]
        return self.generation_prompt

    def messages(self):
        return self.chat

    def context_ids(self, messages=None):
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def context(self, turn_cut: int=None):
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def append(self, turn, completion=None, additional_info=None):
        self.chat.append(turn)
        self.chat_completions.append(completion)
        self.additional_info.append(additional_info)
        if completion is None:
            self.chat_ids.append(self.get_turn_context(len(self.chat) - 1))
            self.log_probs.append([0.0] * len(self.chat_ids[-1]))
            self.token_mask.append([False] * len(self.chat_ids[-1]))
        else:
            completion_tokens = completion["choices"][0]["message"]["raw_output_ids"]
            completion_log_probs = completion["choices"][0]["message"]["response_log_probs"] or [0.0] * len(completion_tokens)
            self.chat_ids.append(self.get_generation_prompt() + completion_tokens)
            self.log_probs.append([0.0] * len(self.get_generation_prompt()) + completion_log_probs)
            self.token_mask.append([False] * len(self.get_generation_prompt()) + [True] * len(completion_tokens))
            if len(completion_tokens) == 0 or completion_tokens[-1] != self.tokenizer.eos_token_id:
                self.chat_ids[-1].append(self.tokenizer.eos_token_id)
                self.log_probs[-1].append(0.0)
                self.token_mask[-1].append(False)

    def fork(self):
        """Copy this context resetting all token masks and log probs to zero.
        All existing turns become pure context (no training signal). Useful for one-shot
        inference agents (e.g. hint generation) that reuse a prior agent's conversation."""
        new = object.__new__(self.__class__)
        new.tokenizer = self.tokenizer
        new.config = self.config
        new.init_len = self.init_len
        new.prompt_turn = self.prompt_turn
        new._apply_chat_template = self._apply_chat_template
        new.prompt_length = self.prompt_length
        new.response_length = self.response_length
        new.context_uid = self.context_uid
        new.chat = copy.deepcopy(self.chat)
        new.chat_completions = copy.deepcopy(self.chat_completions)
        new.chat_ids = copy.deepcopy(self.chat_ids)
        new.log_probs = [[0.0] * len(ids) for ids in self.chat_ids]
        new.token_mask = [[False] * len(ids) for ids in self.chat_ids]
        new.additional_info = copy.deepcopy(self.additional_info)
        new.generation_prompt = self.generation_prompt
        new.metrics = None
        new.enable_think = self.enable_think
        new.reserve_length = 0  # fork uses full response_length
        # Use the full current context length so step() computes max_new_tokens correctly
        new.prompt_ids_len = len(sum(new.chat_ids, []))
        return new

    def rollback(self, k=1):
        self.chat = self.chat[:-k]
        self.chat_completions = self.chat_completions[:-k]
        self.chat_ids = self.chat_ids[:-k]
        self.log_probs = self.log_probs[:-k]
        self.token_mask = self.token_mask[:-k]
        self.additional_info = self.additional_info[:-k]

    def get_metrics(self):
        if self.metrics is None:
            return {}
        return self.metrics

    async def get_data(self):
        prompt_turn = self.prompt_turn
        prompt_length = self.prompt_length
        response_length = self.response_length

        prompt_ids = sum(self.chat_ids[:prompt_turn], [])
        if len(prompt_ids) > prompt_length:
            print('[PROMPT] prompt truncated')
            prompt_ids = prompt_ids[-prompt_length:]

        response_ids = sum(self.chat_ids[prompt_turn:], [])[:response_length]
        response_logprobs = sum(self.log_probs[prompt_turn:], [])[:response_length]
        response_mask = [1 if m else 0 for turn in self.token_mask[self.prompt_turn:] for m in turn][:response_length]
        process_reward_mask = sum([[info.get('process_reward', 0) if isinstance(info, dict) else 0] * len(turn)
                                   for turn, info in zip(self.chat_ids, self.additional_info)][prompt_turn:], [])
        process_reward_mask = [p * m for p, m in zip(process_reward_mask, response_mask)][:response_length]
        return {
            'prompt_ids': prompt_ids,
            'response_ids': response_ids,
            'response_logprobs': response_logprobs,
            'response_mask': response_mask,
            'process_reward_mask': process_reward_mask,
            'num_turns': len(self.chat_ids),
            'messages': self.chat,
        }

    async def get_agent_output(self, agent_reward, extra_info=None, teacher_prompt=None, gen_uid=None, agent_role=None):  # TODO@Weiwei: agent_role=None means primary; set role for non-primary agents (e.g. judge)
        """
        Args:
            teacher_prompt: optional list of chat messages for OPD teacher context, e.g.
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                Tokenized here and stored as teacher_prompt_ids in extra_fields ->
                non_tensor_batch -> OPDActorRolloutRefWorker.compute_teacher_log_probs.
                Pass None to disable distillation for this sample.
        """
        # TODO(OPD)@Weiwei [Done] tokenize teacher_prompt chat and store as token ids
        extra_fields = {"reward_extra_info": extra_info or {}}
        if gen_uid is not None:
            extra_fields["gen_uid"] = gen_uid
        if agent_role is not None:
            extra_fields["agent_role"] = agent_role
        if teacher_prompt is not None:
            extra_fields["teacher_prompt_ids"] = self._apply_chat_template(
                teacher_prompt, add_generation_prompt=True, tokenize=True
            )

        out = await self.get_data()
        out = AgentLoopOutput(
            prompt_ids=out['prompt_ids'],
            response_ids=out['response_ids'],
            response_mask=out['response_mask'],
            response_logprobs=out['response_logprobs'],
            multi_modal_data={},
            reward_score=agent_reward,
            num_turns=out['num_turns'],
            metrics=AgentLoopMetrics(),
            extra_fields=extra_fields,
        )
        return out


class Agent(AgentContext):
    # Agent utils
    def __init__(self, llm_client, conversations, tokenizer, config, prompt_turn=2, enable_think=None, reserve_length=0):
        super().__init__(conversations, tokenizer, config, prompt_turn=prompt_turn, enable_think=enable_think, reserve_length=reserve_length)
        self.llm_client = llm_client
        self.info_cache = {}

    async def step(self, max_new_tokens=None):
        prompt = self.context()
        max_len = self.prompt_ids_len + self.response_length - self.reserve_length
        if max_new_tokens is not None:
            max_len = min(len(prompt) + max_new_tokens, 131072)
        completion = await self.llm_client.create_completion(
            prompt, uid=self.context_uid, max_len=max_len, messages=self.chat)
        if completion is None:
            return None
        response = completion["choices"][0]["message"]["content"]
        self.append({'role': 'assistant', 'content': response}, completion)
        return response

    async def react(self, run_action, max_turn=64, max_tokens=None, session_timeout=60 * 60,
                    should_continue=None, summary_prompt=None, safe_finish=None, observation_prompt=None):
        # Run react for max_turn turn
        if should_continue is None:
            should_continue = lambda st: True
        session_start_time = time.time()
        iteration = 0
        if max_tokens is not None:
            max_tokens = max_tokens - 512
        else:
            max_tokens = self.response_length - 512

        last_response = None
        response = None
        init_len = len(self.context(turn_cut=self.prompt_turn))
        while iteration < max_turn:
            if time.time() - session_start_time > session_timeout:  # TODO add session timeout
                print('[SESSION] Session Timeout')
                break
            if len(self.context()) - init_len > max_tokens:  # summary
                break

            iteration += 1
            response = await self.step()
            if response is None:
                break

            if not should_continue(response):
                last_response = response
                break
            if safe_finish is not None and safe_finish(response) is not None:
                observation = safe_finish(response)
            else:
                observation = await run_action(response)
            if observation is None:
                break
            if observation_prompt:
                observation += '\n' + observation_prompt
            self.append({'role': 'user', 'content': observation, })

        if last_response is None and summary_prompt is not None:
            if len(self.context()) - init_len > self.response_length - 1024:  # summary
                self.rollback(k=2)
            if self.chat[-1]['role'] == 'user':
                self.append({'role': 'assistant', 'content': "", })
            self.append({'role': 'user', 'content': summary_prompt, })
            last_response = await self.step(max_new_tokens=4096)
        elif last_response is None:
            last_response = str(response)

        return {'last_response': last_response, 'iteration': iteration}

    def set_process_reward(self, turn, reward):
        if isinstance(turn, str) and turn.lower() == 'all':
            turn = [i for i in range(len(self.chat))]
        if not isinstance(turn, list):
            turn = [turn]
        for i in turn:
            if i <= 0:
                continue
            if i > len(self.chat) - 1:
                continue
            if self.chat_completions[i] is None:
                continue
            if self.additional_info[i] is None:
                self.additional_info[i] = {}
            self.additional_info[i]['process_reward'] = reward

    def set_cache(self, key, value):
        self.info_cache[key] = value

    def fork(self):
        new = super().fork()
        new.llm_client = self.llm_client
        new.info_cache = {}
        return new

    def think_format_correct(self):
        for i, turn in enumerate(self.chat):
            if turn['role'] != 'assistant' or self.chat_completions[i] is None:
                continue
            content = turn.get('content', '') or ''
            starts_think = content.startswith('<think>') or content.startswith('<seed:think>')
            try:
                if self.enable_think:
                    if not starts_think:
                        return 0
                    has_close = '</think>' in content or '</seed:think>' in content
                    if not has_close:
                        return 0
                    _, after_think = split_think(content)
                    if not after_think.strip():
                        return 0
                else:
                    if starts_think:
                        return 0
            except Exception:
                return 0
        return 1



@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    is_train: bool
    tokenizer: PreTrainedTokenizer | AutoTokenizer | None = None
    llm_client: LLMClass = None



async def run_action(env, response):
    try:
        try:
            act = time.time()
            env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)
            if time.time() - act > 10:
                print('Action Cost', time.time() - act)
        except asyncio.TimeoutError:
            print('[ACTION] Action timed out after 120 seconds')
            env_return = {'observation': 'Action timed out after 120 seconds'}
        if 'action' in env_return:
            action, arguments = env_return['action'], env_return.get('arguments', {})
            if action == 'finish':
                return None
        elif env_return.get('observation', None) == 'finish':
            return None
        observation = env_return.pop('observation', 'Empty')
    except Exception as e:
        observation = f"Error: {e}"
    return observation
