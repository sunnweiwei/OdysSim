"""
ToMBench agent for Harmony evaluation.

Evaluates theory-of-mind reasoning on the ToMBench benchmark. Each agent loop
processes exactly one multiple-choice question and returns per-sample accuracy
as the reward.

Based on:
  "ToMBENCH: Benchmarking Theory of Mind in Large Language Models"
  (https://arxiv.org/pdf/2402.15052)

Benchmark characteristics:
  - 8 theory-of-mind tasks
  - 6 ability dimensions
  - 31 specific abilities
  - bilingual Chinese / English inventory
  - MCQ-only evaluation with accuracy as the metric

This implementation follows the paper's evaluation setup closely:
  - official vanilla / CoT prompt format
  - 2-choice and 4-choice questions
  - exact-match accuracy
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from agents.utils import Agent, process_post_chat


SYSTEM_PROMPT_ZH = '''下面给你提供一段故事，一个问题和若干答案选项，请你根据故事内容和给定的问题，按照常理推测，选择一个最可能的答案选项，并输出答案序号。
注意：
（1）请先一步步思考，对问题的答案进行推理分析，最后请输出最可能的答案序号，格式为：<answer>答案序号</answer>，例如，最可能的答案选项为”A. 手提包”，则输出”<answer>A</answer>”；
（2）请必须从给定的答案选项”A、B、C、D”中选择一个作为输出；
（3）再次强调，你必须先给出一步步推理的结果，最后再输出最可能的答案序号。你不应该直接输出答案。'''

SYSTEM_PROMPT_EN = '''Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
Note:
(1) Please first think step by step, conduct analysis on the answers to the questions, and finally output the most likely answer index in the format: <answer>Answer Index</answer>;
(2) You must choose one of the given answer options “A, B, C, D” as your answer;
(3) Again, you must first output the results of step-by-step reasoning, and finally output the most likely answer index. You should not directly output the answer index.'''


def extract_choice_letter(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def build_system_prompt(language: str) -> str:
    return SYSTEM_PROMPT_ZH if language == "zh" else SYSTEM_PROMPT_EN


def get_options_dict(row: dict) -> dict[str, str]:
    options = row.get("options_dict")
    if isinstance(options, dict):
        return {str(letter): str(text) for letter, text in options.items()}
    choices = row.get("options") or []
    return {letter: str(choice) for letter, choice in zip(["A", "B", "C", "D"], choices)}


def build_user_prompt(row: dict) -> str:
    lines = []
    if row["language"] == "zh":
        lines.append("[故事]")
        lines.append(row["story"])
        lines.append("")
        lines.append("[问题]")
        lines.append(row["question"])
        lines.append("")
        lines.append("[答案选项]")
    else:
        lines.append("[Story]")
        lines.append(row["story"])
        lines.append("")
        lines.append("[Question]")
        lines.append(row["question"])
        lines.append("")
        lines.append("[Candidate Answers]")

    options = get_options_dict(row)
    for letter in ["A", "B", "C", "D"]:
        if letter in options:
            lines.append(f"{letter}. {options[letter]}")

    return "\n".join(lines)


async def agent_loop(data: dict, context):
    row = data["extra_info"]
    system_prompt = build_system_prompt(row["language"])
    user_prompt = build_user_prompt(row)
    chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)
    response = await agent.step()
    final_prediction = extract_choice_letter(response or "")
    correct_letter = str(row.get("correct_letter", "")).strip().upper()
    reward = 1.0 if final_prediction == correct_letter and correct_letter else 0.0

    extra_info = {
        "all/score": reward,
        "tombench/reward": reward,
        "tombench/response_len": len(response.split())
    }

    output = await agent.get_agent_output(reward, extra_info=extra_info)
    await process_post_chat(data, context, agent.chat, output)
    return output
