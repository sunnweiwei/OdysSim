import re
from agents.utils import Agent, process_post_chat


def extract_last_boxed(s):
    matches = re.findall(r'\\boxed\{([^}]*)\}', s)
    return matches[-1] if matches else None


async def agent_loop(data, context):
    extra_info = data["extra_info"]
    prompt = extra_info["prompt"]
    answer = extra_info["answer"]
    llm_client = context.llm_client
    tokenizer = context.tokenizer
    config = context.config
    chat = [
        {'role': 'system', 'content': "You are a math agent. Solve the problem and put the final answer in \\boxed{}."},
        {'role': 'user', 'content': prompt}]
    agent = Agent(llm_client, chat, tokenizer, config, prompt_turn=2)
    response = await agent.step()
    predict_answer = extract_last_boxed(response)

    from math_verify import parse, verify
    gold = parse(answer, parsing_timeout=None)
    parsed_predict = parse(predict_answer, parsing_timeout=None)
    acc = int(verify(gold, parsed_predict, timeout_seconds=None))

    teacher_prompt = [
        {"role": "system",
         "content": "You are a math expert. You will be given a problem and the correct final answer. Write a step-by-step solution that leads to this answer."},
        {"role": "user", "content": f"{prompt}\nReference answer: {answer}\n\nExplain the reasoning step by step."}
    ]

    output = await agent.get_agent_output(
        acc,
        extra_info={
            "all/score": acc,
            "math/acc": acc,
            "math/has_boxed": int(predict_answer is not None),
            "math/response_length": len(response.split()) if response else 0,
        },
        teacher_prompt=teacher_prompt)

    await process_post_chat(data, context, agent.chat, output)

    return output
