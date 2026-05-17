from ..utils import  BaseEnv, RuntimeServiceError, extract_fn_call
from ..tool_prompt import convert_tools_to_description, TOOL_PROMPT
from openai import AsyncOpenAI
import asyncio
import os


class TauEnv(BaseEnv):
    """Tau environment client with automatic recovery via conversation replay."""
    def __init__(self, env_name, task_index):
        super().__init__(env_name=env_name, task_index=task_index)

    def get_system_prompt(self):
        """Get formatted system prompt from environment meta_info."""
        if not self.meta_info:
            ping_result = self.ping()
            if ping_result['exists']:
                self.meta_info = ping_result['meta_info']

        if not self.meta_info:
            return ""

        tools_info = self.meta_info.get('tools_info', [])
        wiki = self.meta_info.get('wiki', '')
        tool_description = TOOL_PROMPT.format(description=convert_tools_to_description(tools_info))
        return wiki + '\n\n' + tool_description


async def agent_loop(data, context):
    task_index = data["task_index"]
    env_name = data["env_name"]
    llm_client = context["client"]
    model = context["model"]

    tau_env = TauEnv(env_name=env_name, task_index=task_index)
    try:
        await asyncio.to_thread(tau_env.initialize)
    except RuntimeServiceError as e:
        raise f"⚠️ **Runtime Service Unavailable**\n\nThe tau-bench environment service is temporarily unavailable. Please try again in a moment.\n\nError: {e}"

    system_prompt = tau_env.get_system_prompt()
    chat = [{'role': 'system', 'content': system_prompt}]

    user_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_system_prompt = f"""{tau_env.meta_info['instruction']}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisfied, generate '###STOP###' as a standalone message without anything else to end the conversation.
- If transferring to a human, after the agent confirms the transfer is successful, you must generate '###STOP###' immediately to end the conversation with the agent.- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""
    user_history = [
        {"role": "system", "content": user_system_prompt, },
        {"role": "user", "content": "Hi! How can I help you today?"},
    ]

    user_response = tau_env.meta_info['initial_question']
    user_history.append({'role': 'assistant', 'content': user_response})
    chat.append({'role': 'user', 'content': user_response})
    print('User:', user_response)

    for iteration in range(100):
        # Check for cancellation before each API call
        response = await llm_client.responses.create(model=model, input=chat)
        content = response.output[-1].content[0].text

        chat.append({'role': 'assistant', 'content': content})
        fn_call = extract_fn_call(content)
        observation = None

        # Check if extract_fn_call returned a format error
        if fn_call is not None and isinstance(fn_call, dict) and 'error' in fn_call:
            observation = fn_call['error']
            chat.append({'role': 'system', 'content': observation})
            continue  # Let agent try again with correct format

        if fn_call is not None and isinstance(fn_call, list) and len(fn_call) > 0:
            try:
                observation = ""
                for fn in fn_call:
                    observation += '\n\n' if len(observation) > 0 else ''
                    observation += await asyncio.to_thread(tau_env.step, fn, conversation=chat)
            except RuntimeServiceError as e:
                raise f"\n\n⚠️ **Runtime Service Unavailable**\n\nCould not execute tool call. The environment service is temporarily unavailable. Please try again.\n\nError: {e}"
            except Exception as e:
                observation = f"Error executing tool: {e}"

        if observation is None:
            # Ask question, user simulator respond
            print('AI:', ' '.join(content.split()))
            user_history.append({'role': 'user', 'content': content})
            user_response = await user_client.responses.create(model='gpt-5-mini', input=user_history)
            user_response = user_response.output[-1].content[0].text
            if '###STOP###' in user_response:
                break
            user_history.append({'role': 'assistant', 'content': user_response})
            chat.append({'role': 'user', 'content': user_response})
            print('User:', user_response)
            continue
        else:
            chat.append({'role': 'system', 'content': observation})
    reward = await asyncio.to_thread(tau_env.get_reward)
    print('Reward:', reward)
    return {
        "reward": reward,
        "chat": chat,

        # add any optional field
        "user_history": user_history,
    }

