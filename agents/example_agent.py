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

import json


async def agent_loop(data, context):
    # Data is a dict that contains all data needed for the current instance.
    chat = data["chat"]
    client = context["client"]  # Client used for the agent/model.

    for _ in range(20):
        # Keep chat append-only (i.e., no slices like chat[-10:], and no history edits like chat[2] = "xx").
        response = client.responses.create(input=chat)
        chat.append(response)
        # Optionally call a tool to interact with the environment.
        # Make sure the env is a separate server that can be called over HTTP.
        # obs = call_tool(response)
        # chat.append(obs)

    reward = int(chat)  # Some way to calculate the reward.

    # Return chat, reward, and anything else you want.
    return {
        "reward": reward,
        "chat": chat,
    }

    # For multi-agent case, return a list of dicts.
    # return [
    #     {
    #         "reward": reward,
    #         "chat": chat,
    #     },
    # ]


async def agent_loop_streaming(data, context):
    # Example loop that supports streaming, better for human evaluation.
    # `chat` can be a partial history (a checkpoint) that this function continues rolling out.
    chat = data["chat"]
    meta_info = data.get("meta_info", [])
    client = context["client"]

    # Optional: display something on the side canvas.
    yield "<|canvas|>anything you want to display on right canvas<|/canvas|>"

    for _ in range(20):
        response = client.responses.create(input=chat)
        chat.append(response)

        # Stream model output as incremental chunks to the frontend.
        yield response

        # Optional: tagged messages to control frontend rendering.
        yield "<|tool|>this is for env obs<|/tool|>"
        yield "<|think|>this is for folded thinking content<|/think|>"
        yield "<|highlight|>this is for highlight message<|/highlight|>"

        # If the agent wants to ask a question, stop and emit a checkpoint.
        if "ask_question" in response:
            # Emit meta info to restore from this checkpoint later (append as a row in meta_info).
            checkpoint = {"xx": "xxx"}
            meta_info.append({"info": f"state: {json.dumps(checkpoint)}"})
            yield meta_info[-1]
            return

    reward = int(chat)  # Some way to calculate the reward.
    yield {
        "reward": reward,
        "chat": chat,
        "meta_info": meta_info,
    }
    return


async def test_code():
    from openai import OpenAI

    chat = [{"role": "user", "content": "hello"}]
    results = await agent_loop({"chat": chat}, {"client": OpenAI()})
    print(results)


async def test_streaming_code():
    from openai import OpenAI

    client = OpenAI()
    chat = [{"role": "user", "content": "hello"}]

    # agent_loop_plus is an async generator, so consume it with async for.
    async for chunk in agent_loop_streaming({"chat": chat, "meta_info": []}, {"client": client}):
        print(chunk)
