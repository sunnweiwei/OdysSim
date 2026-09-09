# Copyright 2026 Individual Contributor: OdysSim Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Local TauBench HTTP runtime for OdysSim's external user simulator.

Start with ``python -m agents.tau_usi.runtime``. This service makes no LLM calls.
Use the pinned upstream TauBench revision documented in README.md.
"""

import asyncio
import copy
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from tau_bench.envs import get_env
from tau_bench.types import Action

IDLE_SECONDS = 2 * 60 * 60


class ExternalUser:
    """The rollout process owns user turns, so the environment needs no LLM."""

    def reset(self, instruction=None):
        return ""

    def step(self, content):
        raise ValueError("User turns belong to the external simulator; submit tools only")

    def get_total_cost(self):
        return 0.0


def convert_value(value, expected_type):
    """Match AgentArena's conversion of XML tool arguments to schema types."""
    if expected_type in (None, "any") or not isinstance(value, str):
        return value
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except ValueError:
            pass
    try:
        if expected_type in ("int", "integer"):
            return int(value)
        if expected_type in ("float", "number"):
            return float(value)
        if expected_type in ("bool", "boolean"):
            if value.lower() in ("true", "yes", "1"):
                return True
            if value.lower() in ("false", "no", "0"):
                return False
    except (ValueError, TypeError):
        pass
    return value


def convert_arguments(name, arguments, tools):
    result = copy.deepcopy(arguments)
    schema = next((t["function"] for t in tools if t["function"]["name"] == name), {})
    properties = schema.get("parameters", {}).get("properties", {})
    for key, value in result.items():
        spec = properties.get(key, {})
        result[key] = convert_value(value, spec.get("type"))
        items = spec.get("items", {})
        if spec.get("type") == "array" and isinstance(result[key], list) and items.get("type") == "object":
            for item in result[key]:
                if isinstance(item, dict):
                    for subkey, subspec in items.get("properties", {}).items():
                        if subkey in item:
                            item[subkey] = convert_value(item[subkey], subspec.get("type"))
    return result


class CreateRequest(BaseModel):
    env_type: Literal["tau"]
    params: str


class TaskParams(BaseModel):
    env_name: Literal["retail", "airline"]
    task_index: int = Field(ge=0)
    task_split: Literal["test"] = "test"


class RuntimeRequest(BaseModel):
    runtime_id: str
    params: str = "{}"


def parse_params(raw):
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(400, "params must be a JSON object encoded as a string") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "params must encode an object")
    return value


@dataclass
class Runtime:
    env: object
    meta_info: str
    touched: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=threading.RLock)


runtimes = {}
registry_lock = threading.RLock()


def lookup(runtime_id):
    with registry_lock:
        runtime = runtimes.get(runtime_id)
        if runtime is None or time.monotonic() - runtime.touched > IDLE_SECONDS:
            runtimes.pop(runtime_id, None)
            raise HTTPException(404, "Runtime not found or expired")
        runtime.touched = time.monotonic()
        return runtime


@asynccontextmanager
async def lifespan(app):
    async def cleanup():
        while True:
            await asyncio.sleep(60)
            with registry_lock:
                expired = [key for key, value in runtimes.items() if time.monotonic() - value.touched > IDLE_SECONDS]
                for key in expired:
                    runtimes.pop(key, None)

    task = asyncio.create_task(cleanup())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        with registry_lock:
            runtimes.clear()


app = FastAPI(title="OdysSim Tau-USI runtime", lifespan=lifespan)


@app.post("/create")
def create(request: CreateRequest):
    try:
        params = TaskParams.model_validate(parse_params(request.params))
    except ValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    limit = 115 if params.env_name == "retail" else 50
    if params.task_index >= limit:
        raise HTTPException(400, f"{params.env_name} test task_index must be less than {limit}")
    # HUMAN construction does not call input(); replace it before reset().
    env = get_env(
        env_name=params.env_name,
        task_index=params.task_index,
        task_split=params.task_split,
        user_strategy="human",
        user_model="unused",
    )
    env.user = ExternalUser()
    env.reset(task_index=params.task_index)
    meta = json.dumps(
        {
            **params.model_dump(),
            "initial_question": "",
            "instruction": env.task.instruction,
            "wiki": env.wiki,
            "tools_info": env.tools_info,
        }
    )
    runtime_id = str(uuid.uuid4())
    with registry_lock:
        runtimes[runtime_id] = Runtime(env, meta)
    return {"runtime_id": runtime_id, "meta_info": meta}


@app.post("/ping")
def ping(request: RuntimeRequest):
    try:
        runtime = lookup(request.runtime_id)
    except HTTPException:
        return {"exists": False, "has_ping": False, "meta_info": None, "message": "Runtime not found"}
    return {"exists": True, "has_ping": False, "meta_info": runtime.meta_info, "message": "Environment exists"}


@app.post("/step")
def step(request: RuntimeRequest):
    runtime = lookup(request.runtime_id)
    params = parse_params(request.params)
    name, arguments = params.get("name"), params.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise HTTPException(400, "step requires name and an arguments object")
    with runtime.lock:
        if name not in runtime.env.tools_map:
            raise HTTPException(400, f"Unknown tool: {name}")
        arguments = convert_arguments(name, arguments, runtime.env.tools_info)
        result = runtime.env.step(Action(name=name, kwargs=arguments)).observation
    return {"result": result if isinstance(result, str) else json.dumps(result)}


@app.post("/reward")
def reward(request: RuntimeRequest):
    runtime = lookup(request.runtime_id)
    params = parse_params(request.params)
    conversation = params.get("conversation") or []
    if not isinstance(conversation, list) or any(not isinstance(m, dict) for m in conversation):
        raise HTTPException(400, "conversation must be a list of messages")
    with runtime.lock:
        # Upstream calculates ground truth by mutating the env. Score a copy so
        # repeated reward requests cannot turn a failed task into a success.
        env = copy.deepcopy(runtime.env)
        for message in conversation:
            if message.get("role") == "assistant" and message.get("content"):
                env.actions.append(Action(name="respond", kwargs={"content": message["content"]}))
        value = env.calculate_reward().reward
    return {"reward": value}


@app.post("/stop")
def stop(request: RuntimeRequest):
    with registry_lock:
        success = runtimes.pop(request.runtime_id, None) is not None
    return {"success": success, "message": "Stopped" if success else "Runtime not found"}


@app.get("/health")
def health():
    with registry_lock:
        return {"status": "healthy", "active_environments": len(runtimes)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8005)
