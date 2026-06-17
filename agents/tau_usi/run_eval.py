"""Standalone, verl-free driver for OdysSim's tau_usi rollout.

Runs :func:`agents.tau_usi.agent.rollout_one_task` (OdysSim's OWN model-as-user-
simulator rollout) one task at a time against the tau runtime service, for
EXTERNAL (API) user-sim models — **no verl, no torch, no GPU**. Writes a
``<model>_task_results.json`` consumable by ``usi_metric``.

This exists because evaluating a non-self-hosted (API) user-sim model does not
need the verl RL engine: the rollout is OpenAI calls + a tokenizer + HTTP to the
runtime. (The verl/torch coupling in ``agents/utils.py`` was incidental and is
now lazy; see that file.)

Env:
  RUNTIME_SERVICE_URL              tau runtime (e.g. http://localhost:8005)
  OPENAI_API_KEY / OPENAI_BASE_URL          agent model (call_openai, gpt-5-nano default)
  OPENAI_AGENT_API_KEY / OPENAI_AGENT_BASE_URL   user-sim model (CallAPI)
  TAU_USI_TOKENIZER                stand-in HF tokenizer for context bookkeeping
                                   (default Qwen/Qwen2.5-0.5B-Instruct)

Usage:
  python -m agents.tau_usi.run_eval --user-sim-model gpt-4o-mini \
      --domains retail:0-9,airline:0-4 --workers 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace

from transformers import AutoTokenizer

from agents.tau_usi.agent import rollout_one_task
from agents.utils import CallAPI

TOKENIZER_ID = os.getenv("TAU_USI_TOKENIZER", "Qwen/Qwen2.5-0.5B-Instruct")


class EvalContext:
    """Minimal stand-in for verl's TaskContext — only what rollout_one_task reads."""

    def __init__(self, llm_client, tokenizer, config):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = False
        self.global_step = 0

    def get(self, key, default=None):
        return default


def parse_domains(spec: str):
    """'retail:0-9,airline:0-4' -> [('retail',0), ..., ('airline',4)]."""
    tasks = []
    for part in spec.split(","):
        dom, rng = part.split(":")
        dom = dom.strip()
        if "-" in rng:
            a, b = rng.split("-")
            idxs = range(int(a), int(b) + 1)
        else:
            idxs = [int(rng)]
        tasks.extend((dom, i) for i in idxs)
    return tasks


async def _run(args):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    config = SimpleNamespace(
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=1.0, top_p=1.0, top_k=-1,
        repetition_penalty=1.0, calculate_log_probs=False,
    )
    llm = CallAPI(url=args.user_sim_model, tokenizer=tok, config=config)
    ctx = EvalContext(llm, tok, config)

    tasks = parse_domains(args.domains)
    sem = asyncio.Semaphore(args.workers)

    async def one(dom, idx):
        data = {"env_name": dom, "task_index": idx, "instance_id": f"{dom}_{idx}"}
        async with sem:
            t0 = time.monotonic()
            try:
                rec, _ = await rollout_one_task(data, ctx)
                rec["elapsed_seconds"] = round(time.monotonic() - t0, 1)
                rec["user_sim_model"] = args.user_sim_model
                print(f"[{dom}_{idx}] reward={rec['reward']} turns={len(rec['conversation'])} "
                      f"term={rec['termination_reason']} ({rec['elapsed_seconds']}s)", flush=True)
                return rec
            except Exception as e:
                print(f"[{dom}_{idx}] ERROR: {type(e).__name__}: {e}", flush=True)
                return None

    recs = await asyncio.gather(*[one(d, i) for d, i in tasks])
    results = [r for r in recs if r is not None]

    payload = {
        "benchmark": "tau_usi",
        "model": os.getenv("OPENAI_AGENT_MODEL", "gpt-5-nano"),
        "user_sim_model": args.user_sim_model,
        "n": len(results),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    rewards = [r["reward"] for r in results]
    succ = 100.0 * sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\nWROTE {args.out}: {len(results)}/{len(tasks)} tasks, agent success={succ:.0f}%", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-sim-model", required=True, help="model id for the user simulator (CallAPI)")
    ap.add_argument("--domains", default="retail:0-9,airline:0-4", help="e.g. 'retail:0-9,airline:0-4'")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None, help="output *_task_results.json (default from model name)")
    ap.add_argument("--prompt-length", type=int, default=16000)
    ap.add_argument("--response-length", type=int, default=8000)
    args = ap.parse_args(argv)
    if not args.out:
        args.out = f"{args.user_sim_model.replace('/', '_')}_task_results.json"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
