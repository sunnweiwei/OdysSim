"""Score OdysSim's one-by-one tau_usi eval into the canonical USI table.

OdysSim's tau_usi rollout (``agent.py``) evaluates one task at a time and writes
a ``*_task_results.json`` file (``{"results": [{instance_id, conversation,
survey, reward, ...}, ...]}``). This CLI is a pure post-hoc aggregator: it
converts those per-task records into the ``eval_results`` schema and runs the
canonical :func:`usi_metric.compute_all_with_variance`, producing the *same*
D1-D4 / Eval / ECE / USI numbers AgentArena's leaderboard reports.

The rollout stays strictly one-task-at-a-time; only the *scoring* aggregates
(USI is a distribution-level metric and cannot be computed per task).

Layout of the data dir (default ``<repo>/data/tau_usi``, override with
``--data-dir`` or ``$TAU_USI_DATA_DIR``), mirrored from AgentArena's
``annotation_analysis/data``::

    tau_bench_tasks_unified.json     # 165 tasks x 3 human annotators (REQUIRED)
    survey_data/survey_comparable_*.json  # optional: for the Eval term
    eval_results/eval_results_*.json # optional: only for --difficulty baselines
                                     #           or the --print-all leaderboard

The ECE difficulty bins are shipped frozen in ``tau_usi_difficulty.json`` (a
~6 KB ``{task_key: pooled_success}`` map, the only thing the 135 MB of baseline
``eval_results`` were needed for), so the default ``--difficulty frozen`` scores
a model from just ``tau_bench_tasks_unified.json``. Re-freeze when the baselines
change with ``python -m agents.tau_usi.usi_metric --eval-results-dir <dir>``.

Usage::

    python -m agents.tau_usi.score_eval results/v6_task_results.json --label osim-8b-v6
    python -m agents.tau_usi.score_eval eval_results_osim-8b-v6.json   # already converted

Outputs (next to the input, or under ``--out-dir``):
  - ``eval_results_<label>.json``     the converted records
  - ``<label>_aggregate_metrics.json``  the USI row(s), mean+/-std per metric
and prints the leaderboard-style table to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agents.tau_usi import usi_metric as um

_METRIC_KEYS = ["D1_conv", "D2_info", "D3_clarif", "D4_react", "eval_agree", "ece", "usi"]


def _default_data_dir() -> Path:
    env = os.getenv("TAU_USI_DATA_DIR")
    if env:
        return Path(env)
    # <repo>/data/tau_usi  (this file is <repo>/agents/tau_usi/score_eval.py)
    return Path(__file__).resolve().parents[2] / "data" / "tau_usi"


def task_results_to_eval_results(payload) -> dict:
    """Convert an OdysSim ``*_task_results.json`` payload (or an already-converted
    ``eval_results`` dict) into the flat ``{instance_id: entry}`` eval schema."""
    # Already in eval_results schema: flat dict of per-task entries.
    if isinstance(payload, dict) and "results" not in payload:
        if payload and all(isinstance(v, dict) and "conversation" in v for v in payload.values()):
            return payload

    records = payload["results"] if isinstance(payload, dict) else payload
    out: dict[str, dict] = {}
    for r in records:
        iid = r.get("instance_id") or f"{r['domain']}_{r['task_index']}"
        out[iid] = {
            "instance_id": iid,
            "agent_id": r.get("agent_id", "agent-origin"),
            "conversation": r["conversation"],
            "survey": r.get("survey", {}) or {},
            "reward": float(r.get("reward", 0) or 0),
            "keep": bool(r.get("keep", True)),
        }
    return out


def emit_survey_comparable(eval_results: dict, survey_dir: Path, label: str) -> Path | None:
    """Write ``survey_comparable_<label>.json`` from the per-task surveys so a
    brand-new model also gets an Eval (survey agreement) term.

    Only base instance ids are emitted (single-run eval), so the model is scored
    against human batch 1 only -- a partial but valid eval_agree. Skipped if any
    survey is empty for every task.
    """
    comparable = {
        iid: {"instance_id": iid, "survey": entry["survey"]}
        for iid, entry in eval_results.items()
        if entry.get("survey")
    }
    if not comparable:
        return None
    survey_dir.mkdir(parents=True, exist_ok=True)
    path = survey_dir / f"survey_comparable_{label}.json"
    path.write_text(json.dumps(comparable, indent=2, ensure_ascii=False))
    return path


def _fmt_cell(pair, scale=1.0, decimals=1):
    """Format a (mean, std) cell. ``scale`` rescales (ECE is stored 0-1, shown
    x100); D1-D4 / Eval / USI are already on a 0-100 scale."""
    if not pair or pair[0] is None:
        return f"{'NA':>11s}"
    m, s = pair
    return f" {m * scale:5.{decimals}f}±{s * scale:<4.{decimals}f}"


def print_table(results, focus=None):
    hdr = "{:32s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
        "Model", "D1", "D2", "D3", "D4", "Eval", "ECE", "USI")
    print(hdr)
    print("-" * len(hdr))
    # Human first, then by USI desc
    human = [r for r in results if r["name"] == "Human (inter-ann.)"]
    rest = sorted([r for r in results if r["name"] != "Human (inter-ann.)"],
                  key=lambda r: -(r["usi"][0] if r["usi"] and r["usi"][0] is not None else -1))
    for r in human + rest:
        mark = " *" if focus and r["name"] == focus else ""
        line = f"{r['name'][:32]:32s}"
        for k in ("D1_conv", "D2_info", "D3_clarif", "D4_react"):
            line += _fmt_cell(r[k])
        line += _fmt_cell(r["eval_agree"])
        line += _fmt_cell(r["ece"], scale=100.0, decimals=2)  # ECE stored 0-1, shown x100
        line += _fmt_cell(r["usi"], decimals=2)
        print(line + mark)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="OdysSim *_task_results.json (or an eval_results_*.json dict)")
    p.add_argument("--data-dir", default=None, help="tau_usi data dir (default <repo>/data/tau_usi or $TAU_USI_DATA_DIR)")
    p.add_argument("--label", default=None, help="model tag (default: derived from filename)")
    p.add_argument("--out-dir", default=None, help="where to write converted + metrics JSON (default: input's dir)")
    p.add_argument("--difficulty", choices=["frozen", "baselines", "self"], default="frozen",
                   help="ECE difficulty reference: 'frozen' (shipped tau_usi_difficulty.json, needs no baseline "
                        "files — default), 'baselines' (recompute from local eval_results), or 'self' (paper self-pooled)")
    p.add_argument("--difficulty-json", default=None,
                   help=f"frozen difficulty map to use (default: {um.FROZEN_DIFFICULTY_PATH.name} next to usi_metric.py)")
    p.add_argument("--emit-survey-comparable", action="store_true",
                   help="derive survey_comparable_<label>.json from the eval's surveys so the model gets an Eval term (partial: human batch 1 only)")
    p.add_argument("--print-all", action="store_true", help="print the full leaderboard, not just the scored model (needs baseline eval_results)")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    unified = data_dir / "tau_bench_tasks_unified.json"
    ev_dir = data_dir / "eval_results"
    survey_dir = data_dir / "survey_data"
    if not unified.exists():
        p.error(f"missing tau_bench_tasks_unified.json under {data_dir} (sync from the AgentArena box; see module docstring)")
    # Baseline eval_results are only needed to recompute difficulty or to print
    # the leaderboard; the frozen map makes them optional otherwise.
    need_baselines = args.difficulty == "baselines" or args.print_all
    if need_baselines and not ev_dir.exists():
        p.error(f"missing eval_results/ under {data_dir}; needed for --difficulty baselines / --print-all "
                "(or use the default --difficulty frozen, which needs no baselines)")

    in_path = Path(args.input)
    label = args.label or in_path.stem.replace("_task_results", "").replace("eval_results_", "")
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_path.read_text())
    eval_results = task_results_to_eval_results(payload)
    n_kept = sum(1 for v in eval_results.values() if v.get("keep", True))
    print(f"[score_eval] {label}: {len(eval_results)} records ({n_kept} kept) from {in_path.name}")

    converted_path = out_dir / f"eval_results_{label}.json"
    converted_path.write_text(json.dumps(eval_results, ensure_ascii=False))

    if args.emit_survey_comparable:
        sp = emit_survey_comparable(eval_results, survey_dir, label)
        print(f"[score_eval] survey_comparable -> {sp}" if sp else "[score_eval] no surveys to emit")

    baselines = um.resolve_baselines(ev_dir) if ev_dir.exists() else []

    # ── ECE difficulty: frozen map (default) needs no baseline files ──
    difficulty = None
    diff_files = None
    if args.difficulty == "frozen":
        difficulty = um.load_difficulty(args.difficulty_json)
        print(f"[score_eval] difficulty: frozen map ({len(difficulty)} tasks) — no baselines needed")
    elif args.difficulty == "baselines":
        diff_files = baselines
        missing = len(um.PUBLISHED_BASELINES) - len(baselines)
        if missing:
            print(f"[score_eval] WARNING: {missing}/{len(um.PUBLISHED_BASELINES)} baselines absent; "
                  f"difficulty pools over the {len(baselines)} present (numbers may drift from frozen).")
    # else 'self': difficulty/diff_files stay None (self-pooled over llm_files)

    # Include baselines as leaderboard rows only when printing the full table.
    llm_files = ([*baselines, converted_path] if (args.print_all and baselines) else [converted_path])
    survey_arg = survey_dir if survey_dir.exists() else None

    batches = um.load_batches_from_unified(unified)
    results = um.compute_all_with_variance(
        batches, llm_files, survey_arg, difficulty_files=diff_files, difficulty=difficulty)

    by_name = {r["name"]: r for r in results}
    row = by_name.get(label)
    print()
    if args.print_all:
        print_table(results, focus=label)
    else:
        print_table([by_name["Human (inter-ann.)"], row], focus=label)
    print()

    # Write the scored model's aggregate metrics (+ Human reference).
    def _ser(r):
        return {"name": r["name"], **{k: r.get(k) for k in _METRIC_KEYS}}
    metrics = {
        "label": label,
        "difficulty_reference": args.difficulty,
        "n_baselines_used": len(baselines) if need_baselines else 0,
        "model": _ser(row) if row else None,
        "human_inter_annotator": _ser(by_name["Human (inter-ann.)"]),
    }
    metrics_path = out_dir / f"{label}_aggregate_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    usi = row["usi"][0] if row and row["usi"] else None
    print(f"[score_eval] USI={usi:.2f}" if usi is not None else "[score_eval] USI=NA")
    print(f"[score_eval] wrote {converted_path.name}, {metrics_path.name} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
