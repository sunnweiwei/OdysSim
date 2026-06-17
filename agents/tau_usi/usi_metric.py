"""Canonical tau-USI scorer for OdysSim.

A faithful, self-contained port of AgentArena's
``annotation_analysis/analyze_interaction.compute_all_with_variance`` — the
*evaluation* metric behind the tau-USI leaderboard. OdysSim's one-by-one
tau_usi rollout (``agent.py``) produces per-task records; this module
aggregates a whole set of them into the same D1-D4 / Eval / ECE / USI numbers
AgentArena reports, so OdysSim scores match.

    USI = mean(D1_conv, D2_info, D3_clarif, D4_react, [eval_agree,] (1 - ECE) * 100),
          averaged over the 3 human annotation batches.

  - D1-D4   Dice-Sorensen similarity between the model's *pooled* feature row
            and each human batch's row (``compute_dimension_dice`` / ``build_row``).
  - ECE     difficulty-binned calibration (``compute_ece_pair``). By default the
            difficulty bins are pooled over the models being scored; pass
            ``difficulty_files`` to FIX them to a reference baseline set so a
            newly-evaluated model is scored on an established yardstick.
  - Eval    ordinal survey agreement vs human annotators
            (``compute_model_eval_agreement``; requires survey data).

USI is intrinsically distribution-level, so the *rollout* stays one-task-at-a-
time (preserved in ``agent.py``) while *scoring* aggregates here.

Feature extraction is the single shared definition in
``agents.tau_usi.utils.extract_conversation_features`` — this module only does
aggregation. Ported from AgentArena ``analyze_interaction.py``
(md5 c195960d96ba6ceacad182601948a085); kept line-faithful so the numbers match.

CLI::

    # score a one-by-one eval into the USI table (default difficulty = shipped frozen map)
    python -m agents.tau_usi.usi_metric score results/v6_task_results.json --label osim-8b-v6
    # re-freeze the ECE difficulty map when the baselines change
    python -m agents.tau_usi.usi_metric freeze --eval-results-dir data/tau_usi/eval_results

Needs only ``data/tau_usi/tau_bench_tasks_unified.json`` (+ optional ``survey_data/``
for the Eval term); the ECE difficulty bins ship frozen in ``tau_usi_difficulty.json``.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from agents.tau_usi.utils import (
    DIMENSION_KEYS,
    FIELD_ORDINAL,
    extract_conversation_features,
)

# ── ECE (calibration) constants ──────────────────────────────────────────────
# CANONICAL ECE: difficulty-binned, per-batch. Difficulty = pooled success rate;
# tasks are placed into 5 bins by these cutoffs, then |sim_rate - human_rate| is
# summed weighted by bin size.
B_CUTOFFS = [0.20, 0.40, 0.60, 0.80]
N_BINS = 5
MIN_OTHER_SAMPLES = 10

# AgentArena's published baseline set (ALL_LLM_DATAS). Used as the fixed ECE
# difficulty reference so a newly-evaluated OdysSim model is scored on the same
# yardstick as the leaderboard. (gemini-2.5-pro is intentionally excluded — only
# 39 entries.) Kept in sync with analyze_interaction.ALL_LLM_DATAS.
PUBLISHED_BASELINES = [
    "gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
    "UserLM-8b", "CoSER-Llama-3.1-8B", "Human-Like-Qwen2.5-7B-Instruct",
    "humanlm-opinion",
    "deepseek-ai_DeepSeek-V3.1", "gemini-2.0-flash", "gemini-2.5-flash",
    "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3-pro-preview",
    "gemini-3.1-pro-preview", "meta-llama_Llama-3.3-70B-Instruct-Turbo",
    "meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8", "MiniMaxAI_MiniMax-M2.5",
    "moonshotai_Kimi-K2.5", "openai_gpt-oss-120b", "Qwen_Qwen2.5-7B-Instruct-Turbo",
    "Qwen_Qwen3-235B-A22B-Thinking-2507", "Qwen_Qwen3-Next-80B-A3B-Instruct",
    "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219-v1_0", "claude-haiku-4-5-20251001-v1_0",
    "claude-opus-4-20250514-v1_0", "claude-sonnet-4-20250514-v1_0",
]


def resolve_baselines(eval_results_dir, names=None):
    """Return existing ``eval_results_{name}.json`` paths under ``eval_results_dir``.

    Defaults to :data:`PUBLISHED_BASELINES`. Silently skips any missing file so
    a partial local data mirror still scores (difficulty just pools over what is
    present); the set actually used is returned for the caller to report.
    """
    eval_results_dir = Path(eval_results_dir)
    names = names if names is not None else PUBLISHED_BASELINES
    return [eval_results_dir / f"eval_results_{n}.json" for n in names
            if (eval_results_dir / f"eval_results_{n}.json").exists()]


# Frozen ECE difficulty map shipped with the repo (task_key -> pooled success
# over the published baselines). Lets scoring run without the ~135MB of baseline
# eval_results — the difficulty is the only thing they were needed for.
FROZEN_DIFFICULTY_PATH = Path(__file__).resolve().parent / "tau_usi_difficulty.json"


def compute_difficulty_map(eval_results_dir, names=None):
    """Compute the ECE difficulty dict from baseline eval_results files.

    Returns ``(difficulty, used_names)`` where difficulty is ``{task_key: float}``
    (pooled success across all baseline runs) and used_names is the baselines
    actually found. This is the value that :func:`compute_all_with_variance`
    derives from ``difficulty_files`` — freezing it avoids recomputation.
    """
    paths = resolve_baselines(eval_results_dir, names)
    difficulty = compute_difficulty([per_run_rewards(p) for p in paths])
    return difficulty, [p.stem.replace("eval_results_", "") for p in paths]


def dump_difficulty(difficulty: dict, path, source: str = "PUBLISHED_BASELINES", n_baselines: int | None = None):
    """Write the difficulty map to JSON with provenance (sorted, stable)."""
    payload = {
        "source": source,
        "n_baselines": n_baselines,
        "n_tasks": len(difficulty),
        "difficulty": {k: difficulty[k] for k in sorted(difficulty)},
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_difficulty(path=None) -> dict:
    """Load a frozen difficulty map. Accepts the structured ``{difficulty: ...}``
    payload or a flat ``{task_key: float}`` map. Defaults to the shipped file."""
    path = Path(path) if path is not None else FROZEN_DIFFICULTY_PATH
    data = json.loads(Path(path).read_text())
    raw = data["difficulty"] if isinstance(data, dict) and "difficulty" in data else data
    return {k: float(v) for k, v in raw.items()}


def task_key(s: str) -> str:
    """Collapse an instance id to ``{domain}_{task_index}`` (drops the run suffix)."""
    m = re.match(r"^(airline|retail)_(\d+)", s)
    return f"{m.group(1)}_{m.group(2)}" if m else s


def bin_index(value: float) -> int:
    for i, c in enumerate(B_CUTOFFS):
        if value < c:
            return i
    return N_BINS - 1


def per_run_rewards(path) -> dict[str, dict[int, int]]:
    """task_key -> {run_idx: 0/1 reward} for one simulator's eval_results file."""
    d = json.load(open(path))
    per: dict[str, dict[int, int]] = {}
    for k, v in d.items():
        if v.get("keep") is False:
            continue
        r = 1 if (v.get("reward", 0) or 0) > 0 else 0
        m = re.match(r"^(airline|retail)_(\d+)_(\d+)", k)
        run_idx = int(m.group(3)) if m else 1
        per.setdefault(task_key(k), {})[run_idx] = r
    return per


def batch_rewards(batch: dict) -> dict[str, int]:
    """task_key -> 0/1 human reward for one annotation batch."""
    out: dict[str, int] = {}
    for k, v in batch.items():
        if not v.get("keep", True):
            continue
        r = 1 if (v.get("reward", 0) or 0) > 0 else 0
        out[task_key(k)] = r
    return out


def compute_difficulty(per_run_list: list[dict[str, dict[int, int]]]) -> dict[str, float]:
    """task difficulty = mean success pooled across ALL simulators' runs."""
    all_tasks: set[str] = set()
    for runs in per_run_list:
        all_tasks.update(runs.keys())
    difficulty: dict[str, float] = {}
    for t in all_tasks:
        pool: list[int] = []
        for runs in per_run_list:
            if t in runs:
                pool.extend(runs[t].values())
        if len(pool) >= MIN_OTHER_SAMPLES:
            difficulty[t] = float(np.mean(pool))
    return difficulty


def compute_ece_pair(sim: dict[str, int], hum: dict[str, int], difficulty: dict[str, float]):
    """Canonical ECE for one (sim, human) reward map: bin tasks by pooled
    difficulty (cutoffs 0.2/0.4/0.6/0.8), sum |sim_rate - human_rate| weighted
    by bin size."""
    tasks = [t for t in sim if t in hum and t in difficulty]
    if not tasks:
        return None
    bins_idx: dict[int, list[str]] = {b: [] for b in range(N_BINS)}
    for t in tasks:
        bins_idx[bin_index(difficulty[t])].append(t)
    n = len(tasks)
    ece = 0.0
    for b in range(N_BINS):
        chunk = bins_idx[b]
        if not chunk:
            continue
        s = float(np.mean([sim[t] for t in chunk]))
        h = float(np.mean([hum[t] for t in chunk]))
        ece += (len(chunk) / n) * abs(s - h)
    return ece


# ── Eval agreement (survey-based) ────────────────────────────────────────────

def _normalize_ordinal(entry: dict, field: str):
    """Return ordinal survey value normalized to [0,1], or None if missing."""
    omap = FIELD_ORDINAL.get(field, {})
    ans = (entry.get("survey", {}).get(field) or {}).get("answer")
    if not isinstance(ans, str) or ans not in omap:
        return None
    lo, hi = min(omap.values()), max(omap.values())
    return (omap[ans] - lo) / (hi - lo) if hi > lo else 1.0


def compute_eval_agreement(h_entry: dict, l_entry: dict, rng=None):
    """Per-task eval agreement = (1 - mean |h-l| across survey fields) * 100.

    If the LLM entry is missing a field, a random valid value is assigned
    (uniform over the field's ordinal range), penalizing non-answers. Uses
    ``np.random.default_rng`` to match AgentArena exactly.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    diffs = []
    for field in FIELD_ORDINAL:
        hv = _normalize_ordinal(h_entry, field)
        if hv is None:
            continue
        lv = _normalize_ordinal(l_entry, field)
        if lv is None:
            omap = FIELD_ORDINAL[field]
            unique_vals = sorted(set(omap.values()))
            lo, hi = min(unique_vals), max(unique_vals)
            picked = rng.choice(unique_vals)
            lv = (picked - lo) / (hi - lo) if hi > lo else 1.0
        diffs.append(abs(hv - lv))
    if not diffs:
        return None
    return (1.0 - float(np.mean(diffs))) * 100


def compute_model_eval_agreement(batches: list[dict], model_name: str, survey_dir):
    """Eval agreement (mean, std) across the 3 human batches for a model.

    Reads ``survey_comparable_{model_name}.json`` from ``survey_dir``. Returns
    None if that file does not exist.
    """
    survey_path = Path(survey_dir) / f"survey_comparable_{model_name}.json"
    if not survey_path.exists():
        return None
    survey_all = json.loads(survey_path.read_text())

    # Split survey data into 3 sub-batches matching human batch conventions
    survey_batches: list[dict] = [{}, {}, {}]
    for iid, entry in survey_all.items():
        if iid.endswith("_ann2"):
            survey_batches[1][iid[:-5]] = entry
        elif iid.endswith("_ann3"):
            survey_batches[2][iid[:-5]] = entry
        else:
            survey_batches[0][iid] = entry

    rng = np.random.default_rng(42)
    per_batch_scores = []
    for bi, si in zip(batches, survey_batches):
        bi_kept = {k: v for k, v in bi.items() if v.get("keep", True)}
        overlap = sorted(set(bi_kept) & set(si))
        if not overlap:
            continue
        task_scores = []
        for iid in overlap:
            score = compute_eval_agreement(bi_kept[iid], si[iid], rng=rng)
            if score is not None:
                task_scores.append(score)
        if task_scores:
            per_batch_scores.append(float(np.mean(task_scores)))

    if not per_batch_scores:
        return None
    return (float(np.mean(per_batch_scores)), float(np.std(per_batch_scores)))


# ── Dice-Sorensen similarity per dimension ───────────────────────────────────

def compute_dimension_dice(human_row: dict, llm_row: dict) -> dict[str, float]:
    """Dice-Sorensen similarity per dimension from summary rows.

    For each metric m:  sim_m = 2 * min(M_m, H_m) / (M_m + H_m)
    Dimension score = mean(sim_m) * 100.
    """
    _feat_to_row = {
        "politeness_rate": "politeness_pct",
        "short_msg_rate": "short_msg_pct",
        "formality_rate": "formality_pct",
        "ack_only_rate": "ack_only_pct",
        "verbosity_cv": "verbosity_cv",
        "repeat_rate": "repeat_pct",
        "id_confuse_rate": "id_confuse_pct",
        "info_frontload": "info_frontload_pct",
        "id_density": "id_density",
        "user_words_per_turn": "user_wds_per_turn",
        "opening_words": "opening_words",
        "uncertainty_rate": "uncertainty_pct",
        "certainty_rate": "certainty_pct",
        "pushback_q_rate": "pushback_q_pct",
        "clarify_q_rate": "clarify_q_pct",
        "info_q_rate": "info_q_pct",
        "emotion_rate": "emotion_pct",
        "accusation_rate": "accusation_pct",
        "pivot_rate": "pivot_pct",
    }
    result: dict[str, float] = {}
    for dim, keys in DIMENSION_KEYS.items():
        sims = []
        for key in keys:
            row_key = _feat_to_row.get(key, key)
            h = abs(human_row[row_key])
            m = abs(llm_row[row_key])
            if h + m > 0:
                sims.append(2.0 * min(h, m) / (h + m))
            else:
                sims.append(1.0)  # both zero -> identical
        result[dim] = float(np.mean(sims) * 100)
    return result


# ── Row builder ──────────────────────────────────────────────────────────────

def build_row(entries: dict, source: str) -> dict:
    """Build a summary (mean-over-tasks) feature row from a set of entries.

    Feature extraction is delegated to the shared
    ``extract_conversation_features`` so there is one feature definition across
    OdysSim's RL reward and this eval metric.
    """
    feats = [
        f
        for e in entries.values()
        if (f := extract_conversation_features(e.get("conversation", []), source)) is not None
    ]
    rewards = [int(e["reward"]) for e in entries.values()]

    return {
        "n": len(entries),
        # D1: Conversation Patterns
        "turns": float(np.mean([f["n_turns"] for f in feats])),
        "user_wds_per_turn": float(np.mean([f["user_words_per_turn"] for f in feats])),
        "short_msg_pct": float(np.mean([f["short_msg_rate"] for f in feats]) * 100),
        "politeness_pct": float(np.mean([f["politeness_rate"] for f in feats]) * 100),
        "formality_pct": float(np.mean([f["formality_rate"] for f in feats]) * 100),
        "ack_only_pct": float(np.mean([f["ack_only_rate"] for f in feats]) * 100),
        "verbosity_cv": float(np.mean([f["verbosity_cv"] for f in feats])),
        "repeat_pct": float(np.mean([f["repeat_rate"] for f in feats]) * 100),
        "id_confuse_pct": float(np.mean([f["id_confuse_rate"] for f in feats]) * 100),
        # D2: Information Density
        "info_frontload_pct": float(np.mean([f["info_frontload"] for f in feats]) * 100),
        "id_density": float(np.mean([f["id_density"] for f in feats])),
        "opening_words": float(np.mean([f["opening_words"] for f in feats])),
        # D3: Clarification
        "uncertainty_pct": float(np.mean([f["uncertainty_rate"] for f in feats]) * 100),
        "certainty_pct": float(np.mean([f["certainty_rate"] for f in feats]) * 100),
        "pushback_q_pct": float(np.mean([f["pushback_q_rate"] for f in feats]) * 100),
        "clarify_q_pct": float(np.mean([f["clarify_q_rate"] for f in feats]) * 100),
        "info_q_pct": float(np.mean([f["info_q_rate"] for f in feats]) * 100),
        # D4: Error Reaction
        "emotion_pct": float(np.mean([f["emotion_rate"] for f in feats]) * 100),
        "accusation_pct": float(np.mean([f["accusation_rate"] for f in feats]) * 100),
        "pivot_pct": float(np.mean([f["pivot_rate"] for f in feats]) * 100),
        # Outcome
        "success_pct": float(np.mean(rewards) * 100),
    }


# ── Batch loading and run splitting ──────────────────────────────────────────

def load_batches_from_unified(path) -> list[dict]:
    """Split unified.json (165 x 3 annotators) into 3 batch dicts with base IDs."""
    all_data = json.loads(Path(path).read_text())
    batch1, batch2, batch3 = {}, {}, {}
    for iid, entry in all_data.items():
        if iid.endswith("_ann2"):
            batch2[iid[:-5]] = entry
        elif iid.endswith("_ann3"):
            batch3[iid[:-5]] = entry
        else:
            batch1[iid] = entry
    return [batch1, batch2, batch3]


def split_model_runs(model_data: dict) -> list[dict]:
    """Split model data into per-run dicts with base IDs.

    Multi-run models (e.g. 3 runs) use _1/_2/_3 suffixes -> 3 separate dicts.
    Single-run models (<=170 entries) -> returned as-is in a list.
    """
    if len(model_data) <= 170:
        return [model_data]

    runs: dict[str, dict] = {}
    for iid, entry in model_data.items():
        base, suffix = iid.rsplit("_", 1)
        if suffix in ("1", "2", "3"):
            runs.setdefault(suffix, {})[base] = entry
        else:
            runs.setdefault("0", {})[iid] = entry
    return [runs[k] for k in sorted(runs.keys())]


def _average_rows(rows: list[dict]) -> dict:
    """Average numeric values across multiple summary rows."""
    if len(rows) == 1:
        return rows[0]
    avg: dict[str, Any] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        if isinstance(vals[0], (int, float)):
            avg[key] = float(np.mean(vals))
        else:
            avg[key] = vals[0]
    return avg


def _model_label(path) -> str:
    """Derive a short model label from an eval_results filename stem."""
    stem = Path(path).stem  # e.g. "eval_results_UserLM-8b"
    if stem.startswith("eval_results_"):
        return stem[len("eval_results_"):]
    return stem


def compute_all_with_variance(batches, llm_files, survey_dir=None, difficulty_files=None, difficulty=None):
    """Compute mean+/-std for every cell of the USI table (CANONICAL metric).

    USI = mean(D1_conv, D2_info, D3_clarif, D4_react, [eval_agree,] (1-ECE)*100),
    averaged over the 3 human annotation batches.

    ECE difficulty bins (one of, in priority order):
      * ``difficulty``: a precomputed ``{task_key: float}`` map (e.g. the frozen
        :func:`load_difficulty`) — no baseline files needed.
      * ``difficulty_files``: paths whose pooled success defines the bins (a fixed
        reference set scores added models against an uncontaminated difficulty).
      * neither: self-pooled over ``llm_files`` (reproduces the paper).

    Returns a list of result dicts, each with keys: name, D1_conv, D2_info,
    D3_clarif, D4_react, eval_agree, ece, usi -- each (mean, std) or None.
    """
    results = []

    # ── Canonical ECE setup: per-run rewards + pooled difficulty across all sims ──
    llm_paths = [Path(f).resolve() for f in llm_files]
    all_per_run = [per_run_rewards(p) for p in llm_paths]
    if difficulty is None:
        if difficulty_files is None:
            difficulty = compute_difficulty(all_per_run)
        else:
            difficulty = compute_difficulty([per_run_rewards(Path(f).resolve()) for f in difficulty_files])
    h_batches = [batch_rewards(b) for b in batches]

    # ── Human-human baseline (3 pairs) ──────────────────────────────────────
    human_pairs = list(combinations(range(len(batches)), 2))
    hh_dice_vals = {d: [] for d in DIMENSION_KEYS}
    hh_ece_vals = []

    for i, j in human_pairs:
        bi = {k: v for k, v in batches[i].items() if v.get("keep", True)}
        bj = {k: v for k, v in batches[j].items() if v.get("keep", True)}
        overlap = sorted(set(bi) & set(bj))
        if not overlap:
            continue
        row_i = build_row({k: bi[k] for k in overlap}, "human")
        row_j = build_row({k: bj[k] for k in overlap}, "human")
        dice = compute_dimension_dice(row_i, row_j)
        for d in DIMENSION_KEYS:
            hh_dice_vals[d].append(dice[d])
        ece = compute_ece_pair(h_batches[i], h_batches[j], difficulty)
        if ece is not None:
            hh_ece_vals.append(ece)

    # ── Human eval agreement (fair: same-conversation comparison) ──────────
    hh_eval_agree_val = None
    ea_mean_for_usi = None
    if survey_dir is not None:
        human_eval_path = Path(survey_dir) / "survey_comparable_human-eval-100.json"
        if human_eval_path.exists():
            human_eval_data = json.loads(human_eval_path.read_text())
            batch1 = {k: v for k, v in batches[0].items() if v.get("keep", True)}
            rng_hh = np.random.default_rng(42)
            task_scores = []
            for iid, eval_entry in human_eval_data.items():
                if iid in batch1:
                    score = compute_eval_agreement(batch1[iid], eval_entry, rng=rng_hh)
                    if score is not None:
                        task_scores.append(score)
            if task_scores:
                hh_eval_agree_val = (float(np.mean(task_scores)), float(np.std(task_scores)))
                ea_mean_for_usi = hh_eval_agree_val[0]

    # USI per human pair (indexed against the ECE list)
    hh_usi_vals = []
    for idx in range(len(hh_ece_vals)):
        d1 = hh_dice_vals["D1_conv"][idx]
        d2 = hh_dice_vals["D2_info"][idx]
        d3 = hh_dice_vals["D3_clarif"][idx]
        d4 = hh_dice_vals["D4_react"][idx]
        ece = hh_ece_vals[idx]
        if ea_mean_for_usi is not None:
            usi = (d1 + d2 + d3 + d4 + ea_mean_for_usi + (1 - ece) * 100) / 6
        else:
            usi = (d1 + d2 + d3 + d4 + (1 - ece) * 100) / 5
        hh_usi_vals.append(usi)

    results.append({
        "name": "Human (inter-ann.)",
        "D1_conv": (float(np.mean(hh_dice_vals["D1_conv"])), float(np.std(hh_dice_vals["D1_conv"]))),
        "D2_info": (float(np.mean(hh_dice_vals["D2_info"])), float(np.std(hh_dice_vals["D2_info"]))),
        "D3_clarif": (float(np.mean(hh_dice_vals["D3_clarif"])), float(np.std(hh_dice_vals["D3_clarif"]))),
        "D4_react": (float(np.mean(hh_dice_vals["D4_react"])), float(np.std(hh_dice_vals["D4_react"]))),
        "eval_agree": hh_eval_agree_val,
        "ece": (float(np.mean(hh_ece_vals)), float(np.std(hh_ece_vals))) if hh_ece_vals else (None, None),
        "usi": (float(np.mean(hh_usi_vals)), float(np.std(hh_usi_vals))) if hh_usi_vals else (None, None),
    })

    # ── Model scores (pooled model vs each of 3 human batches) ──────────────
    for idx_f, llm_file in enumerate(llm_files):
        llm_path = llm_paths[idx_f]
        model_name = _model_label(llm_path)
        llm_all = json.loads(llm_path.read_text())
        runs = split_model_runs(llm_all)
        runs_per_task = all_per_run[idx_f]
        run_indices = sorted({r for d in runs_per_task.values() for r in d})

        dice_vals = {d: [] for d in DIMENSION_KEYS}
        ece_vals = []
        usi_vals = []

        eval_agree_result = None
        if survey_dir:
            eval_agree_result = compute_model_eval_agreement(batches, model_name, survey_dir)

        for j, bi in enumerate(batches):
            bi_kept = {k: v for k, v in bi.items() if v.get("keep", True)}

            # D1-D4: pooled model row across runs vs this human batch
            run_rows = []
            for run in runs:
                run_kept = {k: v for k, v in run.items() if v.get("keep", True)}
                overlap = sorted(set(bi_kept) & set(run_kept))
                if not overlap:
                    continue
                run_rows.append((overlap, build_row({k: run_kept[k] for k in overlap}, "llm")))
            if not run_rows:
                continue
            pooled_model_row = _average_rows([r for _, r in run_rows])
            all_overlap_ids = sorted(set().union(*(set(ov) for ov, _ in run_rows)))
            human_paired = {k: bi_kept[k] for k in all_overlap_ids}
            human_row = build_row(human_paired, "human")
            dice = compute_dimension_dice(human_row, pooled_model_row)
            for d in DIMENSION_KEYS:
                dice_vals[d].append(dice[d])

            # Canonical ECE: each single run vs this batch, averaged across runs
            hb = h_batches[j]
            run_eces = []
            for ri in run_indices:
                sim_run = {t: rec[ri] for t, rec in runs_per_task.items() if ri in rec}
                e = compute_ece_pair(sim_run, hb, difficulty)
                if e is not None:
                    run_eces.append(e)
            if not run_eces:
                continue
            ece = float(np.mean(run_eces))
            ece_vals.append(ece)

            if eval_agree_result is not None:
                ea_mean = eval_agree_result[0]
                usi = (dice["D1_conv"] + dice["D2_info"] + dice["D3_clarif"] +
                       dice["D4_react"] + ea_mean + (1 - ece) * 100) / 6
            else:
                usi = (dice["D1_conv"] + dice["D2_info"] + dice["D3_clarif"] +
                       dice["D4_react"] + (1 - ece) * 100) / 5
            usi_vals.append(usi)

        results.append({
            "name": model_name,
            "D1_conv": (float(np.mean(dice_vals["D1_conv"])), float(np.std(dice_vals["D1_conv"]))),
            "D2_info": (float(np.mean(dice_vals["D2_info"])), float(np.std(dice_vals["D2_info"]))),
            "D3_clarif": (float(np.mean(dice_vals["D3_clarif"])), float(np.std(dice_vals["D3_clarif"]))),
            "D4_react": (float(np.mean(dice_vals["D4_react"])), float(np.std(dice_vals["D4_react"]))),
            "eval_agree": eval_agree_result,
            "ece": (float(np.mean(ece_vals)), float(np.std(ece_vals))) if ece_vals else (None, None),
            "usi": (float(np.mean(usi_vals)), float(np.std(usi_vals))) if usi_vals else (None, None),
        })

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
# `score`  — aggregate OdysSim's one-by-one *_task_results.json into the USI table
# `freeze` — recompute the shipped ECE difficulty map from baseline eval_results

def _default_data_dir():
    import os
    env = os.getenv("TAU_USI_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data" / "tau_usi"


def task_results_to_eval_results(payload):
    """Convert a ``*_task_results.json`` payload (``{"results": [...]}``) — or an
    already-converted flat ``eval_results`` dict — into ``{instance_id: entry}``."""
    if isinstance(payload, dict) and "results" not in payload:
        if payload and all(isinstance(v, dict) and "conversation" in v for v in payload.values()):
            return payload
    records = payload["results"] if isinstance(payload, dict) else payload
    out = {}
    for r in records:
        iid = r.get("instance_id") or f"{r['domain']}_{r['task_index']}"
        out[iid] = {
            "instance_id": iid, "agent_id": r.get("agent_id", "agent-origin"),
            "conversation": r["conversation"], "survey": r.get("survey", {}) or {},
            "reward": float(r.get("reward", 0) or 0), "keep": bool(r.get("keep", True)),
        }
    return out


def _fmt_cell(pair, scale=1.0, dec=1):
    if not pair or pair[0] is None:
        return f"{'NA':>11s}"
    return f" {pair[0] * scale:5.{dec}f}±{pair[1] * scale:<4.{dec}f}"


def _print_table(results, focus=None):
    hdr = "{:32s}" + " {:>11s}" * 7
    print(hdr.format("Model", "D1", "D2", "D3", "D4", "Eval", "ECE", "USI"))
    print("-" * 116)
    human = [r for r in results if r["name"] == "Human (inter-ann.)"]
    rest = sorted([r for r in results if r["name"] != "Human (inter-ann.)"],
                  key=lambda r: -(r["usi"][0] if r["usi"] and r["usi"][0] is not None else -1))
    for r in human + rest:
        line = f"{r['name'][:32]:32s}"
        for k in ("D1_conv", "D2_info", "D3_clarif", "D4_react"):
            line += _fmt_cell(r[k])
        line += _fmt_cell(r["eval_agree"])
        line += _fmt_cell(r["ece"], scale=100.0, dec=2)  # ECE stored 0-1, shown x100
        line += _fmt_cell(r["usi"], dec=2)
        print(line + (" *" if focus and r["name"] == focus else ""))


def _score_cli(args):
    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    unified = data_dir / "tau_bench_tasks_unified.json"
    ev_dir, survey_dir = data_dir / "eval_results", data_dir / "survey_data"
    if not unified.exists():
        raise SystemExit(f"missing {unified} (sync tau_bench_tasks_unified.json from the AgentArena box)")

    in_path = Path(args.input)
    label = args.label or in_path.stem.replace("_task_results", "").replace("eval_results_", "")
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_results = task_results_to_eval_results(json.loads(in_path.read_text()))
    converted = out_dir / f"eval_results_{label}.json"
    converted.write_text(json.dumps(eval_results, ensure_ascii=False))
    print(f"[usi] {label}: {len(eval_results)} records from {in_path.name}")

    baselines = resolve_baselines(ev_dir) if ev_dir.exists() else []
    difficulty = diff_files = None
    if args.difficulty == "frozen":
        difficulty = load_difficulty(args.difficulty_json)
        print(f"[usi] difficulty: frozen map ({len(difficulty)} tasks) — no baselines needed")
    elif args.difficulty == "baselines":
        if not baselines:
            raise SystemExit(f"--difficulty baselines needs {ev_dir}/eval_results_*.json")
        diff_files = baselines
    # else 'self': self-pooled over llm_files

    llm_files = ([*baselines, converted] if (args.print_all and baselines) else [converted])
    survey_arg = survey_dir if survey_dir.exists() else None
    batches = load_batches_from_unified(unified)
    results = compute_all_with_variance(batches, llm_files, survey_arg,
                                        difficulty_files=diff_files, difficulty=difficulty)
    by_name = {r["name"]: r for r in results}
    row = by_name.get(label)
    print()
    _print_table(results if args.print_all else [by_name["Human (inter-ann.)"], row], focus=label)
    print()

    keys = ["D1_conv", "D2_info", "D3_clarif", "D4_react", "eval_agree", "ece", "usi"]
    metrics = {
        "label": label, "difficulty_reference": args.difficulty,
        "model": ({"name": label, **{k: row.get(k) for k in keys}} if row else None),
        "human_inter_annotator": {"name": "Human (inter-ann.)",
                                  **{k: by_name["Human (inter-ann.)"].get(k) for k in keys}},
    }
    mpath = out_dir / f"{label}_aggregate_metrics.json"
    mpath.write_text(json.dumps(metrics, indent=2))
    usi = row["usi"][0] if row and row["usi"] else None
    print(f"[usi] USI={usi:.2f} -> {mpath.name}" if usi is not None else f"[usi] USI=NA -> {mpath.name}")


def _freeze_cli(args):
    difficulty, used = compute_difficulty_map(args.eval_results_dir)
    if len(used) != len(PUBLISHED_BASELINES):
        print(f"WARNING: {len(used)}/{len(PUBLISHED_BASELINES)} baselines found; pooling over those present.")
    dump_difficulty(difficulty, args.out, source="PUBLISHED_BASELINES", n_baselines=len(used))
    print(f"froze {len(difficulty)} task difficulties from {len(used)} baselines -> {args.out}")


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="tau-USI scorer — matches AgentArena's compute_all_with_variance.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="aggregate a *_task_results.json into the USI table")
    s.add_argument("input", help="OdysSim *_task_results.json (or an already-converted eval_results dict)")
    s.add_argument("--data-dir", default=None, help="default <repo>/data/tau_usi or $TAU_USI_DATA_DIR")
    s.add_argument("--label", default=None)
    s.add_argument("--out-dir", default=None)
    s.add_argument("--difficulty", choices=["frozen", "baselines", "self"], default="frozen",
                   help="frozen (shipped map, no baselines — default) | baselines (recompute) | self (paper self-pool)")
    s.add_argument("--difficulty-json", default=None)
    s.add_argument("--print-all", action="store_true", help="print the full leaderboard (needs baseline eval_results)")
    s.set_defaults(fn=_score_cli)

    f = sub.add_parser("freeze", help="recompute the shipped ECE difficulty map from baselines")
    f.add_argument("--eval-results-dir", required=True)
    f.add_argument("--out", default=str(FROZEN_DIFFICULTY_PATH))
    f.set_defaults(fn=_freeze_cli)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    _main()
