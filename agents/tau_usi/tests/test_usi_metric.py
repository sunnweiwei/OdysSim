"""Tests for the canonical tau-USI scorer (agents.tau_usi.usi_metric).

Two layers:
  * synthetic unit tests — hand-computable Dice / ECE / eval-agreement; no data,
    always run.
  * golden regression — asserts the scorer reproduces the exact numbers verified
    against AgentArena's analyze_interaction.compute_all_with_variance on
    osim-8b-v6. Skipped automatically if the local data mirror is absent.

Run: pytest agents/tau_usi/tests/test_usi_metric.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from agents.tau_usi import usi_metric as um
from agents.tau_usi import score_eval

DATA_DIR = score_eval._default_data_dir()
UNIFIED = DATA_DIR / "tau_bench_tasks_unified.json"
EV_DIR = DATA_DIR / "eval_results"
SURVEY_DIR = DATA_DIR / "survey_data"
TASK_RESULTS = DATA_DIR / "v6_task_results.json"


# ── synthetic unit tests (no data) ───────────────────────────────────────────

def test_bin_index_and_task_key():
    assert um.bin_index(0.0) == 0
    assert um.bin_index(0.19) == 0
    assert um.bin_index(0.5) == 2   # 0.5 < 0.6
    assert um.bin_index(0.95) == 4
    assert um.task_key("retail_12_3") == "retail_12"
    assert um.task_key("airline_7") == "airline_7"


def test_compute_dimension_dice_d4():
    # D4_react = mean Dice over {emotion, accusation, pivot}
    #   emotion:    2*min(10,10)/20 = 1.0
    #   accusation: both 0           = 1.0
    #   pivot:      2*min(20,10)/30  = 0.6667
    #   mean * 100 = 88.888...
    human = {"emotion_pct": 10.0, "accusation_pct": 0.0, "pivot_pct": 20.0,
             # other dims present but irrelevant here
             "politeness_pct": 0, "short_msg_pct": 0, "formality_pct": 0,
             "ack_only_pct": 0, "verbosity_cv": 0, "repeat_pct": 0, "id_confuse_pct": 0,
             "info_frontload_pct": 0, "id_density": 0, "user_wds_per_turn": 0, "opening_words": 0,
             "uncertainty_pct": 0, "certainty_pct": 0, "pushback_q_pct": 0,
             "clarify_q_pct": 0, "info_q_pct": 0}
    llm = dict(human, emotion_pct=10.0, accusation_pct=0.0, pivot_pct=10.0)
    dice = um.compute_dimension_dice(human, llm)
    assert dice["D4_react"] == pytest.approx(100 * (1 + 1 + 2 * 10 / 30) / 3)
    # all-zero dimension -> identical -> 100
    assert dice["D1_conv"] == pytest.approx(100.0)


def test_compute_ece_pair():
    sim = {"a": 1, "b": 0}
    hum = {"a": 1, "b": 1}
    difficulty = {"a": 0.5, "b": 0.5}  # both fall in bin 2 -> one chunk
    # s=0.5, h=1.0 -> 1.0 * |0.5-1.0| = 0.5
    assert um.compute_ece_pair(sim, hum, difficulty) == pytest.approx(0.5)
    # no overlapping/difficulty-known tasks -> None
    assert um.compute_ece_pair({"x": 1}, {"x": 1}, {}) is None


def test_compute_eval_agreement():
    field = "human_like"  # {No:1, Partially:2, Yes:3}
    same = um.compute_eval_agreement(
        {"survey": {field: {"answer": "Yes"}}},
        {"survey": {field: {"answer": "Yes"}}},
    )
    assert same == pytest.approx(100.0)
    opposite = um.compute_eval_agreement(
        {"survey": {field: {"answer": "Yes"}}},   # 1.0
        {"survey": {field: {"answer": "No"}}},     # 0.0
    )
    assert opposite == pytest.approx(0.0)
    # no human fields -> None
    assert um.compute_eval_agreement({"survey": {}}, {"survey": {}}) is None


def test_split_model_runs_single():
    data = {f"retail_{i}": {} for i in range(10)}
    assert um.split_model_runs(data) == [data]  # <=170 -> single run


# ── golden regression (data-gated) ───────────────────────────────────────────

_HAVE_DATA = UNIFIED.exists() and EV_DIR.exists() and TASK_RESULTS.exists()

# Verified bit-identical (max abs diff 0.0) vs AgentArena
# analyze_interaction.compute_all_with_variance on osim-8b-v6,
# difficulty_files = the 31 published baselines.
GOLDEN_OSIM_8B_V6 = {
    "D1_conv": 45.87058977689687,
    "D2_info": 85.13184627861251,
    "D3_clarif": 65.84818502653555,
    "D4_react": 78.64063046072489,
    "ece": 0.17979797979797982,
    "usi": 71.50229071259439,
    "eval_agree": None,  # no survey_comparable_osim-8b-v6.json -> 5-component USI
}


def _score_v6(difficulty=None, difficulty_files=None):
    payload = score_eval.json.loads(TASK_RESULTS.read_text())
    eval_results = score_eval.task_results_to_eval_results(payload)
    model_path = EV_DIR / "_golden_osim-8b-v6.json"
    model_path.write_text(score_eval.json.dumps(eval_results))
    try:
        batches = um.load_batches_from_unified(UNIFIED)
        files = ([model_path] if difficulty is not None
                 else um.resolve_baselines(EV_DIR) + [model_path])
        results = um.compute_all_with_variance(
            batches, files, SURVEY_DIR, difficulty_files=difficulty_files, difficulty=difficulty)
    finally:
        model_path.unlink(missing_ok=True)
    return {r["name"]: r for r in results}["_golden_osim-8b-v6"]


def _assert_golden(row):
    for key, expected in GOLDEN_OSIM_8B_V6.items():
        got = row[key]
        if expected is None:
            assert got is None, f"{key}: expected None, got {got}"
        else:
            assert got[0] == pytest.approx(expected, abs=1e-9), f"{key}: {got[0]} != {expected}"


def test_frozen_difficulty_file_shipped():
    # committed alongside usi_metric.py — not data-gated
    diff = um.load_difficulty()
    assert len(diff) == 165
    assert all(0.0 <= v <= 1.0 for v in diff.values())


@pytest.mark.skipif(not _HAVE_DATA, reason="tau_usi data mirror not present (sync from AgentArena box)")
def test_golden_osim_8b_v6_from_baselines():
    assert len(um.resolve_baselines(EV_DIR)) == len(um.PUBLISHED_BASELINES), "expected all 31 baselines present"
    _assert_golden(_score_v6(difficulty_files=um.resolve_baselines(EV_DIR)))


@pytest.mark.skipif(not _HAVE_DATA, reason="tau_usi data mirror not present (sync from AgentArena box)")
def test_golden_osim_8b_v6_from_frozen_map():
    # frozen map must reproduce the same numbers without any baseline files
    _assert_golden(_score_v6(difficulty=um.load_difficulty()))
    # and the frozen map must equal a fresh recompute from the baselines
    assert um.compute_difficulty_map(EV_DIR)[0] == um.load_difficulty()
