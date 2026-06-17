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

import threading
from random import Random
from typing import Any

from agents.tau_usi.utils import _RATE_FEATURES, DIMENSION_KEYS, FIELD_ORDINAL, _safe_mean


class FeatureStatsBuffer:
    """
    EMA buffer tracking the model's current feature distribution across batches.

    Passed in via ``context["feature_stats_buffer"]``.  Thread-safe so it can
    be shared across concurrent ``agent_loop`` coroutines.

    Usage::

        buf = FeatureStatsBuffer(ema_alpha=0.1)
        buf.add(features_dict)   # called after each rollout
        mu = buf.read()          # returns current EMA estimates
    """

    def __init__(self, ema_alpha: float = 0.1) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self._alpha = ema_alpha
        self._mu: dict[str, float] = {}
        self._n: int = 0
        self._lock = threading.Lock()

    def add(self, features: dict[str, float]) -> None:
        """Update EMA estimates with features from one new rollout."""
        with self._lock:
            self._n += 1
            for key, value in features.items():
                if key not in self._mu:
                    self._mu[key] = float(value)
                else:
                    self._mu[key] = (1.0 - self._alpha) * self._mu[key] + self._alpha * float(value)

    def read(self) -> dict[str, float]:
        """Return a snapshot of current EMA estimates."""
        with self._lock:
            return dict(self._mu)

    @property
    def n_samples(self) -> int:
        return self._n


def _mse_weight(h_k: float, mu_k: float, ci_half: float) -> float:
    """
    Linear weight for MSE-based distributional reward.

    Returns (h_k - mu_k), zero inside the dead zone [h_k ± ci_half].
    Magnitude shrinks naturally as mu_k → h_k, giving exponential convergence.
    """
    gap = h_k - mu_k
    if abs(gap) <= ci_half:
        return 0.0
    return gap


def _kl_weight(h_k: float, mu_k: float, ci_half: float, clip: float = 3.0) -> float:
    """
    Forward-KL (human || model) gradient weight for Bernoulli features.

    Gradient of KL(Bernoulli(h_k) || Bernoulli(mu_k)) w.r.t. mu_k:
        h_k / mu_k - (1 - h_k) / (1 - mu_k)

    Stronger than MSE when mu_k is far from h_k (especially near 0 or 1).
    Clipped and dead-zoned to prevent explosion.
    """
    gap = h_k - mu_k
    if abs(gap) <= ci_half:
        return 0.0
    eps = 1e-6
    mu_safe = max(eps, min(1.0 - eps, mu_k))
    grad = h_k / mu_safe - (1.0 - h_k) / (1.0 - mu_safe)
    return max(-clip, min(clip, grad))


def _normalize_ordinal(entry: dict[str, Any], field: str) -> float | None:
    ordinal_map = FIELD_ORDINAL.get(field, {})
    answer = (entry.get("survey", {}).get(field) or {}).get("answer")
    if not isinstance(answer, str) or answer not in ordinal_map:
        return None
    low = min(ordinal_map.values())
    high = max(ordinal_map.values())
    if high == low:
        return 1.0
    return (ordinal_map[answer] - low) / (high - low)


def compute_eval_agreement(
    human_entry: dict[str, Any],
    llm_entry: dict[str, Any],
    rng: Random,
) -> float | None:
    """
    Per-task eval agreement: (1 - mean absolute ordinal difference) * 100.
    """
    diffs = []
    for field in FIELD_ORDINAL:
        human_value = _normalize_ordinal(human_entry, field)
        if human_value is None:
            continue
        llm_value = _normalize_ordinal(llm_entry, field)
        if llm_value is None:
            ordinal_values = sorted(set(FIELD_ORDINAL[field].values()))
            low = min(ordinal_values)
            high = max(ordinal_values)
            picked = rng.choice(ordinal_values)
            llm_value = 1.0 if high == low else (picked - low) / (high - low)
        diffs.append(abs(human_value - llm_value))

    if not diffs:
        return None
    return (1.0 - _safe_mean(diffs)) * 100.0


def compute_distributional_reward(
    features: dict[str, float],
    human_targets: dict[str, float],
    mu_estimates: dict[str, float],
    mode: str = "mse",
    ci_z: float = 1.0,
    sigma_fallback: float = 0.1,
    model_reward: float | None = None,
    human_reward: float | None = None,
    model_survey: dict[str, Any] | None = None,
    human_survey: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Per-sample proxy reward covering all USI components.

    D1–D4 (distributional moment matching):
        For each dimension, reward = mean over its features of (w_k * x_k),
        where w_k reflects how far the model distribution (mu_k) is from the
        human target (h_k).  Dead zone ±ci_z*sigma_fallback suppresses noise
        when the model is already within expected variance.

    ECE proxy (requires ``model_reward`` and ``human_reward``):
        (1 - |round(model_reward) - round(human_reward)|) * 100.
        100 if the binary task outcomes match, 0 if they differ.

    Eval agreement (requires ``model_survey`` and ``human_survey``):
        Ordinal agreement across survey fields, in [0, 100].
        Format: {field: {"answer": <option_string>}, ...}.

    Returns:
        Dict with keys ``"D1_conv"``, ``"D2_info"``, ``"D3_clarif"``,
        ``"D4_react"``, ``"ece"``, ``"eval_agreement"`` (0.0 when inputs not
        provided), and ``"total"`` (mean of all six components).
    """
    ci_half = ci_z * sigma_fallback
    result: dict[str, float] = {}

    # D1–D4: distributional moment matching
    for dim_name, feature_keys in DIMENSION_KEYS.items():
        feat_rewards = []
        for feat_key in feature_keys:
            h_k = human_targets.get(feat_key)
            x_i = features.get(feat_key)
            if h_k is None or x_i is None:
                continue
            mu_k = mu_estimates.get(feat_key, h_k)
            if mode == "kl" and feat_key in _RATE_FEATURES:
                w = _kl_weight(h_k, mu_k, ci_half)
            else:
                w = _mse_weight(h_k, mu_k, ci_half)
            feat_rewards.append(w * float(x_i))
        result[dim_name] = _safe_mean(feat_rewards) if feat_rewards else 0.0

    # ECE proxy: reward matching the human pass/fail outcome on this task
    if model_reward is not None and human_reward is not None:
        result["ece"] = (1.0 - abs(round(model_reward) - round(human_reward))) * 100.0
    else:
        result["ece"] = 0.0

    # Eval agreement: ordinal survey agreement vs human annotation
    if model_survey is not None and human_survey is not None:
        human_entry = {"survey": human_survey}
        llm_entry = {"survey": model_survey}
        score = compute_eval_agreement(human_entry, llm_entry, rng=Random(42))
        result["eval_agreement"] = score if score is not None else 0.0
    else:
        result["eval_agreement"] = 0.0

    result["total"] = _safe_mean(list(result.values()))
    return result
