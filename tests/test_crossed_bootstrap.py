"""The crossed bootstrap must see seed variance the single-level one cannot."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.crossed_bootstrap import (crossed_bootstrap_difference,
                                              holm_adjust)


def _accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def _make(rng, n, flip):
    """Predictions agreeing with truth except on `flip` fraction of cases."""
    y = rng.integers(0, 5, size=n)
    p = y.copy()
    idx = rng.choice(n, size=int(flip * n), replace=False)
    p[idx] = (p[idx] + 1) % 5
    return y, p


def test_a_consistent_advantage_is_detected() -> None:
    rng = np.random.default_rng(0)
    n = 800
    y = rng.integers(0, 5, size=n)
    ref, cand = {}, {}
    for s in range(5):
        r = y.copy()
        bad = rng.choice(n, size=200, replace=False)
        r[bad] = (r[bad] + 1) % 5
        c = y.copy()
        bad = rng.choice(n, size=120, replace=False)
        c[bad] = (c[bad] + 1) % 5
        ref[s], cand[s] = r, c
    out = crossed_bootstrap_difference(y, ref, cand, metric=_accuracy,
                                       n_bootstrap=400, seed=1)
    assert out["difference"] > 0
    assert out["ci_lower"] > 0, "a consistent advantage should exclude zero"
    assert out["n_seeds"] == 5
    assert out["sign_agreement"] == 5


def test_seed_disagreement_widens_the_interval_to_include_zero() -> None:
    """The failure mode the single-level bootstrap is blind to.

    Each seed shows a large within-seed effect, but the sign alternates across
    seeds. A case-only bootstrap on any one seed would be emphatic; the crossed
    interval must span zero.
    """
    rng = np.random.default_rng(2)
    n = 800
    y = rng.integers(0, 5, size=n)
    ref, cand = {}, {}
    for s in range(6):
        strong, weak = (120, 220) if s % 2 == 0 else (220, 120)
        r = y.copy()
        bad = rng.choice(n, size=weak, replace=False)
        r[bad] = (r[bad] + 1) % 5
        c = y.copy()
        bad = rng.choice(n, size=strong, replace=False)
        c[bad] = (c[bad] + 1) % 5
        ref[s], cand[s] = r, c
    out = crossed_bootstrap_difference(y, ref, cand, metric=_accuracy,
                                       n_bootstrap=400, seed=3)
    assert out["ci_lower"] < 0 < out["ci_upper"], (
        "alternating per-seed signs must produce an interval spanning zero")
    assert out["sign_agreement"] < out["n_seeds"]


def test_one_case_sample_is_shared_across_seeds() -> None:
    """Pairing is the point: identical models must give exactly zero."""
    rng = np.random.default_rng(4)
    n = 500
    y = rng.integers(0, 5, size=n)
    preds = {s: _make(rng, n, 0.3)[1] for s in range(4)}
    out = crossed_bootstrap_difference(y, preds, preds, metric=_accuracy,
                                       n_bootstrap=200, seed=5)
    assert out["difference"] == 0.0
    assert out["ci_lower"] == 0.0 and out["ci_upper"] == 0.0, (
        "independent case samples per model would make this non-zero")


def test_fewer_than_two_paired_seeds_raises() -> None:
    y = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError, match="two paired seeds"):
        crossed_bootstrap_difference(y, {0: y}, {0: y}, metric=_accuracy)


def test_misaligned_predictions_raise() -> None:
    y = np.array([0, 1, 2, 3])
    short = np.array([0, 1])
    with pytest.raises(ValueError, match="align"):
        crossed_bootstrap_difference(y, {0: y, 1: y}, {0: short, 1: y},
                                     metric=_accuracy)


def test_holm_is_monotone_and_bounded() -> None:
    raw = [0.001, 0.02, 0.04, 0.5]
    adj = holm_adjust(raw)
    assert all(a >= r for a, r in zip(adj, raw))
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj == sorted(adj), "step-down adjustment must stay monotone"
    named = holm_adjust(raw, ["a", "b", "c", "d"])
    assert set(named) == {"a", "b", "c", "d"}
