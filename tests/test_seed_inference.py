"""Formal inference must be a calibrated test, not bootstrap tail mass.

The regression these guard: ``crossed_bootstrap_difference`` returned a
``p_value`` computed as the tail mass of the bootstrap distribution, and two
analyses -- the Q1 configuration decomposition and the headline foundation-model
comparison -- Holm-corrected it and reported the result as surviving correction.
That distribution is generated around the empirical estimate, not under
H0: delta = 0, so the quantity was never a p-value and correcting it lent it an
authority it did not have.

Nothing caught it because no test ever asserted on that key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("scipy")

from src.evaluation.seed_inference import (  # noqa: E402
    min_attainable_signflip_p,
    seed_level_test,
)


def test_the_bootstrap_no_longer_returns_a_p_value() -> None:
    """Absent, not NaN, so stale callers fail loudly instead of silently."""
    from src.evaluation.crossed_bootstrap import crossed_bootstrap_difference

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    reference = {s: rng.integers(0, 5, size=200) for s in (42, 1, 2)}
    candidate = {s: rng.integers(0, 5, size=200) for s in (42, 1, 2)}

    result = crossed_bootstrap_difference(
        y_true, reference, candidate,
        metric=lambda t, p: float((t == p).mean()), n_bootstrap=50, seed=0)

    assert "p_value" not in result, (
        "the uncalibrated bootstrap p-value is back; formal inference belongs "
        "in seed_inference.seed_level_test")
    assert "per_seed_difference" in result, "inference needs the per-seed effects"


def test_the_sign_flip_floor_is_two_over_two_to_the_n() -> None:
    """The observed assignment and its negation always qualify."""
    assert min_attainable_signflip_p(3) == pytest.approx(0.25)
    assert min_attainable_signflip_p(4) == pytest.approx(0.125)
    assert min_attainable_signflip_p(5) == pytest.approx(0.0625)
    assert min_attainable_signflip_p(10) == pytest.approx(0.001953125)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_no_effect_size_beats_the_floor_at_small_n(n) -> None:
    """A separated effect still cannot go below the floor. This is the point.

    At three seeds a reader seeing p = 0.25 must not read it as evidence of
    absence: it is the smallest value the test can produce.
    """
    # Hugely separated from zero but not identical, so the t-test is not
    # degenerate and the assertion is about the permutation floor alone.
    enormous = [1000.0 + i for i in range(n)]
    result = seed_level_test(enormous)
    assert result["p_signflip"] == pytest.approx(min_attainable_signflip_p(n))
    assert result["p_signflip"] >= 2.0 / 2 ** n


def test_a_consistent_effect_is_detected_by_the_t_test() -> None:
    values = [0.086, 0.067, 0.103]           # the Q1 EyePACS resolution seeds
    result = seed_level_test(values)
    assert result["mean"] == pytest.approx(0.0853, abs=1e-3)
    assert result["p_ttest"] < 0.05
    assert result["sign_agreement"] == 3


def test_a_null_is_not_detected() -> None:
    values = [0.0304, 0.0153, -0.0148]       # the Q1 EyePACS batch seeds
    result = seed_level_test(values)
    assert result["p_ttest"] > 0.4
    assert result["sign_agreement"] == 2


def test_the_t_test_and_the_flip_disagree_in_the_expected_direction() -> None:
    """The permutation test is the more conservative of the two at small n."""
    values = [0.05, 0.06, 0.07, 0.04, 0.05]
    result = seed_level_test(values)
    assert result["p_ttest"] < result["p_signflip"], (
        "the exact test should be no more powerful than the parametric one here")


def test_fewer_than_two_seeds_raises() -> None:
    with pytest.raises(ValueError, match="at least two seeds"):
        seed_level_test([0.05])


def test_per_seed_values_are_returned_for_reporting() -> None:
    values = [0.01, -0.02, 0.03]
    result = seed_level_test(values)
    assert result["per_seed"] == pytest.approx(values)
    assert result["n_seeds"] == 3
    assert result["seed_sd"] == pytest.approx(float(np.std(values, ddof=1)))


def test_no_analysis_holm_corrects_the_bootstrap_tail_mass() -> None:
    """Guards the specific misuse across the scripts that do inference."""
    root = Path(__file__).resolve().parents[1]
    for name in ("analyse_crossed.py", "analyse_configuration.py",
                 "analyse_interaction.py"):
        source = (root / name).read_text(encoding="utf-8")
        if "holm_adjust(" not in source:
            continue
        for line in source.splitlines():
            if "holm_adjust(" in line and "def " not in line:
                assert "p_value" not in line, (
                    f"{name} Holm-corrects a bootstrap p_value: {line.strip()}")
