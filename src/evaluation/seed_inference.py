"""Formal inference on per-seed paired effects.

Why this is separate from the bootstrap
---------------------------------------
``crossed_bootstrap_difference`` used to return a ``p_value`` computed as the
tail mass of the bootstrap distribution::

    p = 2 * min(mean(draws <= 0), mean(draws >= 0))

That quantity is not a calibrated null-hypothesis test. The bootstrap
distribution is generated around the *empirical* estimate, not under
H0: delta = 0, so its tail mass answers "how much of the resampling
distribution sits on the other side of zero", which is a statement about the
estimate's spread, not about the probability of the data under a null. Holm-
correcting it compounds the error by dressing an uncalibrated number in the
apparatus of a controlled family-wise error rate.

It was used that way. The Q1 configuration decomposition and the headline
foundation-model comparison both Holm-corrected bootstrap tail mass and
reported the result as surviving correction. This module replaces that.

The division of labour is now:

* the crossed bootstrap gives the point estimate, the 95% uncertainty interval,
  the seed SD and the sign agreement;
* this module gives the formal test, on the per-seed paired effects.

Two tests, and what each is for
-------------------------------
``p_ttest`` is primary: a one-sample t-test of the per-seed deltas against
zero. It assumes approximate normality of the seed effects, which at three
seeds is an assumption and not a check.

``p_signflip`` is a sensitivity analysis: the exact permutation test over all
2^n sign assignments, which assumes only symmetry under the null. It is the
more honest test at small n and the less powerful one -- and at small n it
*cannot* reach significance at all. The observed assignment and its exact
negation always land at or beyond the observed value, so the smallest
attainable two-sided p is 2 / 2^n:

    n = 3   ->  0.2500      n = 5   ->  0.0625
    n = 4   ->  0.1250      n = 10  ->  0.0020

At three seeds no effect of any size can produce a sign-flip p below 0.25.
``min_attainable_p`` is returned so a report can state that limit rather than
leave a reader to infer that 0.25 means "no effect".
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np

__all__ = ["seed_level_test", "min_attainable_signflip_p"]


def min_attainable_signflip_p(n_seeds: int) -> float:
    """Smallest two-sided p the exact sign-flip test can return at ``n_seeds``."""
    if n_seeds < 1:
        return float("nan")
    return 2.0 / (2.0 ** n_seeds)


def seed_level_test(values: Sequence[float]) -> dict[str, Any]:
    """Formal inference on per-seed paired effects.

    ``values`` are the per-seed differences (candidate minus reference). The
    return carries both tests, the descriptive spread, and the floor on the
    permutation p so a caller cannot report 0.25 at three seeds as if it were
    evidence of absence.
    """
    from scipy import stats

    array = np.asarray(list(values), dtype=float)
    n = len(array)
    if n < 2:
        raise ValueError(f"need at least two seeds for a seed-level test, got {n}")

    observed = float(array.mean())

    # Primary: one-sample t-test against zero.
    t_stat, p_ttest = stats.ttest_1samp(array, 0.0)

    # Sensitivity: exact sign-flip permutation over all 2^n assignments.
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    means = (signs * array).mean(axis=1)
    p_signflip = float((np.abs(means) >= abs(observed) - 1e-12).mean())

    return {
        "mean": observed,
        "n_seeds": n,
        "seed_sd": float(np.std(array, ddof=1)),
        "sign_agreement": int(np.sum(np.sign(array) == np.sign(observed))),
        "t_stat": float(t_stat),
        "p_ttest": float(p_ttest),
        "p_signflip": p_signflip,
        "min_attainable_p": min_attainable_signflip_p(n),
        "per_seed": [float(v) for v in array],
    }
