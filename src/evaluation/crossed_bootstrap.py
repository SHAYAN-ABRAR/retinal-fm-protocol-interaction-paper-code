"""Crossed seed x case bootstrap for paired model comparisons.

Why the existing inference was not enough
-----------------------------------------
Two sources of uncertainty act on every comparison in this project, and the
tooling only ever measured one at a time:

* ``paired_bootstrap_difference`` resamples **test cases within one seed**. It
  answers "did this trained model beat that trained model on this test set",
  which is a real question and not the one a reader assumes. It is blind to the
  fact that another seed produces another model.
* The "delta > across-seed SD" rule sees seed variance but has no calibrated
  error rate. It is a heuristic, and three results in this project cleared it at
  three seeds and failed at five.

Reported alone, the first is anti-conservative: DDR and APTOS partial
fine-tuning both carry bootstrap intervals excluding zero on their anchor seed
while being nulls across seeds.

Crossed, not nested
-------------------
Every training seed is evaluated on the *same* test set. Cases are therefore
**crossed** with seeds, not nested inside them, and the resampling has to
respect that:

1. resample seeds with replacement;
2. resample cases with replacement **once per iteration**;
3. apply that one case sample to every selected seed and to *both* models;
4. recompute each metric per seed and model on that case sample -- QWK and ECE
   are non-linear, so they cannot be averaged from per-case values;
5. take the paired model difference within each seed;
6. average the paired differences over the sampled seeds.

Step 3 is what makes it crossed. Drawing independent case samples per seed would
break the pairing that gives the comparison its power, and would inflate the
interval. Step 4 is why this cannot be vectorised into a simple mean.

What it does not do
-------------------
With a handful of seeds the seed-level resample is coarse -- five seeds admit
only so many distinct multisets -- so the interval is honest about seed
uncertainty but not precise about it. That is a property of the design, not of
the estimator, and the seed count is reported beside every interval.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["crossed_bootstrap_difference", "holm_adjust"]


def crossed_bootstrap_difference(
    y_true: np.ndarray,
    reference: Mapping[int, np.ndarray],
    candidate: Mapping[int, np.ndarray],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired difference between two models over seeds and cases jointly.

    ``reference`` and ``candidate`` map a training seed to that seed's
    predictions on the shared test set. Only seeds present in both are used, so
    the comparison stays paired.

    ``metric`` takes ``(y_true, y_pred)`` and returns a scalar. The difference
    reported is ``candidate - reference``.
    """
    shared = sorted(set(reference) & set(candidate))
    if len(shared) < 2:
        raise ValueError(
            f"need at least two paired seeds, got {len(shared)}: {shared}")
    y_true = np.asarray(y_true)
    n_cases = len(y_true)
    for s in shared:
        if len(reference[s]) != n_cases or len(candidate[s]) != n_cases:
            raise ValueError(
                f"seed {s}: predictions must align with y_true ({n_cases})")

    per_seed = {s: float(metric(y_true, candidate[s]) - metric(y_true, reference[s]))
                for s in shared}
    observed = float(np.mean(list(per_seed.values())))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        picked = rng.choice(shared, size=len(shared), replace=True)
        # ONE case sample per iteration, shared by every seed and both models.
        cases = rng.integers(0, n_cases, size=n_cases)
        truth = y_true[cases]
        deltas = [float(metric(truth, candidate[s][cases])
                        - metric(truth, reference[s][cases])) for s in picked]
        draws[i] = float(np.mean(deltas))

    lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided achieved significance level, the same convention the project's
    # existing bootstrap uses.
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())

    values = np.array(list(per_seed.values()))
    return {
        "difference": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(min(1.0, p)),
        "n_seeds": len(shared),
        "seeds": shared,
        "per_seed_difference": per_seed,
        "seed_sd": float(np.std(values, ddof=1)),
        "sign_agreement": int(np.sum(np.sign(values) == np.sign(observed))),
        "n_bootstrap": n_bootstrap,
        # Kept as a robustness diagnostic only. It is deliberately NOT the
        # significance test; the interval above is.
        "exceeds_seed_sd": bool(abs(observed) > np.std(values, ddof=1)),
    }


def holm_adjust(p_values: Sequence[float], labels: Sequence[str] | None = None):
    """Holm-Bonferroni step-down adjustment.

    Returned in the input order. Note the direction of conservatism: correcting
    makes differences harder to detect, so for a family of hypotheses where the
    claim is that no difference exists, the *uncorrected* p is the more
    demanding test and should be reported as primary.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    if labels is None:
        return adjusted.tolist()
    return dict(zip(labels, adjusted.tolist()))
