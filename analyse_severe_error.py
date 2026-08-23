"""The severe-error rate: in-domain against cross-domain, on matched images.

A severe error is a grade misread by two or more steps. It is the error that
matters clinically -- a two-step miss is what sends a referable patient home --
and it is the metric this project's framing rests on, yet until now it existed
only as a column in a results table with no interval and no matched comparison.

Everything here follows the protocol Phase 6 established for QWK:

* **Matched images.** The cross-domain model's predictions are restricted to
  the in-domain model's held-out split before differencing. Scoring the two on
  different image sets would credit the cross-domain model for images the
  in-domain model trained on.
* **Two bars.** An effect is real only when it exceeds the across-seed SD *and*
  its paired bootstrap interval excludes zero. The bars fail independently and
  in both directions, so both are required.

Runs from saved predictions. No GPU.

Usage:
    python analyse_severe_error.py
    python analyse_severe_error.py --n-bootstrap 10000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
SEEDS = [42, 1, 2]
N_BOOTSTRAP = 5000


def _load(path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import METRIC_FUNCTIONS, paired_bootstrap_difference
    from src.utils.io import project_root
    from src.utils.registry import make_experiment_id

    arguments = sys.argv[1:]
    n_bootstrap = N_BOOTSTRAP
    if "--n-bootstrap" in arguments:
        n_bootstrap = int(arguments[arguments.index("--n-bootstrap") + 1])

    severe = METRIC_FUNCTIONS["severe_error_rate"]
    outputs = project_root() / "outputs"
    predictions = outputs / "predictions"

    rows = []
    for target in ALL_TARGETS:

        def _in_domain(seed: int):
            experiment_id = make_experiment_id(
                protocol="in_domain", sources=[target], target=target,
                backbone=BACKBONE, method=f"erm-b{BATCH_SIZE}", seed=seed,
            )
            return _load(predictions / f"{experiment_id}__target_test[{target}]_predictions.csv")

        # Seeds are paired: the LODO model at seed s is differenced against the
        # in-domain model at seed s, so the SD below is the variation of the
        # difference rather than two independent noise sources added together.
        # Falls back to the seed-42 reference for any seed whose in-domain run
        # does not exist, which is what this did before seeds 1 and 2 were run.
        fallback = _in_domain(42)
        in_by_seed, lodo_by_seed = {}, {}
        for seed in SEEDS:
            lodo_id = make_experiment_id(
                protocol="lodo", sources=[d for d in ALL_TARGETS if d != target],
                target=target, backbone=BACKBONE,
                method=f"erm-b{BATCH_SIZE}", seed=seed,
            )
            frame = _load(predictions / f"{lodo_id}__target_test[{target}]_predictions.csv")
            reference = _in_domain(seed)
            if reference is None:
                reference = fallback
            if frame is not None and reference is not None:
                lodo_by_seed[seed] = frame
                in_by_seed[seed] = reference
        if not lodo_by_seed:
            print(f"  {target}: no matched prediction pair -- skipped")
            continue

        # Images every model in the comparison scored, so the row has one n.
        shared = None
        for seed in lodo_by_seed:
            ids = set(lodo_by_seed[seed]["image_id"]) & set(in_by_seed[seed]["image_id"])
            shared = ids if shared is None else shared & ids
        shared = sorted(shared)

        in_values, lodo_values, deltas = [], [], []
        for seed in sorted(lodo_by_seed):
            lodo_matched = (lodo_by_seed[seed][lodo_by_seed[seed]["image_id"].isin(shared)]
                            .sort_values("image_id"))
            in_matched = (in_by_seed[seed][in_by_seed[seed]["image_id"].isin(shared)]
                          .sort_values("image_id"))
            y_true = in_matched["true_grade"].to_numpy()
            assert (lodo_matched["true_grade"].to_numpy() == y_true).all(), \
                f"{target} seed {seed}: labels disagree between runs"
            in_value = severe(y_true, in_matched["predicted_grade"].to_numpy(), None)
            lodo_value = severe(y_true, lodo_matched["predicted_grade"].to_numpy(), None)
            in_values.append(in_value)
            lodo_values.append(lodo_value)
            deltas.append(lodo_value - in_value)

        in_severe = float(np.mean(in_values))
        in_sd = float(np.std(in_values, ddof=1)) if len(set(in_values)) > 1 else float("nan")
        lodo_mean = float(np.mean(lodo_values))
        lodo_sd = float(np.std(lodo_values, ddof=1)) if len(lodo_values) > 1 else float("nan")
        delta = float(np.mean(deltas))          # positive = cross-domain is worse
        delta_sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")

        # Bootstrap on seed 42, matching how the QWK deployment cost is reported.
        anchor = (in_by_seed[42][in_by_seed[42]["image_id"].isin(shared)]
                  .sort_values("image_id"))
        reference = (lodo_by_seed[42][lodo_by_seed[42]["image_id"].isin(shared)]
                     .sort_values("image_id"))
        result = paired_bootstrap_difference(
            anchor["true_grade"].to_numpy(), anchor["predicted_grade"].to_numpy(),
            reference["predicted_grade"].to_numpy(),
            metric="severe_error_rate", n_bootstrap=n_bootstrap, seed=42,
        )

        exceeds_sd = bool(abs(delta) > delta_sd) if np.isfinite(delta_sd) else False
        excludes_zero = bool(result["significant"])
        verdict = ("REAL (both bars)" if exceeds_sd and excludes_zero
                   else "CI spans zero" if exceeds_sd
                   else "within seed noise")

        rows.append({
            "target": target, "n_test": len(shared),
            "in_domain_severe": in_severe, "in_domain_severe_sd": in_sd,
            "lodo_severe_mean": lodo_mean, "lodo_severe_sd": lodo_sd,
            "n_seeds": len(deltas),
            "delta_severe": delta, "delta_severe_sd": delta_sd,
            "relative_increase": delta / in_severe if in_severe > 0 else float("nan"),
            "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"],
            "p_value": result["p_value"],
            "exceeds_seed_sd": exceeds_sd, "ci_excludes_zero": excludes_zero,
            "verdict": verdict,
        })

    if not rows:
        print("NOT RUN -- no matched prediction pairs found")
        return

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print("=" * 118)
    print("SEVERE-ERROR RATE (|error| >= 2 grades), in-domain vs cross-domain, matched images")
    print("=" * 118)
    print("  Bootstrap interval is on seed 42; the SD is across "
          f"{table['n_seeds'].max()} seeds. Positive delta = cross-domain is worse.\n")
    print(table.round(4).to_string(index=False))

    path = outputs / "tables" / "severe_error_comparison.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved -> {path}")

    real = table[table["verdict"] == "REAL (both bars)"]
    if len(real):
        print("\nEstablished under both bars:")
        for _, r in real.iterrows():
            print(f"  {r['target']:8s} {r['in_domain_severe']:.4f} -> "
                  f"{r['lodo_severe_mean']:.4f}  ({r['relative_increase']:+.0%} relative, "
                  f"n={int(r['n_test']):,})")
    other = table[table["verdict"] != "REAL (both bars)"]
    if len(other):
        print("\nNOT established -- do not report these as effects:")
        for _, r in other.iterrows():
            print(f"  {r['target']:8s} delta {r['delta_severe']:+.4f}  "
                  f"CI [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  {r['verdict']}")


if __name__ == "__main__":
    main()
