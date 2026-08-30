"""Aggregate the four leave-one-domain-out runs into the headline paper result.

This is the experiment the whole project exists to report: for each of the four
domains in turn, train on the other three and test on the held-out one. The
question is not only "how far does accuracy fall" but "does the model *know*
it has fallen" -- hence the calibration columns alongside the ordinal ones.

Three things are reported per target, because they answer different questions:

*The generalization gap* (source validation minus unseen target) is the headline
number, but on its own it conflates two causes. A model can score badly on a new
domain because the images look different (covariate shift) or because the grade
mix is different (prior shift). The two demand different fixes, so the label
distribution of each target is printed next to its score rather than left for a
reader to assume.

*Bootstrap CIs* resample the test images and say how much of the gap is sampling
noise. These vary enormously across targets here -- IDRiD contributes 507 test
images and EyePACS 35,108 -- so a gap that is decisive on one target may be
unresolvable on another. The n is printed with every interval for that reason.

*Test-set size and class support* determine whether a metric is meaningful at
all. QWK on a target with an absent grade is not comparable to QWK on one
without, and per-target class support is printed so that is visible.

Targets that have not been run are reported as NOT RUN. They are never dropped
silently from the table, because an absent row and a bad row look identical once
a mean is taken over whatever happens to be present.

Usage:
    python analyse_lodo.py                      # ERM, seed 42
    python analyse_lodo.py --method mixstyle
    python analyse_lodo.py --seed 1
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
# The reference input configuration for every result in the paper.
REFERENCE_BATCH_SIZE = 32
BATCH_SIZE = 32
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
BOOTSTRAP_METRICS = ["qwk", "f1_macro", "ece", "mae_grade"]
DOMAIN_INDEX = {"ddr": 0, "aptos": 1, "idrid": 2, "eyepacs": 3}


def _experiment_id(target: str, method: str, seed: int) -> str:
    from src.utils.registry import make_experiment_id

    sources = [d for d in ALL_TARGETS if d != target]
    return make_experiment_id(
        protocol="lodo", sources=sources, target=target,
        backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
    )


def load_predictions(target: str, method: str, seed: int, split: str):
    """Return the saved per-image predictions, or None if that run has not happened."""
    import pandas as pd

    from src.utils.io import project_root

    suffix = f"target_test[{target}]" if split == "target" else "source_val"
    path = (
        project_root() / "outputs" / "predictions"
        / f"{_experiment_id(target, method, seed)}__{suffix}_predictions.csv"
    )
    return pd.read_csv(path) if path.exists() else None


def _probabilities(frame):
    import numpy as np

    columns = [c for c in frame.columns if c.startswith("probability_grade_")]
    columns.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    return np.asarray(frame[columns].to_numpy(), dtype=np.float64)


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import bootstrap_all_metrics
    from src.utils.io import project_root

    arguments = sys.argv[1:]

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    method = _take("--method", "erm")
    seed = int(_take("--seed", "42"))
    n_bootstrap = int(_take("--n-bootstrap", "2000"))

    root = project_root()
    outputs = root / "outputs"
    results_path = outputs / "tables" / "lodo_results.csv"

    print(f"leave-one-domain-out analysis: method={method}, seed={seed}, "
          f"backbone={BACKBONE}, batch={BATCH_SIZE}")

    if not results_path.exists():
        print(f"\nNOT RUN -- {results_path} does not exist. "
              f"Run `python run_lodo.py --method {method} --seeds {seed}` first.")
        return

    results = pd.read_csv(results_path)
    results = results[(results["method"] == method) & (results["seed"] == seed)]
    # lodo_results.csv now also holds domain-balanced-sampler runs, whose
    # "method" column reads exactly the same as their ordinary-sampler
    # counterparts -- only the experiment id distinguishes them. Without this
    # filter an ERM three-seed summary would silently average six rows from two
    # different samplers and report the spread between them as seed noise.
    if "domain_balanced" in results.columns:
        results = results[~results["domain_balanced"].fillna(False).astype(bool)]
    # lodo_results.csv holds every backbone that has been run. Filtering on
    # method and seed alone would pool DenseNet121 with ConvNeXt-Tiny; rows
    # written before the column existed are DenseNet121.
    if "backbone" in results.columns:
        results = results[results["backbone"] == BACKBONE]
    # Third column of the same kind, after sampler and backbone: the Q1
    # batch-size controls are a second 224 px ERM configuration, and pooling
    # them reports the spread between configurations as seed noise.
    if "batch_size" in results.columns:
        results = results[results["batch_size"].fillna(REFERENCE_BATCH_SIZE)
                          .astype(int) == REFERENCE_BATCH_SIZE]

    # ---------------------------------------------------------------- headline
    rows = []
    missing = []
    for target in ALL_TARGETS:
        match = results[results["target"] == target]
        if match.empty:
            missing.append(target)
            rows.append({"target": target, "status": "NOT RUN"})
            continue
        record = match.iloc[-1].to_dict()
        record["status"] = "COMPLETE"
        # One sign convention throughout: delta = target - source, so a
        # negative delta always means "worse on the unseen domain", for every
        # metric. Mixing conventions between a table and its summary line is how
        # a degradation gets read as an improvement.
        record["delta_qwk"] = record["target_qwk"] - record["source_qwk"]
        record["delta_ece"] = record["target_ece"] - record["source_ece"]
        rows.append(record)

    print(f"\n{'=' * 100}\nHEADLINE: source validation vs unseen target\n{'=' * 100}")
    header = (f"{'target':<9} {'domain':<7} {'n_train':>8} {'n_test':>7} "
              f"{'src QWK':>8} {'tgt QWK':>8} {'d QWK':>8} "
              f"{'src ECE':>8} {'tgt ECE':>8} {'T-scaled':>9} {'severe':>7}")
    print(header)
    print("-" * len(header))
    for row in rows:
        target = row["target"]
        index = DOMAIN_INDEX[target]
        if row["status"] == "NOT RUN":
            print(f"{target:<9} {index:<7} {'NOT RUN':>8} {'--':>7} {'--':>8} {'--':>8} "
                  f"{'--':>8} {'--':>8} {'--':>8} {'--':>9} {'--':>7}")
            continue
        scaled = row.get("target_ece_scaled")
        scaled_text = f"{scaled:.4f}" if pd.notna(scaled) else "--"
        print(f"{target:<9} {index:<7} {int(row['n_train']):>8} {int(row['n_test']):>7} "
              f"{row['source_qwk']:>8.4f} {row['target_qwk']:>8.4f} "
              f"{row['delta_qwk']:>+8.4f} "
              f"{row['source_ece']:>8.4f} {row['target_ece']:>8.4f} "
              f"{scaled_text:>9} {row['target_severe']:>7.4f}")

    complete = [r for r in rows if r["status"] == "COMPLETE"]
    if missing:
        print(f"\n!! {len(missing)} of 4 targets NOT RUN: {', '.join(missing)}. "
              "No mean is reported over a partial matrix -- it would not be the "
              "quantity the paper claims to report.")
    elif complete:
        deltas = [r["delta_qwk"] for r in complete]
        worst = min(complete, key=lambda r: r["delta_qwk"])
        print(f"\nmean QWK change on the unseen domain: {np.mean(deltas):+.4f} "
              f"(range {min(deltas):+.4f} to {max(deltas):+.4f}; "
              "negative = worse than source validation)")
        print(f"  !! dominated by {worst['target']} ({worst['delta_qwk']:+.4f}). "
              "Per-target values span an order of magnitude, so this mean "
              "describes none of them -- quote the per-target rows instead.")

    # ------------------------------------------------------- per-target detail
    ci_rows = []
    support_rows = []
    for row in complete:
        target = row["target"]
        frame = load_predictions(target, method, seed, "target")
        if frame is None:
            print(f"\n!! {target}: result row exists but predictions file is missing; "
                  "cannot compute intervals or class support for it")
            continue

        y_true = frame["true_grade"].to_numpy()
        y_pred = frame["predicted_grade"].to_numpy()
        probabilities = _probabilities(frame)

        print(f"\n{'-' * 100}\n{target.upper()} held out  (n={len(frame)}, "
              f"sources={[d for d in ALL_TARGETS if d != target]})\n{'-' * 100}")

        support = pd.Series(y_true).value_counts().reindex(range(5), fill_value=0)
        predicted = pd.Series(y_pred).value_counts().reindex(range(5), fill_value=0)
        absent = [int(g) for g in range(5) if support[g] == 0]
        print("  true support:      " + "  ".join(
            f"g{g}={support[g]:>6d} ({support[g] / len(frame):>5.1%})" for g in range(5)))
        print("  predicted:         " + "  ".join(
            f"g{g}={predicted[g]:>6d} ({predicted[g] / len(frame):>5.1%})" for g in range(5)))
        if absent:
            print(f"  !! grade(s) {absent} absent from this target. QWK and macro F1 "
                  "are computed over the grades that are present and are NOT "
                  "comparable across targets with different support.")
        support_rows.append({
            "target": target, "n_test": len(frame),
            **{f"true_g{g}": int(support[g]) for g in range(5)},
            **{f"pred_g{g}": int(predicted[g]) for g in range(5)},
            "absent_grades": ";".join(str(g) for g in absent) if absent else "",
        })

        intervals = bootstrap_all_metrics(
            y_true, y_pred, probabilities,
            metrics=BOOTSTRAP_METRICS, n_bootstrap=n_bootstrap, seed=seed,
        )
        for name, result in intervals.items():
            note = ""
            if result.n_failed:
                note = (f"  !! {result.n_failed}/{n_bootstrap} resamples undefined "
                        "-- interval is over the remainder")
            print(f"  {name:<10} {result.format()}   (n={result.n_samples}){note}")
            ci_rows.append({"target": target, "method": method, "seed": seed,
                            **result.as_dict()})

    # ------------------------------------------------------------------ output
    tables = outputs / "tables"
    if ci_rows:
        path = tables / f"lodo_{method}_s{seed}_bootstrap_ci.csv"
        pd.DataFrame(ci_rows).to_csv(path, index=False)
        print(f"\nsaved -> {path}")
    if support_rows:
        path = tables / f"lodo_{method}_s{seed}_class_support.csv"
        pd.DataFrame(support_rows).to_csv(path, index=False)
        print(f"saved -> {path}")
    if complete:
        path = tables / f"lodo_{method}_s{seed}_headline.csv"
        pd.DataFrame(complete).to_csv(path, index=False)
        print(f"saved -> {path}")

    # ----------------------------------------------------------------- figures
    try:
        from src.utils.registry import load_registry
        from src.visualization.domain_figures import generate_domain_figures

        registry = load_registry(outputs / "experiment_registry.csv")

        # Restrict to exactly the runs analysed above. Other experiments share a
        # matrix cell legitimately -- the 2-source Stage-C run also trained on
        # DDR and tested on IDRiD -- but a cell holding the mean of a 2-source
        # and a 3-source run is neither experiment, and the paper figure must be
        # the four full LODO runs it is captioned as.
        wanted = {_experiment_id(r["target"], method, seed) for r in complete}
        if "experiment_id" in registry.columns:
            registry = registry[registry["experiment_id"].isin(wanted)]
        print(f"\nfigures built from {len(registry)} run(s): "
              f"{sorted(registry['experiment_id'])}")

        written = generate_domain_figures(
            registry, outputs / "figures", lodo_rows=complete,
            method=method, seed=seed,
        )
        for figure in written:
            print(f"saved -> {figure}")
    except Exception as exc:  # noqa: BLE001 - figures must not lose the numbers
        print(f"!! figure generation failed ({type(exc).__name__}: {exc}); "
              "the tables above are unaffected")


if __name__ == "__main__":
    main()
