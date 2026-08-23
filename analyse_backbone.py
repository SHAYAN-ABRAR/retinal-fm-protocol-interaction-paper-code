"""ConvNeXt-Tiny against DenseNet121 on the leave-one-domain-out matrix.

Phase 8 found that ConvNeXt grades better but calibrates worse -- accuracy and
reliability moving in opposite directions, which is the dissociation this
project is named after. That finding was produced ad hoc and existed only as
``backbone_comparison_s42.csv``: a table with no script behind it, from a single
seed, whose significance bar was borrowed from DenseNet's across-seed SD because
ConvNeXt had no SD of its own.

This script replaces that. It is the generator for the Phase 8 table, and it
reports the comparison at whatever seed count is on disk.

Protocol
--------
* **Paired seeds.** ConvNeXt at seed *s* is differenced against DenseNet at
  seed *s*, so the SD measures the variation of the *difference*. Differencing
  the two means instead would add two independent noise sources together.
* **Matched images.** Both backbones are checked to have scored the same test
  images, with the same labels, before anything is differenced.
* **Two bars.** Real means the effect exceeds the across-seed SD *and* its
  paired bootstrap interval excludes zero.
* **The SD is labelled.** With fewer than two paired seeds there is no SD to
  compute, and the script falls back to DenseNet's -- but it says so, in a
  ``sd_source`` column, on every row. A borrowed bar that is not labelled as
  borrowed is what made the original table misleading.

Calibration is compared after temperature scaling as well as before, because
before-scaling ECE mostly measures how badly each backbone overfits its own
confidence, and the deployable question is what survives the correction. Every
temperature is the one fitted on source validation during the run itself; none
is refitted here, and none has seen the target domain.

Runs from saved predictions. No GPU.

Usage:
    python analyse_backbone.py
    python analyse_backbone.py --n-bootstrap 10000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

REFERENCE_BACKBONE = "densenet121"
CANDIDATE_BACKBONE = "convnext_tiny"
BATCH_SIZE = 32
IMAGE_SIZE = 224
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
SEEDS = [42, 1, 2]
ANCHOR_SEED = 42
N_BOOTSTRAP = 5000


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import METRIC_FUNCTIONS, paired_bootstrap_difference
    from src.utils.io import project_root
    from src.utils.registry import make_experiment_id
    from src.visualization.calibration_figures import load_target_predictions

    arguments = sys.argv[1:]
    n_bootstrap = N_BOOTSTRAP
    if "--n-bootstrap" in arguments:
        n_bootstrap = int(arguments[arguments.index("--n-bootstrap") + 1])

    outputs = project_root() / "outputs"
    severe = METRIC_FUNCTIONS["severe_error_rate"]
    qwk = METRIC_FUNCTIONS["qwk"]
    ece = METRIC_FUNCTIONS["ece"]

    def load(target: str, backbone: str, seed: int):
        experiment_id = make_experiment_id(
            protocol="lodo", sources=[d for d in ALL_TARGETS if d != target],
            target=target, backbone=backbone, method=f"erm-b{BATCH_SIZE}", seed=seed,
        )
        return load_target_predictions(experiment_id, target, outputs_dir=outputs)

    rows = []
    for target in ALL_TARGETS:
        reference_by_seed, candidate_by_seed = {}, {}
        for seed in SEEDS:
            reference = load(target, REFERENCE_BACKBONE, seed)
            candidate = load(target, CANDIDATE_BACKBONE, seed)
            if reference is not None:
                reference_by_seed[seed] = reference
            if candidate is not None:
                candidate_by_seed[seed] = candidate

        paired_seeds = sorted(set(reference_by_seed) & set(candidate_by_seed))
        if ANCHOR_SEED not in paired_seeds:
            print(f"  {target}: no seed-{ANCHOR_SEED} pair -- skipped")
            continue

        # Both backbones see the same LODO test split, so these checks are
        # expected to pass trivially. They are made anyway: an assumption that
        # holds is free to check, and one that quietly stopped holding after a
        # split change would otherwise difference two different image sets.
        anchor = reference_by_seed[ANCHOR_SEED]
        n_test = len(anchor["y_true"])
        for seed in paired_seeds:
            for run in (reference_by_seed[seed], candidate_by_seed[seed]):
                assert len(run["y_true"]) == n_test, (
                    f"{target} seed {seed}: {run['experiment_id']} scored "
                    f"{len(run['y_true'])} images, expected {n_test}")
                assert (run["y_true"] == anchor["y_true"]).all(), (
                    f"{target} seed {seed}: labels disagree between runs")

        def collect(metric, use_probabilities=None):
            """Per-seed reference, candidate and difference for one metric."""
            reference_values, candidate_values, differences = [], [], []
            for seed in paired_seeds:
                pair = []
                for run in (reference_by_seed[seed], candidate_by_seed[seed]):
                    probabilities = run[use_probabilities] if use_probabilities else None
                    pair.append(metric(run["y_true"], run["y_pred"], probabilities))
                reference_values.append(pair[0])
                candidate_values.append(pair[1])
                differences.append(pair[1] - pair[0])
            return (np.array(reference_values), np.array(candidate_values),
                    np.array(differences))

        def sd(values):
            return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

        qwk_reference, qwk_candidate, qwk_delta = collect(qwk)
        severe_reference, severe_candidate, severe_delta = collect(severe)
        ece_reference, ece_candidate, _ = collect(ece, "probabilities")
        ece_t_reference, ece_t_candidate, ece_t_delta = collect(ece, "scaled")

        # Bar 1. With one paired seed there is nothing to take an SD over, so
        # DenseNet's own across-seed SD stands in -- the same proxy the original
        # table used silently. Recorded in sd_source either way.
        if len(paired_seeds) > 1:
            qwk_bar = sd(qwk_delta)
            sd_source = f"paired ({len(paired_seeds)} seeds)"
        else:
            reference_only = [reference_by_seed[s] for s in sorted(reference_by_seed)]
            qwk_bar = sd([qwk(r["y_true"], r["y_pred"], None) for r in reference_only])
            sd_source = f"proxy: {REFERENCE_BACKBONE} SD over {len(reference_only)} seeds"

        # Bar 2, on the anchor seed -- the convention every other analysis in
        # this project follows, so the intervals stay comparable across phases.
        reference_anchor = reference_by_seed[ANCHOR_SEED]
        candidate_anchor = candidate_by_seed[ANCHOR_SEED]
        qwk_test = paired_bootstrap_difference(
            reference_anchor["y_true"], reference_anchor["y_pred"],
            candidate_anchor["y_pred"],
            metric="qwk", n_bootstrap=n_bootstrap, seed=ANCHOR_SEED,
        )
        ece_test = paired_bootstrap_difference(
            reference_anchor["y_true"], reference_anchor["y_pred"],
            candidate_anchor["y_pred"],
            probabilities_a=reference_anchor["scaled"],
            probabilities_b=candidate_anchor["scaled"],
            metric="ece", n_bootstrap=n_bootstrap, seed=ANCHOR_SEED,
        )

        def verdict(delta, bar, excludes_zero, better_when_positive=True):
            if not np.isfinite(bar):
                return "no SD available"
            if abs(delta) <= bar:
                return "within seed noise"
            if not excludes_zero:
                return "CI spans zero"
            improved = (delta > 0) if better_when_positive else (delta < 0)
            return "REAL (better)" if improved else "REAL (worse)"

        ece_t_bar = sd(ece_t_delta) if len(paired_seeds) > 1 else float("nan")
        delta_qwk = float(qwk_delta.mean())
        delta_ece_t = float(ece_t_delta.mean())

        rows.append({
            "target": target,
            "n_test": n_test,
            "n_paired_seeds": len(paired_seeds),
            "seeds": ",".join(str(s) for s in paired_seeds),

            "densenet_qwk": float(qwk_reference.mean()),
            "densenet_qwk_sd": sd(qwk_reference),
            "convnext_qwk": float(qwk_candidate.mean()),
            "convnext_qwk_sd": sd(qwk_candidate),
            "delta_qwk": delta_qwk,
            "delta_qwk_sd": qwk_bar,
            "sd_source": sd_source,
            "delta_over_sd": (float(abs(delta_qwk) / qwk_bar)
                              if np.isfinite(qwk_bar) and qwk_bar > 0 else float("nan")),
            "qwk_ci_lower": qwk_test["ci_lower"],
            "qwk_ci_upper": qwk_test["ci_upper"],
            "qwk_p": qwk_test["p_value"],
            "qwk_excludes_zero": bool(qwk_test["significant"]),
            "qwk_verdict": verdict(delta_qwk, qwk_bar, bool(qwk_test["significant"])),

            "densenet_ece": float(ece_reference.mean()),
            "convnext_ece": float(ece_candidate.mean()),
            "densenet_ece_scaled": float(ece_t_reference.mean()),
            "densenet_ece_scaled_sd": sd(ece_t_reference),
            "convnext_ece_scaled": float(ece_t_candidate.mean()),
            "convnext_ece_scaled_sd": sd(ece_t_candidate),
            "delta_ece_scaled": delta_ece_t,
            "delta_ece_scaled_sd": ece_t_bar,
            "ece_ci_lower": ece_test["ci_lower"],
            "ece_ci_upper": ece_test["ci_upper"],
            "ece_p": ece_test["p_value"],
            "ece_excludes_zero": bool(ece_test["significant"]),
            # Lower ECE is better, so a negative difference is the improvement.
            "ece_verdict": verdict(delta_ece_t, ece_t_bar,
                                   bool(ece_test["significant"]),
                                   better_when_positive=False),

            "densenet_severe": float(severe_reference.mean()),
            "convnext_severe": float(severe_candidate.mean()),
            "delta_severe": float(severe_delta.mean()),
            "delta_severe_sd": sd(severe_delta),

            "densenet_temperature": reference_anchor["temperature"],
            "convnext_temperature": candidate_anchor["temperature"],
        })

    if not rows:
        print("NOT RUN -- no matched backbone pairs found")
        return

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 260)

    print("=" * 120)
    print("BACKBONE: ConvNeXt-Tiny vs DenseNet121, leave-one-domain-out, "
          f"{IMAGE_SIZE} px, matched images, paired seeds")
    print("=" * 120)
    print("  delta = ConvNeXt - DenseNet. Positive QWK favours ConvNeXt; "
          "positive ECE favours DenseNet, since lower ECE is better.")
    print(f"  Bootstrap interval is on seed {ANCHOR_SEED}; the SD is across "
          "paired seeds where more than one exists.\n")

    print(table[[
        "target", "n_test", "n_paired_seeds", "densenet_qwk", "convnext_qwk",
        "delta_qwk", "delta_qwk_sd", "delta_over_sd", "qwk_ci_lower",
        "qwk_ci_upper", "qwk_verdict",
    ]].round(4).to_string(index=False))

    print("\nCalibration after temperature scaling, fitted on source validation:\n")
    print(table[[
        "target", "densenet_ece_scaled", "convnext_ece_scaled",
        "delta_ece_scaled", "delta_ece_scaled_sd", "ece_ci_lower",
        "ece_ci_upper", "ece_verdict",
    ]].round(4).to_string(index=False))

    borrowed = table[table["sd_source"].str.startswith("proxy")]
    if len(borrowed):
        print(f"\n  NOTE: {len(borrowed)} of {len(table)} rows use a borrowed SD "
              f"({borrowed['sd_source'].iloc[0]}).")
        print("  Those verdicts are provisional until ConvNeXt has two or more "
              "seeds of its own.")

    path = outputs / "tables" / "backbone_comparison.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved -> {path}")

    # The superseded single-seed table stays on disk as a record. Where the two
    # overlap they must agree; a silent divergence between two tables describing
    # the same runs is a failure this project has already had once.
    legacy_path = outputs / "tables" / "backbone_comparison_s42.csv"
    if legacy_path.exists() and ANCHOR_SEED == 42:
        legacy = pd.read_csv(legacy_path).set_index("target")
        for _, row in table.iterrows():
            if row["target"] not in legacy.index:
                continue
            old = legacy.loc[row["target"]]
            for new_column, old_column in [("qwk_ci_lower", "ci_lower"),
                                           ("qwk_ci_upper", "ci_upper")]:
                if abs(float(row[new_column]) - float(old[old_column])) > 5e-3:
                    raise SystemExit(
                        f"{row['target']}: {new_column}={row[new_column]:.4f} "
                        f"disagrees with {legacy_path.name} "
                        f"{old_column}={old[old_column]:.4f}")
        print(f"  checked against {legacy_path.name}: seed-42 intervals agree")


if __name__ == "__main__":
    main()
