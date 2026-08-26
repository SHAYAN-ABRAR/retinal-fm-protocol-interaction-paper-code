"""RETFound against its matched ImageNet control, frozen features both sides.

The question this settles
------------------------
A RETFound probe on its own cannot answer anything. Its numbers can only be set
beside the fine-tuned DenseNet121 LODO results, and that comparison confounds
two things at once -- a 304 M retinal ViT against a 7 M ImageNet CNN, *and* a
frozen linear probe against full fine-tuning. A gap could come from either.

So the control is the same DenseNet121, ImageNet weights, frozen, run through
an identical pipeline: same 1024-dimensional feature width, same per-dimension
standardisation fitted on source-training only, same linear head, same 200-epoch
schedule, same splits, same evaluation, same temperature scaling. The single
difference between the two sets of rows is which network produced the features.

Source and target are both reported, and that is the point
----------------------------------------------------------
A representation can fit the source domains well and transfer badly, and those
are different claims. Reporting only the target would hide which one is
happening; reporting both makes the source-to-target drop itself a measurable
quantity, and it is the quantity a domain-generalization paper is about.

Bars, as everywhere else in this project
----------------------------------------
A difference counts only if it exceeds the across-seed SD of the paired
per-seed differences AND its paired bootstrap CI excludes zero. Seed s of one
backbone is differenced against seed s of the other. Three seeds is a small
sample for an SD, which is exactly why the SD is a bar and not a footnote.

Runs from saved predictions. No GPU.

Usage:
    python analyse_linear_probe.py
    python analyse_linear_probe.py --n-bootstrap 10000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

REFERENCE = "retfound_cfp"
CONTROL = "densenet121"
BATCH_SIZE = 256
IMAGE_SIZE = 224
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
TARGETS = ["ddr", "aptos", "idrid"]
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

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    n_bootstrap = int(_take("--n-bootstrap", str(N_BOOTSTRAP)))
    outputs = project_root() / "outputs"
    qwk = METRIC_FUNCTIONS["qwk"]
    severe = METRIC_FUNCTIONS["severe_error_rate"]

    table = pd.read_csv(outputs / "tables" / "linear_probe_results.csv")

    def load(backbone: str, target: str, seed: int):
        experiment_id = make_experiment_id(
            protocol="lodo", sources=[d for d in ALL_DOMAINS if d != target],
            target=target, backbone=backbone,
            method=f"linprobe-b{BATCH_SIZE}", seed=seed,
        )
        return load_target_predictions(experiment_id, target, outputs_dir=outputs)

    def cell(backbone: str, target: str, column: str):
        rows = table[(table.backbone == backbone) & (table.target == target)]
        return rows.set_index("seed")[column]

    print("=" * 118)
    print("FROZEN LINEAR PROBE: RETFound (retinal, 304 M) vs DenseNet121 "
          "(ImageNet, 7 M) -- identical pipeline, features only differ")
    print("=" * 118)
    print("  delta = control - reference, per seed, then averaged. Positive "
          "means ImageNet features won.")
    print("  A difference must clear the across-seed SD and have a bootstrap "
          "CI excluding zero.\n")

    records = []
    for target in TARGETS:
        ref_t, con_t = cell(REFERENCE, target, "target_qwk"), cell(CONTROL, target, "target_qwk")
        ref_s, con_s = cell(REFERENCE, target, "source_qwk"), cell(CONTROL, target, "source_qwk")
        seeds = [s for s in SEEDS if s in ref_t.index and s in con_t.index]
        if not seeds:
            print(f"  {target}: NOT RUN")
            continue

        deltas = np.array([con_t[s] - ref_t[s] for s in seeds])
        delta_sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
        delta = float(deltas.mean())

        reference_run = load(REFERENCE, target, ANCHOR_SEED)
        control_run = load(CONTROL, target, ANCHOR_SEED)
        low = high = p_value = float("nan")
        if reference_run is not None and control_run is not None:
            boot = paired_bootstrap_difference(
                reference_run["y_true"], reference_run["y_pred"],
                control_run["y_pred"], metric="qwk", n_bootstrap=n_bootstrap)
            low, high, p_value = boot["ci_lower"], boot["ci_upper"], boot["p_value"]

        over = abs(delta) / delta_sd if delta_sd and delta_sd == delta_sd else float("nan")
        clears = over > 1.0 and not (low <= 0 <= high)
        verdict = ("ImageNet features better" if clears and delta > 0 else
                   "RETFound features better" if clears and delta < 0 else
                   "within seed noise")

        records.append({
            "target": target, "n_seeds": len(seeds),
            "retfound_source": float(ref_s[seeds].mean()),
            "control_source": float(con_s[seeds].mean()),
            "retfound_target": float(ref_t[seeds].mean()),
            "control_target": float(con_t[seeds].mean()),
            "delta_target_qwk": delta, "delta_sd": delta_sd, "delta_over_sd": over,
            "ci_lower": low, "ci_upper": high, "p_value": p_value,
            "retfound_drop": float(ref_s[seeds].mean() - ref_t[seeds].mean()),
            "control_drop": float(con_s[seeds].mean() - con_t[seeds].mean()),
            "retfound_severe": float(cell(REFERENCE, target, "target_severe")[seeds].mean()),
            "control_severe": float(cell(CONTROL, target, "target_severe")[seeds].mean()),
            "verdict": verdict,
        })

    frame = pd.DataFrame(records)
    if frame.empty:
        print("nothing to compare")
        return

    show = frame[["target", "n_seeds", "retfound_target", "control_target",
                  "delta_target_qwk", "delta_sd", "delta_over_sd",
                  "ci_lower", "ci_upper", "p_value", "verdict"]]
    print("--- target (cross-domain) ---")
    print(show.round(4).to_string(index=False))

    print("\n--- source validation: does the representation fit the training "
          "domains at all? ---")
    fit = frame[["target", "retfound_source", "control_source",
                 "retfound_target", "control_target",
                 "retfound_drop", "control_drop"]]
    print(fit.round(4).to_string(index=False))

    print("\n--- severe errors (|error| >= 2 grades) ---")
    print(frame[["target", "retfound_severe", "control_severe"]].round(4).to_string(index=False))

    wins = int((frame.retfound_source > frame.control_source).sum())
    print(f"\nRETFound fits the SOURCE better on {wins} of {len(frame)} targets "
          f"(mean +{(frame.retfound_source - frame.control_source).mean():.4f} QWK).")
    beats = int((frame.control_target > frame.retfound_target).sum())
    print(f"ImageNet features score higher on the TARGET on {beats} of "
          f"{len(frame)} targets "
          f"(mean +{(frame.control_target - frame.retfound_target).mean():.4f} QWK).")
    established = frame[frame.verdict != "within seed noise"]
    print(f"Clearing both bars: "
          f"{', '.join(established.target) if len(established) else 'none'}.")

    destination = outputs / "tables" / "linear_probe_comparison.csv"
    frame.to_csv(destination, index=False)
    print(f"\nsaved -> {destination}")


if __name__ == "__main__":
    main()
