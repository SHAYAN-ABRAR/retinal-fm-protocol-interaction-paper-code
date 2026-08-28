"""RETFound against its own ImageNet initialisation, both fine-tuned.

Why this exists separately from analyse_linear_probe.py
------------------------------------------------------
Phase 12 compared the two backbones as frozen feature extractors. The one fatal
objection to that is that RETFound is *meant* to be fine-tuned -- its own paper
reports fine-tuning, not linear probing -- so a frozen-feature gap might say
nothing about how anyone would actually use it. This reads the fine-tuned runs
and answers that objection with the same two bars.

The comparison stays matched: both models are ViT-L/16 at 303.3 M parameters,
both have exactly their last 4 of 24 blocks unfrozen (50.4 M trainable), both
train at lr 1e-4, batch 16, on identical splits with identical augmentation.
``vit_large_patch16_224.mae`` is the checkpoint RETFound's own args name as its
initialisation, so the pretraining corpus remains the only variable.

Source and target, again
------------------------
Reported together for the reason Phase 12 gives: a representation can fit the
source domains well and still transfer badly, and only the pair distinguishes
"worse model" from "worse transfer". The source-to-target drop is the quantity
a domain-generalization claim actually rests on.

Bars
----
A difference counts only if it exceeds the across-seed SD of the paired per-seed
differences AND its paired bootstrap CI excludes zero. The bootstrap resamples
images within one seed, so it cannot see seed-to-seed variation -- which is
exactly how MixStyle produced a p < 0.001 effect in Phase 11 that vanished
across seeds. The SD bar is what catches that.

Runs from saved predictions and the summary table. No GPU.

Usage:
    python analyse_finetune.py
    python analyse_finetune.py --targets ddr --n-bootstrap 10000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

REFERENCE = "retfound_cfp"
CONTROL = "vit_large_mae_in1k"
TRAINABLE_BLOCKS = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
DEFAULT_TARGETS = ["ddr", "aptos", "idrid"]
ANCHOR_SEED = 42
N_BOOTSTRAP = 5000


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import paired_bootstrap_difference
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

    targets = _take("--targets", ",".join(DEFAULT_TARGETS)).split(",")
    n_bootstrap = int(_take("--n-bootstrap", str(N_BOOTSTRAP)))

    outputs = project_root() / "outputs"
    table = pd.read_csv(outputs / "tables" / "lodo_results.csv")
    # trainable_blocks is what separates these runs from the full-network ones
    # in the same file. Without it the fine-tuned ViT-L rows and the DenseNet
    # LODO rows would be averaged together.
    table = table[table.get("trainable_blocks").eq(TRAINABLE_BLOCKS)]

    def load(backbone: str, target: str, seed: int):
        tag = (f"erm-b{BATCH_SIZE}-tb{TRAINABLE_BLOCKS}"
               f"-lr{LEARNING_RATE:g}")
        experiment_id = make_experiment_id(
            protocol="lodo", sources=[d for d in ALL_DOMAINS if d != target],
            target=target, backbone=backbone, method=tag, seed=seed,
        )
        return load_target_predictions(experiment_id, target, outputs_dir=outputs)

    def cell(backbone: str, target: str, column: str):
        rows = table[(table.backbone == backbone) & (table.target == target)]
        return rows.set_index("seed")[column]

    print("=" * 120)
    print(f"FINE-TUNED (last {TRAINABLE_BLOCKS} of 24 blocks): {REFERENCE} vs "
          f"{CONTROL} -- RETFound against the checkpoint it was built from")
    print("=" * 120)
    print("  delta = control - reference per seed, then averaged. Positive "
          "means the ImageNet initialisation won.")
    print("  Both bars required: |delta| > across-seed SD, and a bootstrap CI "
          "excluding zero.\n")

    records = []
    for target in targets:
        ref_t, con_t = cell(REFERENCE, target, "target_qwk"), cell(CONTROL, target, "target_qwk")
        seeds = sorted(set(ref_t.index) & set(con_t.index))
        if not seeds:
            print(f"  {target}: NOT RUN")
            continue
        if len(seeds) < 2:
            print(f"  {target}: only seed(s) {seeds} paired -- no SD, skipping")
            continue

        ref_s, con_s = cell(REFERENCE, target, "source_qwk"), cell(CONTROL, target, "source_qwk")
        deltas = np.array([con_t[s] - ref_t[s] for s in seeds])
        delta = float(deltas.mean())
        delta_sd = float(np.std(deltas, ddof=1))

        anchor = ANCHOR_SEED if ANCHOR_SEED in seeds else seeds[0]
        low = high = p_value = float("nan")
        a, b = load(REFERENCE, target, anchor), load(CONTROL, target, anchor)
        if a is not None and b is not None:
            boot = paired_bootstrap_difference(
                a["y_true"], a["y_pred"], b["y_pred"],
                metric="qwk", n_bootstrap=n_bootstrap)
            low, high, p_value = boot["ci_lower"], boot["ci_upper"], boot["p_value"]

        over = abs(delta) / delta_sd if delta_sd else float("nan")
        clears = over > 1.0 and not (low <= 0 <= high)
        verdict = ("ImageNet init better" if clears and delta > 0 else
                   "RETFound better" if clears and delta < 0 else
                   "within seed noise")

        agree = int(np.sum(np.sign(deltas) == np.sign(delta)))
        records.append({
            "target": target, "n_seeds": len(seeds), "anchor_seed": anchor,
            "retfound_source": float(ref_s[seeds].mean()),
            "control_source": float(con_s[seeds].mean()),
            "retfound_target": float(ref_t[seeds].mean()),
            "retfound_sd": float(np.std(ref_t[seeds], ddof=1)),
            "control_target": float(con_t[seeds].mean()),
            "control_sd": float(np.std(con_t[seeds], ddof=1)),
            "delta_qwk": delta, "delta_sd": delta_sd, "delta_over_sd": over,
            "seeds_agreeing": agree,
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

    print("--- target (cross-domain) ---")
    print(frame[["target", "n_seeds", "retfound_target", "control_target",
                 "delta_qwk", "delta_sd", "delta_over_sd", "seeds_agreeing",
                 "ci_lower", "ci_upper", "p_value", "verdict"]]
          .round(4).to_string(index=False))

    print("\n--- does it fit the sources, or only transfer badly? ---")
    print(frame[["target", "retfound_source", "control_source",
                 "retfound_target", "control_target",
                 "retfound_drop", "control_drop"]].round(4).to_string(index=False))

    print("\n--- severe errors (|error| >= 2 grades) and seed stability ---")
    print(frame[["target", "retfound_severe", "control_severe",
                 "retfound_sd", "control_sd"]].round(4).to_string(index=False))

    for _, r in frame.iterrows():
        if r.seeds_agreeing < r.n_seeds:
            print(f"    note: {r.target} -- only {int(r.seeds_agreeing)} of "
                  f"{int(r.n_seeds)} seeds share the mean's sign; the interval "
                  f"belongs to seed {int(r.anchor_seed)} alone.")

    established = frame[frame.verdict != "within seed noise"]
    print(f"\nClearing both bars: "
          f"{', '.join(established.target) if len(established) else 'none'}"
          f"  ({len(established)} of {len(frame)} targets).")
    if len(frame):
        print(f"No target favours RETFound: "
              f"{bool((frame.delta_qwk > 0).all())}")

    destination = outputs / "tables" / "finetune_comparison.csv"
    frame.to_csv(destination, index=False)
    print(f"\nsaved -> {destination}")


if __name__ == "__main__":
    main()
