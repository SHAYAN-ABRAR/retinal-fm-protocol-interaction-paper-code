"""The domain-generalization comparison: four methods against ERM, two samplers.

This is the generator for the paper's central negative claim. Until 2026-08-25
that claim rested on two feature-space methods (Deep CORAL, MixStyle) run on one
two-source configuration with a 507-image test set. It now covers four methods
spanning three families -- feature alignment, style augmentation, group
reweighting and invariance -- on the leave-one-domain-out target with the
largest gap in the project and 35,108 test images.

Two samplers, and why the second exists
---------------------------------------
Under ordinary shuffling a batch of 32 from the EyePACS-target pool is roughly
24 DDR, 8 APTOS and **zero IDRiD**. Every method here needs several domains per
batch: Deep CORAL cannot estimate a covariance from fewer than four samples, IRM
needs two to split, GroupDRO estimates a group loss from whatever is present. So
the natural-sampler arm tests the methods *as they would run on this data*, and
the domain-balanced arm tests them *as their authors intended*. A negative result
that holds in both is not answerable with "you did not give the method a fair
run".

**ERM is run under both samplers too**, and that is the point. The balanced
sampler oversamples IDRiD's 335 images roughly twelve times per epoch, which
changes the domain prior and the class prior with it. Without an ERM control on
the same sampler, a method scoring badly under balanced batches could not be
told apart from the sampler scoring badly.

Multiplicity, and which direction is conservative
-------------------------------------------------
Eight comparisons are made against ERM. The reflex is to correct for that, and
here the reflex is backwards. Holm-Bonferroni makes differences *harder* to
detect, and the claim being defended is that there are none -- so correcting
would make the claim easier to assert, not harder. The uncorrected result is
therefore the primary one, and the Holm-adjusted p-value is reported beside it
as the weaker statement. A method that fails to beat ERM uncorrected has failed
under the more demanding test.

Did the method actually run?
---------------------------
Each row carries the DG component's own diagnostics from its evaluation report --
how often Deep CORAL's alignment term was degenerate, how often IRM skipped a
domain's penalty, how often GroupDRO saw a domain as a single image. A method
that never fired is not evidence about the method, and the table says so rather
than leaving it to be assumed.

Runs from saved predictions. No GPU.

Usage:
    python analyse_methods.py
    python analyse_methods.py --target ddr --n-bootstrap 10000
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
IMAGE_SIZE = 224
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
DEFAULT_TARGET = "eyepacs"
BASELINE = "erm"
METHODS = ["deep_coral", "mixstyle", "groupdro", "irm"]
SEEDS = [42, 1, 2]
ANCHOR_SEED = 42
N_BOOTSTRAP = 5000

SAMPLERS = [("natural", False), ("domain-balanced", True)]


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import (
        METRIC_FUNCTIONS,
        holm_bonferroni,
        paired_bootstrap_difference,
    )
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

    target = _take("--target", DEFAULT_TARGET)
    n_bootstrap = int(_take("--n-bootstrap", str(N_BOOTSTRAP)))

    outputs = project_root() / "outputs"
    qwk = METRIC_FUNCTIONS["qwk"]
    severe = METRIC_FUNCTIONS["severe_error_rate"]
    ece = METRIC_FUNCTIONS["ece"]
    sources = [d for d in ALL_DOMAINS if d != target]

    def method_tag(method: str, balanced: bool) -> str:
        tag = f"{method}-b{BATCH_SIZE}"
        if IMAGE_SIZE != 224:
            tag += f"-r{IMAGE_SIZE}"
        if balanced:
            tag += "-dbal"
        return tag

    def load(method: str, balanced: bool, seed: int):
        experiment_id = make_experiment_id(
            protocol="lodo", sources=sources, target=target, backbone=BACKBONE,
            method=method_tag(method, balanced), seed=seed,
        )
        run = load_target_predictions(experiment_id, target, outputs_dir=outputs)
        if run is not None:
            report = outputs / "reports" / f"{experiment_id}_evaluation.json"
            payload = json.loads(report.read_text(encoding="utf-8"))
            run["components"] = payload.get("component_statistics", {})
        return run

    def component_note(run) -> str:
        """One line on whether the method's machinery actually engaged."""
        stats = run.get("components") or {}
        if "deep_coral" in stats:
            fraction = stats["deep_coral"].get("degenerate_fraction", 0.0)
            return f"alignment inactive in {fraction:.0%} of batches"
        objective = stats.get("objective") or {}
        if "single_sample_fraction" in objective:
            worst = max(objective["single_sample_fraction"])
            return f"a domain was a single image in {worst:.0%} of its batches"
        if "skipped_fraction" in objective:
            worst = max(objective["skipped_fraction"])
            return f"a domain's penalty skipped in {worst:.0%} of its batches"
        if "mixstyle" in stats:
            return "mixstyle active"
        return ""

    rows = []
    for sampler_name, balanced in SAMPLERS:
        baseline_by_seed = {s: load(BASELINE, balanced, s) for s in SEEDS}
        baseline_by_seed = {s: r for s, r in baseline_by_seed.items() if r is not None}
        if ANCHOR_SEED not in baseline_by_seed:
            print(f"  {sampler_name}: no ERM baseline at seed {ANCHOR_SEED} -- skipped")
            continue

        anchor_truth = baseline_by_seed[ANCHOR_SEED]["y_true"]

        for method in METHODS:
            candidate_by_seed = {s: load(method, balanced, s) for s in SEEDS}
            candidate_by_seed = {s: r for s, r in candidate_by_seed.items() if r is not None}
            paired = sorted(set(baseline_by_seed) & set(candidate_by_seed))
            if ANCHOR_SEED not in paired:
                print(f"  {sampler_name}/{method}: no seed-{ANCHOR_SEED} pair -- NOT RUN")
                continue

            for seed in paired:
                for run in (baseline_by_seed[seed], candidate_by_seed[seed]):
                    assert (run["y_true"] == anchor_truth).all(), (
                        f"{sampler_name}/{method} seed {seed}: labels disagree "
                        "between runs")

            def collect(metric, probabilities_key=None):
                base, cand, diff = [], [], []
                for seed in paired:
                    pair = []
                    for run in (baseline_by_seed[seed], candidate_by_seed[seed]):
                        probabilities = run[probabilities_key] if probabilities_key else None
                        pair.append(metric(run["y_true"], run["y_pred"], probabilities))
                    base.append(pair[0])
                    cand.append(pair[1])
                    diff.append(pair[1] - pair[0])
                return np.array(base), np.array(cand), np.array(diff)

            def sd(values):
                return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

            qwk_base, qwk_cand, qwk_delta = collect(qwk)
            sev_base, sev_cand, sev_delta = collect(severe)
            ece_base, ece_cand, ece_delta = collect(ece, "scaled")

            baseline_anchor = baseline_by_seed[ANCHOR_SEED]
            candidate_anchor = candidate_by_seed[ANCHOR_SEED]
            test = paired_bootstrap_difference(
                baseline_anchor["y_true"], baseline_anchor["y_pred"],
                candidate_anchor["y_pred"],
                metric="qwk", n_bootstrap=n_bootstrap, seed=ANCHOR_SEED,
            )

            delta = float(qwk_delta.mean())
            bar = sd(qwk_delta)
            # The interval below is computed on the anchor seed alone, while
            # delta is the mean over seeds. Those are different quantities and
            # they can disagree in sign: MixStyle here is +0.020 on seed 42 and
            # -0.006 averaged over three. Reporting the mean beside the anchor's
            # interval and nothing else produces a row that reads as incoherent,
            # so the anchor's own difference is carried alongside it.
            delta_anchor = float(qwk_delta[paired.index(ANCHOR_SEED)])
            if not np.isfinite(bar):
                verdict = "no SD available"
            elif abs(delta) <= bar:
                verdict = "within seed noise"
            elif not test["significant"]:
                verdict = "CI spans zero"
            else:
                verdict = "BEATS ERM" if delta > 0 else "WORSE than ERM"

            rows.append({
                "sampler": sampler_name,
                "method": method,
                "n_seeds": len(paired),
                "n_test": len(anchor_truth),
                "erm_qwk": float(qwk_base.mean()),
                "method_qwk": float(qwk_cand.mean()),
                "delta_qwk": delta,
                "delta_qwk_seed42": delta_anchor,
                "delta_qwk_sd": bar,
                "delta_over_sd": abs(delta) / bar if np.isfinite(bar) and bar else float("nan"),
                "ci_lower": test["ci_lower"],
                "ci_upper": test["ci_upper"],
                "p_value": test["p_value"],
                "verdict": verdict,
                "erm_severe": float(sev_base.mean()),
                "method_severe": float(sev_cand.mean()),
                "delta_severe": float(sev_delta.mean()),
                "delta_severe_sd": sd(sev_delta),
                "erm_ece_scaled": float(ece_base.mean()),
                "method_ece_scaled": float(ece_cand.mean()),
                "delta_ece_scaled": float(ece_delta.mean()),
                "component_note": component_note(candidate_anchor),
            })

    if not rows:
        print("NOT RUN -- no matched method/ERM pairs found")
        return

    table = pd.DataFrame(rows)

    # Reported, not applied: see the module docstring. Correcting makes
    # differences harder to find, and the claim is that there are none.
    adjusted = holm_bonferroni(
        table["p_value"].tolist(),
        [f"{r.sampler}/{r.method}" for r in table.itertuples()])
    table["p_holm"] = [record["p_adjusted"] for record in adjusted]

    pd.set_option("display.width", 250)
    print("=" * 122)
    print(f"DOMAIN GENERALIZATION: methods vs ERM, LODO target {target.upper()}, "
          f"{IMAGE_SIZE} px, paired seeds, matched images")
    print("=" * 122)
    print("  delta = method - ERM under the SAME sampler. Positive favours the method.")
    print("  delta_qwk is the mean over seeds; the interval and p-value belong")
    print(f"  to delta_qwk_seed{ANCHOR_SEED}, not to the mean. The SD is across paired seeds.")
    print()

    for sampler_name, _ in SAMPLERS:
        part = table[table["sampler"] == sampler_name]
        if not len(part):
            continue
        print(f"--- {sampler_name} sampler "
              f"(ERM {part['erm_qwk'].iloc[0]:.4f}) ---")
        print(part[[
            "method", "n_seeds", "method_qwk", "delta_qwk", "delta_qwk_sd",
            "delta_over_sd", "delta_qwk_seed42", "ci_lower", "ci_upper",
            "p_value", "p_holm", "verdict",
        ]].round(4).to_string(index=False))
        disagree = part[(part["delta_qwk"] * part["delta_qwk_seed42"]) < 0]
        for row in disagree.itertuples():
            print(f"    note: {row.method} is {row.delta_qwk_seed42:+.4f} on seed "
                  f"{ANCHOR_SEED} but {row.delta_qwk:+.4f} averaged over "
                  f"{row.n_seeds} seeds -- the interval describes the former.")
        print()

    print("Did the method's machinery engage?\n")
    for row in table.itertuples():
        if row.component_note:
            print(f"  {row.sampler:15s} {row.method:12s} {row.component_note}")

    beat = table[table["verdict"] == "BEATS ERM"]
    print()
    if len(beat):
        print(f"  {len(beat)} configuration(s) beat ERM under both bars:")
        for row in beat.itertuples():
            print(f"    {row.sampler}/{row.method}: {row.delta_qwk:+.4f}")
    else:
        print("  NO method beats ERM under both bars, in either sampler.")

    path = outputs / "tables" / f"method_comparison_lodo_{target}.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved -> {path}")

    # The sampler itself is a result, and it is not the same question as the
    # methods: it says whether rebalancing the domain prior helps at all.
    print("\nThe sampler, independent of any method:")
    for name, balanced in SAMPLERS:
        runs = [load(BASELINE, balanced, s) for s in SEEDS]
        runs = [r for r in runs if r is not None]
        if not runs:
            continue
        values = [qwk(r["y_true"], r["y_pred"], None) for r in runs]
        spread = (f" +/- {np.std(values, ddof=1):.4f}" if len(values) > 1 else "")
        print(f"  ERM, {name:15s} QWK {np.mean(values):.4f}{spread}  ({len(values)} seeds)")


if __name__ == "__main__":
    main()
