"""The headline comparison under the crossed seed x case bootstrap.

Supersedes the two-bar rule as the primary inference for the foundation-model
comparison. The two bars are still reported -- seed SD and sign agreement --
but as robustness diagnostics beside a calibrated interval, not as the test.

This changes verdicts in both directions, which is the point of doing it:

* IDRiD frozen was "within seed noise" under the two bars because the anchor
  seed's case-only interval spanned zero. Crossed over ten seeds its interval
  excludes zero uncorrected -- but does not survive Holm.
* IDRiD partial fine-tuning was "RETFound better" at 1.13x the seed SD. Crossed,
  its interval spans zero (p = 0.063). The claim does not hold and is dropped.
* DDR partial fine-tuning was a null at 0.82x. Crossed, it excludes zero
  uncorrected (p = 0.043) and is a null again after Holm.

Multiplicity: six comparisons form one family (two protocols x three held-out
domains), so Holm-Bonferroni is applied across all six. Note the direction of
conservatism -- correction makes differences harder to detect, so for the
partial-fine-tuning arm, where the claim being defended is that no difference
exists, the *uncorrected* p is the more demanding test and is reported first.

Runs from saved predictions. No GPU.

Usage:
    python analyse_crossed.py
    python analyse_crossed.py --n-bootstrap 5000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
TARGETS = ["ddr", "aptos", "idrid"]
REFERENCE = "retfound-cfp"
CONTROL = "vit-large-mae-in1k"
PROTOCOLS = [
    ("frozen", "linprobe-b256", list(range(10)) + [42]),
    ("partial_finetune_4", "erm-b16-tb4-lr0.0001", [42, 1, 2, 3, 4]),
]
N_BOOTSTRAP = 2000


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.crossed_bootstrap import (crossed_bootstrap_difference,
                                                  holm_adjust)
    from src.evaluation.seed_inference import seed_level_test
    from src.evaluation.metrics import quadratic_weighted_kappa
    from src.utils.io import project_root
    from src.visualization.calibration_figures import load_target_predictions

    arguments = sys.argv[1:]

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            i = arguments.index(flag)
            v = arguments[i + 1]
            del arguments[i:i + 2]
            return v
        return default

    n_bootstrap = int(_take("--n-bootstrap", str(N_BOOTSTRAP)))
    outputs = project_root() / "outputs"

    def qwk(y_true, y_pred):
        return quadratic_weighted_kappa(y_true, y_pred, num_classes=5)

    def load(backbone, target, seed, tag):
        sources = "-".join(sorted(d for d in ALL_DOMAINS if d != target))
        return load_target_predictions(
            f"lodo_{sources}__{target}_{backbone}_{tag}_s{seed}",
            target, outputs_dir=outputs)

    records = []
    for protocol, tag, seeds in PROTOCOLS:
        for target in TARGETS:
            reference, candidate, truth = {}, {}, None
            for seed in seeds:
                a = load(REFERENCE, target, seed, tag)
                b = load(CONTROL, target, seed, tag)
                if a is None or b is None:
                    continue
                reference[seed], candidate[seed] = a["y_pred"], b["y_pred"]
                truth = a["y_true"]
            if truth is None or len(reference) < 2:
                print(f"  {protocol}/{target}: NOT RUN")
                continue
            r = crossed_bootstrap_difference(
                truth, reference, candidate, metric=qwk,
                n_bootstrap=n_bootstrap, seed=7)
            records.append({
                "protocol": protocol, "target": target,
                "n_seeds": r["n_seeds"], "n_test": len(truth),
                "delta_qwk": r["difference"],
                "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                "seed_sd": r["seed_sd"],
                "sign_agreement": r["sign_agreement"],
                "exceeds_seed_sd": r["exceeds_seed_sd"],
                # Inference on the per-seed effects. The bootstrap no longer
                # returns a p_value: its tail mass is built around the
                # empirical estimate, not under H0, so Holm-correcting it gave
                # an uncalibrated quantity the look of a controlled error rate.
                **{k: v for k, v in seed_level_test(
                    list(r["per_seed_difference"].values())).items()
                   if k in {"t_stat", "p_ttest", "p_signflip",
                            "min_attainable_p"}},
            })

    if not records:
        print("nothing to compare")
        return

    frame = pd.DataFrame(records)
    frame["p_holm"] = holm_adjust(frame["p_ttest"].tolist())
    frame["ci_excludes_zero"] = (frame.ci_lower > 0) | (frame.ci_upper < 0)
    # A claim needs both bars: the seed-level test (does another seed agree?)
    # and the crossed interval (does another sample of patients agree?). They
    # part company on IDRiD frozen, where ten seeds give Holm p = 0.014 but a
    # 507-case test set leaves the interval spanning zero.
    established = (frame.p_holm < 0.05) & frame.ci_excludes_zero
    frame["verdict"] = np.where(
        established,
        np.where(frame.delta_qwk > 0, "ImageNet init better", "RETFound better"),
        np.where(frame.p_holm < 0.05,
                 "seed-level only; crossed interval spans zero",
                 "no difference after correction"))

    print("=" * 112)
    print("CROSSED SEED x CASE BOOTSTRAP -- RETFound vs its ImageNet-MAE "
          "initialisation")
    print("=" * 112)
    print("  delta = ImageNet-MAE minus RETFound. Seeds resampled with "
          "replacement; one case sample per")
    print("  iteration shared across every seed and both models. The interval "
          "is the bootstrap and quantifies")
    print("  uncertainty; p_ttest is a one-sample t-test on the per-seed "
          "effects and is the inference, Holm-")
    print("  corrected across all six comparisons; p_signflip is the exact "
          "permutation sensitivity check.\n")
    print(frame[["protocol", "target", "n_seeds", "delta_qwk", "ci_lower",
                 "ci_upper", "p_ttest", "p_holm", "p_signflip", "seed_sd",
                 "sign_agreement", "verdict"]].round(4).to_string(index=False))
    for n_seeds in sorted(frame.n_seeds.unique()):
        floor = float(frame[frame.n_seeds == n_seeds].min_attainable_p.iloc[0])
        print(f"\n  at {int(n_seeds)} seeds the exact sign-flip test cannot "
              f"return a two-sided p below {floor:.4f}.")

    print("\n  Robustness diagnostics (NOT the significance test):")
    for _, r in frame.iterrows():
        flag = "" if r.exceeds_seed_sd == r.ci_excludes_zero else "   <- disagree"
        print(f"    {r.protocol:<20}{r.target:<8}"
              f"|delta|>seedSD={str(bool(r.exceeds_seed_sd)):<5} "
              f"CI excludes 0={str(bool(r.ci_excludes_zero)):<5}{flag}")

    destination = outputs / "tables" / "crossed_bootstrap_foundation.csv"
    frame.to_csv(destination, index=False)
    print(f"\nsaved -> {destination}")


if __name__ == "__main__":
    main()
