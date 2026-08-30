"""Difference-of-differences: does the adaptation protocol change the ranking?

The paper's central claim is an INTERACTION, and until now it was supported by
the wrong evidence. Observing that the ImageNet-vs-RETFound gap is significant
under frozen probing and not significant under partial fine-tuning does not
establish that the two protocols differ: "significant" and "not significant"
can straddle a boundary while the underlying effects are statistically
indistinguishable from each other. The comparison has to be made directly.

For each held-out domain and each seed s in the five seeds common to both
protocols:

    D_frozen(s)  = QWK_ImageNet_frozen(s)  - QWK_RETFound_frozen(s)
    D_partial(s) = QWK_ImageNet_partial(s) - QWK_RETFound_partial(s)
    I(s)         = D_partial(s) - D_frozen(s)

I(s) < 0 means the ImageNet advantage shrinks when the backbone is allowed to
adapt -- the direction the paper argues for.

Two separate procedures, deliberately not conflated
---------------------------------------------------
**Uncertainty** comes from the crossed seed x case bootstrap. Every replicate
draws seeds with replacement, draws ONE case sample, and applies those same case
indices to all four conditions (RETFound frozen, ImageNet frozen, RETFound
partial, ImageNet partial) so the pairing that gives the design its power is
preserved.

**Inference** comes from a paired t-test on the five per-seed I(s) values, with
Holm correction across the three domains. The ordinary percentile bootstrap
resamples around the empirical distribution rather than under H0, so its tail
mass is not a calibrated null p-value; reporting it as one would overstate what
a bootstrap can do. A sign-flip permutation test is reported alongside as a
distribution-free sensitivity check.

n = 5 seeds is a small sample for a t-test and the result is reported with that
stated, not hidden. Failure to reject is NOT evidence of equivalence and is
never described as such.

Runs from saved predictions. No GPU.

Usage:
    python analyse_interaction.py
    python analyse_interaction.py --n-bootstrap 5000
"""

from __future__ import annotations

import itertools
import sys

sys.path.insert(0, ".")

ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
TARGETS = ["ddr", "aptos", "idrid"]
REFERENCE = "retfound-cfp"
CONTROL = "vit-large-mae-in1k"
FROZEN_TAG = "linprobe-b256"
PARTIAL_TAG = "erm-b16-tb4-lr0.0001"
COMMON_SEEDS = [42, 1, 2, 3, 4]
N_BOOTSTRAP = 2000


def main() -> None:
    import numpy as np
    import pandas as pd
    from scipy import stats

    from src.evaluation.crossed_bootstrap import holm_adjust
    from src.evaluation.metrics import quadratic_weighted_kappa
    from src.utils.io import project_root
    from src.visualization.calibration_figures import load_target_predictions

    arguments = sys.argv[1:]

    def _take(flag, default):
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
    for target in TARGETS:
        cells, truth = {}, None
        usable = []
        for seed in COMMON_SEEDS:
            got = {}
            for name, backbone, tag in (
                    ("ref_frozen", REFERENCE, FROZEN_TAG),
                    ("con_frozen", CONTROL, FROZEN_TAG),
                    ("ref_partial", REFERENCE, PARTIAL_TAG),
                    ("con_partial", CONTROL, PARTIAL_TAG)):
                run = load(backbone, target, seed, tag)
                if run is None:
                    break
                got[name] = run["y_pred"]
                truth = run["y_true"]
            if len(got) == 4:
                cells[seed] = got
                usable.append(seed)
        if len(usable) < 3:
            print(f"  {target}: NOT RUN (only {len(usable)} complete seed(s))")
            continue

        # ---- observed interaction, per seed -------------------------------
        per_seed = {}
        for s in usable:
            c = cells[s]
            d_frozen = qwk(truth, c["con_frozen"]) - qwk(truth, c["ref_frozen"])
            d_partial = qwk(truth, c["con_partial"]) - qwk(truth, c["ref_partial"])
            per_seed[s] = d_partial - d_frozen
        values = np.array([per_seed[s] for s in usable])
        observed = float(values.mean())

        # ---- crossed bootstrap: uncertainty only --------------------------
        rng = np.random.default_rng(11)
        n_cases = len(truth)
        draws = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            picked = rng.choice(usable, size=len(usable), replace=True)
            idx = rng.integers(0, n_cases, size=n_cases)   # ONE case sample
            t = truth[idx]
            per = []
            for s in picked:
                c = cells[s]
                df = qwk(t, c["con_frozen"][idx]) - qwk(t, c["ref_frozen"][idx])
                dp = qwk(t, c["con_partial"][idx]) - qwk(t, c["ref_partial"][idx])
                per.append(dp - df)
            draws[b] = float(np.mean(per))
        lower, upper = np.percentile(draws, [2.5, 97.5])

        # ---- inference: paired t-test on per-seed interaction --------------
        t_stat, p_t = stats.ttest_1samp(values, 0.0)

        # ---- sensitivity: exact sign-flip permutation ----------------------
        signs = np.array(list(itertools.product([1, -1], repeat=len(values))))
        means = (signs * values).mean(axis=1)
        p_perm = float((np.abs(means) >= abs(observed) - 1e-12).mean())

        records.append({
            "target": target, "n_seeds": len(usable),
            "n_test": n_cases,
            "interaction": observed,
            "ci_lower": float(lower), "ci_upper": float(upper),
            "seed_sd": float(np.std(values, ddof=1)),
            "sign_agreement": int(np.sum(np.sign(values) == np.sign(observed))),
            "t_stat": float(t_stat), "p_ttest": float(p_t),
            "p_signflip": p_perm,
            "per_seed": {s: round(per_seed[s], 4) for s in usable},
        })

    if not records:
        print("nothing to compare")
        return

    frame = pd.DataFrame(records)
    frame["p_holm"] = holm_adjust(frame["p_ttest"].tolist())

    print("=" * 108)
    print("PROTOCOL INTERACTION  --  (ImageNet - RETFound)_partial  minus  "
          "(ImageNet - RETFound)_frozen")
    print("=" * 108)
    print("  Negative = the ImageNet advantage shrinks when the backbone can "
          "adapt.")
    print("  CI: crossed seed x case bootstrap (uncertainty). p: paired t-test "
          "on per-seed values (inference).")
    print(f"  Holm across the {len(frame)} held-out domains. "
          f"Bootstrap replicates B = {n_bootstrap}.\n")
    show = frame[["target", "n_seeds", "interaction", "ci_lower", "ci_upper",
                  "seed_sd", "sign_agreement", "p_ttest", "p_holm",
                  "p_signflip"]]
    print(show.round(4).to_string(index=False))

    print("\n  per-seed interaction values:")
    for _, r in frame.iterrows():
        print(f"    {r.target:<8}{r.per_seed}")

    print("\n  Interpretation:")
    for _, r in frame.iterrows():
        if r.p_holm < 0.05:
            verdict = ("protocol changes the gap "
                       f"({'shrinks' if r.interaction < 0 else 'grows'})")
        else:
            verdict = "no detectable interaction (NOT evidence of equivalence)"
        print(f"    {r.target:<8}{verdict}")

    destination = outputs / "tables" / "protocol_interaction.csv"
    frame.drop(columns=["per_seed"]).to_csv(destination, index=False)
    print(f"\nsaved -> {destination}")
    print("\n  n = 5 seeds is a small sample for a t-test; the sign-flip column "
          "is the distribution-free check.\n  Its resolution is bounded at "
          f"1/2^5 = {1/2**5:.4f}, so no p below that is reportable.")


if __name__ == "__main__":
    main()
