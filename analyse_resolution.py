"""512 px against 224 px, in-domain and leave-one-domain-out.

This is the largest effect in the project -- larger than any domain-generalization
method, and larger than quadrupling the training set -- and until now it lived in
two tables with no script behind them, from a single seed, with the significance
bar borrowed from the 224 px runs.

What is actually being compared
-------------------------------
Not resolution alone. The 224 px runs use batch 32; the 512 px runs use batch
16, because 512 px at batch 32 does not fit in 8 GB of VRAM. The comparison is
therefore between two *configurations*::

    224 px, batch 32     vs     512 px, batch 16

Resolution is the dominant term and the intended one, but it is not the only
thing that changed, and no run isolates it on this hardware. Every row carries a
``configuration`` column saying so. Calling this a pure resolution effect would
be claiming an experiment that was never run.

Protocol
--------
* **Matched images.** Both resolutions score the same test split, checked by
  image id, with labels asserted equal.
* **Paired seeds.** 512 px seed *s* is differenced against 224 px seed *s*, so
  the SD is the variation of the difference.
* **Two bars.** Real means the effect exceeds the across-seed SD *and* its
  paired bootstrap interval excludes zero.
* **The SD is labelled.** Where 512 px has one seed there is no paired SD, and
  the 224 px across-seed SD stands in -- recorded in ``sd_source``, never left
  to look like a bar that was measured.

Runs from saved predictions. No GPU.

Usage:
    python analyse_resolution.py
    python analyse_resolution.py --n-bootstrap 10000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
SEEDS = [42, 1, 2]
ANCHOR_SEED = 42
N_BOOTSTRAP = 5000

# (label, image size, batch size). The batch size is part of the identity of a
# configuration here, not an incidental setting, because it is what makes the
# 512 px arm fit in 8 GB.
BASELINE = ("224px", 224, 32)
CANDIDATE = ("512px", 512, 16)


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
    qwk = METRIC_FUNCTIONS["qwk"]
    severe = METRIC_FUNCTIONS["severe_error_rate"]
    ece = METRIC_FUNCTIONS["ece"]

    def method_tag(image_size: int, batch_size: int) -> str:
        # Mirrors _method_tag() in run_lodo.py and run_in_domain.py: the
        # resolution suffix appears only when it is not the 224 px default, so
        # existing ids keep their names.
        tag = f"erm-b{batch_size}"
        if image_size != 224:
            tag += f"-r{image_size}"
        return tag

    def load(protocol: str, target: str, image_size: int, batch_size: int, seed: int):
        sources = ([target] if protocol == "in_domain"
                   else [d for d in ALL_TARGETS if d != target])
        experiment_id = make_experiment_id(
            protocol=protocol, sources=sources, target=target, backbone=BACKBONE,
            method=method_tag(image_size, batch_size), seed=seed,
        )
        return load_target_predictions(experiment_id, target, outputs_dir=outputs)

    rows = []
    for protocol in ("in_domain", "lodo"):
        for target in ALL_TARGETS:
            baseline_by_seed, candidate_by_seed = {}, {}
            for seed in SEEDS:
                baseline = load(protocol, target, BASELINE[1], BASELINE[2], seed)
                candidate = load(protocol, target, CANDIDATE[1], CANDIDATE[2], seed)
                if baseline is not None:
                    baseline_by_seed[seed] = baseline
                if candidate is not None:
                    candidate_by_seed[seed] = candidate

            paired_seeds = sorted(set(baseline_by_seed) & set(candidate_by_seed))
            if ANCHOR_SEED not in paired_seeds:
                continue

            anchor = baseline_by_seed[ANCHOR_SEED]
            n_test = len(anchor["y_true"])
            for seed in paired_seeds:
                for run in (baseline_by_seed[seed], candidate_by_seed[seed]):
                    assert len(run["y_true"]) == n_test, (
                        f"{protocol}/{target} seed {seed}: {run['experiment_id']} "
                        f"scored {len(run['y_true'])} images, expected {n_test}")
                    assert (run["y_true"] == anchor["y_true"]).all(), (
                        f"{protocol}/{target} seed {seed}: labels disagree "
                        "between resolutions")

            def collect(metric, use_probabilities=None):
                baseline_values, candidate_values, differences = [], [], []
                for seed in paired_seeds:
                    pair = []
                    for run in (baseline_by_seed[seed], candidate_by_seed[seed]):
                        probabilities = (run[use_probabilities]
                                         if use_probabilities else None)
                        pair.append(metric(run["y_true"], run["y_pred"], probabilities))
                    baseline_values.append(pair[0])
                    candidate_values.append(pair[1])
                    differences.append(pair[1] - pair[0])
                return (np.array(baseline_values), np.array(candidate_values),
                        np.array(differences))

            def sd(values):
                return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

            qwk_baseline, qwk_candidate, qwk_delta = collect(qwk)
            severe_baseline, severe_candidate, severe_delta = collect(severe)
            ece_baseline, ece_candidate, _ = collect(ece, "scaled")

            if len(paired_seeds) > 1:
                qwk_bar = sd(qwk_delta)
                severe_bar = sd(severe_delta)
                sd_source = f"paired ({len(paired_seeds)} seeds)"
            else:
                baseline_only = [baseline_by_seed[s] for s in sorted(baseline_by_seed)]
                qwk_bar = sd([qwk(r["y_true"], r["y_pred"], None) for r in baseline_only])
                severe_bar = sd([severe(r["y_true"], r["y_pred"], None)
                                 for r in baseline_only])
                sd_source = (f"proxy: {BASELINE[0]} SD over "
                             f"{len(baseline_only)} seeds")

            baseline_anchor = baseline_by_seed[ANCHOR_SEED]
            candidate_anchor = candidate_by_seed[ANCHOR_SEED]

            def test(metric_name):
                return paired_bootstrap_difference(
                    baseline_anchor["y_true"], baseline_anchor["y_pred"],
                    candidate_anchor["y_pred"],
                    metric=metric_name, n_bootstrap=n_bootstrap, seed=ANCHOR_SEED,
                )

            qwk_test = test("qwk")
            severe_test = test("severe_error_rate")

            def verdict(delta, bar, excludes_zero, better_when_positive=True):
                if not np.isfinite(bar):
                    return "no SD available"
                if abs(delta) <= bar:
                    return "within seed noise"
                if not excludes_zero:
                    return "CI spans zero"
                improved = (delta > 0) if better_when_positive else (delta < 0)
                return "REAL (better)" if improved else "REAL (worse)"

            delta_qwk = float(qwk_delta.mean())
            delta_severe = float(severe_delta.mean())

            rows.append({
                "protocol": protocol,
                "target": target,
                "n_test": n_test,
                "n_paired_seeds": len(paired_seeds),
                "seeds": ",".join(str(s) for s in paired_seeds),
                "configuration": (f"{BASELINE[0]} b{BASELINE[2]} -> "
                                  f"{CANDIDATE[0]} b{CANDIDATE[2]}"),

                "qwk_224": float(qwk_baseline.mean()),
                "qwk_224_sd": sd(qwk_baseline),
                "qwk_512": float(qwk_candidate.mean()),
                "qwk_512_sd": sd(qwk_candidate),
                "delta_qwk": delta_qwk,
                "delta_qwk_sd": qwk_bar,
                "sd_source": sd_source,
                "delta_over_sd": (float(abs(delta_qwk) / qwk_bar)
                                  if np.isfinite(qwk_bar) and qwk_bar > 0
                                  else float("nan")),
                "qwk_ci_lower": qwk_test["ci_lower"],
                "qwk_ci_upper": qwk_test["ci_upper"],
                "qwk_p": qwk_test["p_value"],
                "qwk_verdict": verdict(delta_qwk, qwk_bar,
                                       bool(qwk_test["significant"])),

                "severe_224": float(severe_baseline.mean()),
                "severe_512": float(severe_candidate.mean()),
                "delta_severe": delta_severe,
                "delta_severe_sd": severe_bar,
                "severe_relative": (delta_severe / float(severe_baseline.mean())
                                    if severe_baseline.mean() > 0 else float("nan")),
                "severe_ci_lower": severe_test["ci_lower"],
                "severe_ci_upper": severe_test["ci_upper"],
                # Fewer severe errors is better, so a negative difference is the
                # improvement.
                "severe_verdict": verdict(delta_severe, severe_bar,
                                          bool(severe_test["significant"]),
                                          better_when_positive=False),

                "ece_scaled_224": float(ece_baseline.mean()),
                "ece_scaled_512": float(ece_candidate.mean()),
                "delta_ece_scaled": float(ece_candidate.mean() - ece_baseline.mean()),
            })

    if not rows:
        print("NOT RUN -- no matched 224/512 prediction pairs found")
        return

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 260)

    print("=" * 122)
    print(f"RESOLUTION: {BASELINE[0]} batch {BASELINE[2]} -> "
          f"{CANDIDATE[0]} batch {CANDIDATE[2]}, matched images, paired seeds")
    print("=" * 122)
    print("  The batch size differs because 512 px at batch 32 does not fit in "
          "8 GB. Resolution is the")
    print("  dominant term in this change but not the only one, and no run on "
          "this hardware isolates it.")
    print(f"  Bootstrap interval is on seed {ANCHOR_SEED}; the SD is across "
          "paired seeds where more than one exists.\n")

    for protocol in ("in_domain", "lodo"):
        part = table[table["protocol"] == protocol]
        if not len(part):
            continue
        print(f"--- {protocol} ---")
        print(part[[
            "target", "n_test", "n_paired_seeds", "qwk_224", "qwk_512",
            "delta_qwk", "delta_qwk_sd", "delta_over_sd", "qwk_ci_lower",
            "qwk_ci_upper", "qwk_verdict",
        ]].round(4).to_string(index=False))
        print()
        print(part[[
            "target", "severe_224", "severe_512", "delta_severe",
            "severe_relative", "severe_ci_lower", "severe_ci_upper",
            "severe_verdict",
        ]].round(4).to_string(index=False))
        print()

    borrowed = table[table["sd_source"].str.startswith("proxy")]
    if len(borrowed):
        print(f"  NOTE: {len(borrowed)} of {len(table)} rows use a borrowed SD.")
        print("  Those verdicts are provisional until 512 px has two or more "
              "seeds of its own:")
        for _, row in borrowed.iterrows():
            print(f"    {row['protocol']:10s} {row['target']}")

    path = outputs / "tables" / "resolution_comparison.csv"
    table.to_csv(path, index=False)
    print(f"\nsaved -> {path}")

    # The superseded single-seed table stays on disk as a record. Where the two
    # overlap they must agree; two tables describing the same runs and quietly
    # disagreeing is a failure this project has already had once.
    legacy_path = outputs / "tables" / "resolution_effect_eyepacs_s42.csv"
    if legacy_path.exists() and ANCHOR_SEED == 42:
        legacy = pd.read_csv(legacy_path).set_index("metric")
        match = table[(table["protocol"] == "in_domain")
                      & (table["target"] == "eyepacs")]
        if len(match) and "qwk" in legacy.index:
            old = float(legacy.loc["qwk", "difference"])
            new = float(match.iloc[0]["delta_qwk"])
            if abs(new - old) > 5e-3:
                raise SystemExit(
                    f"in_domain/eyepacs delta_qwk={new:.4f} disagrees with "
                    f"{legacy_path.name} difference={old:.4f}")
            print(f"  checked against {legacy_path.name}: seed-42 QWK agrees")


if __name__ == "__main__":
    main()
