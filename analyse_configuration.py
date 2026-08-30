"""Separate the resolution effect from the batch-size effect that rode with it.

The problem this exists to fix
------------------------------
The project's largest reported effect was "512 px beats 224 px". It was never
that. Every 512 px run used batch 16, because 512 px at batch 32 does not fit in
8 GB of VRAM, and every 224 px run used batch 32. The two settings moved
together in all 39 runs -- 27 at 224/32, 12 at 512/16, none at 224/16 -- so the
comparison could only ever measure the pair. ``analyse_resolution.py`` says so
in a ``configuration`` column, which is honest but leaves the reader unable to
attribute the effect.

Q1 adds the missing cell: 224 px at batch 16. Three arms now exist, and the
comparison splits into two contrasts that each hold one factor fixed::

    A  batch        224/b32 -> 224/b16     resolution held at 224
    B  resolution   224/b16 -> 512/b16     batch held at 16
    C  combined     224/b32 -> 512/b16     the original confounded comparison

B is the contrast the manuscript's resolution claim actually needs. A is the
contrast that says how much of C was never about resolution at all.

Why the three are not one family
--------------------------------
On the per-seed means C = A + B by construction, so the three contrasts are
algebraically dependent and correcting a single Holm family across all three
would be incoherent -- it would charge a multiplicity penalty for a hypothesis
that is a sum of the other two. Holm is therefore applied *within* each contrast
across the held-out domains, and the family is named in the output.

The residual C - (A + B) is reported as an additivity check. It is zero for the
per-seed means by identity and is included because a non-zero value would mean
a bookkeeping error -- mismatched seeds or test sets -- not a scientific finding.
The interesting non-additivity would be between the crossed-bootstrap intervals,
which are not additive, and those are reported per contrast rather than summed.

What is and is not decomposed
-----------------------------
LODO only, and only the two held-out domains that have a 512 px arm. The
in-domain 512 px runs have no 224/b16 counterpart, so in-domain resolution stays
a configuration effect and is reported as such rather than silently dropped.

Uncertainty is the crossed seed x case bootstrap: seeds resampled, one case
sample per replicate shared across every seed and both arms. Calibration is
descriptive only -- ECE needs probabilities and the crossed bootstrap takes a
``(y_true, y_pred)`` metric -- so it appears in the per-seed table and not in the
intervals.

Runs from saved predictions. No GPU.

Usage:
    python analyse_configuration.py
    python analyse_configuration.py --n-bootstrap 10000
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
# Only these two held-out domains have a 512 px arm.
TARGETS = ["eyepacs", "idrid"]
SEEDS = [42, 1, 2]
N_BOOTSTRAP = 2000

# (label, image size, batch size)
ARMS = {
    "224/b32": (224, 32),
    "224/b16": (224, 16),
    "512/b16": (512, 16),
}

# (name, reference arm, candidate arm, what it isolates)
CONTRASTS = [
    ("batch", "224/b32", "224/b16", "batch size, resolution held at 224 px"),
    ("resolution", "224/b16", "512/b16", "resolution, batch held at 16"),
    ("combined", "224/b32", "512/b16", "both, as originally reported"),
]

# Metrics the crossed bootstrap can carry: they take (y_true, y_pred) only.
BOOTSTRAP_METRICS = ["qwk", "f1_macro", "severe_error_rate"]


def method_tag(image_size: int, batch_size: int) -> str:
    """Mirrors _method_tag() in run_lodo.py.

    The resolution suffix appears only when it is not the 224 px default, so
    ids written before 512 px existed keep their names.
    """
    tag = f"erm-b{batch_size}"
    if image_size != 224:
        tag += f"-r{image_size}"
    return tag


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.bootstrap import METRIC_FUNCTIONS
    from src.evaluation.crossed_bootstrap import (crossed_bootstrap_difference,
                                                  holm_adjust)
    from src.utils.io import project_root
    from src.visualization.calibration_figures import load_target_predictions

    arguments = sys.argv[1:]
    n_bootstrap = N_BOOTSTRAP
    if "--n-bootstrap" in arguments:
        n_bootstrap = int(arguments[arguments.index("--n-bootstrap") + 1])

    outputs = project_root() / "outputs"
    tables = outputs / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    def metric_fn(name):
        """METRIC_FUNCTIONS entries take probabilities; the bootstrap does not."""
        function = METRIC_FUNCTIONS[name]

        def wrapped(y_true, y_pred, _function=function):
            return _function(y_true, y_pred, None)

        return wrapped

    def load(target: str, arm: str, seed: int):
        image_size, batch_size = ARMS[arm]
        sources = "-".join(sorted(d for d in ALL_DOMAINS if d != target))
        experiment_id = (f"lodo_{sources}__{target}_{BACKBONE}_"
                         f"{method_tag(image_size, batch_size)}_s{seed}")
        return load_target_predictions(experiment_id, target, outputs_dir=outputs)

    # ---------------------------------------------------------------- load
    runs: dict[tuple[str, str, int], dict] = {}
    missing: list[str] = []
    for target in TARGETS:
        for arm in ARMS:
            for seed in SEEDS:
                run = load(target, arm, seed)
                if run is None:
                    missing.append(f"{target} {arm} seed {seed}")
                else:
                    runs[(target, arm, seed)] = run

    print(f"loaded {len(runs)} of {len(TARGETS) * len(ARMS) * len(SEEDS)} runs")
    if missing:
        print(f"NOT RUN ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    print()

    # The three arms must have scored the same images with the same labels, or
    # a difference between them is not a difference in the model.
    for target in TARGETS:
        present = [k for k in runs if k[0] == target]
        if not present:
            continue
        anchor = runs[present[0]]
        for key in present:
            run = runs[key]
            assert len(run["y_true"]) == len(anchor["y_true"]), (
                f"{key}: scored {len(run['y_true'])} images, anchor "
                f"{len(anchor['y_true'])} -- different test sets")
            assert (run["y_true"] == anchor["y_true"]).all(), (
                f"{key}: labels disagree with {anchor['experiment_id']}")

    # ------------------------------------------------------- per-seed table
    per_seed_rows = []
    for (target, arm, seed), run in sorted(runs.items()):
        row = {
            "target": target, "arm": arm, "seed": seed,
            "image_size": ARMS[arm][0], "batch_size": ARMS[arm][1],
            "experiment_id": run["experiment_id"], "n_test": run["n"],
            "ece": run["ece"], "ece_scaled": run["ece_scaled"],
            "temperature": run["temperature"],
        }
        for name in BOOTSTRAP_METRICS:
            row[name] = metric_fn(name)(run["y_true"], run["y_pred"])
        per_seed_rows.append(row)

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed_path = tables / "configuration_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)
    print(f"saved -> {per_seed_path.name}  ({len(per_seed)} rows)")

    # ------------------------------------------------------- the contrasts
    records = []
    for name, reference_arm, candidate_arm, isolates in CONTRASTS:
        for target in TARGETS:
            reference, candidate, truth = {}, {}, None
            for seed in SEEDS:
                a = runs.get((target, reference_arm, seed))
                b = runs.get((target, candidate_arm, seed))
                if a is None or b is None:
                    continue
                reference[seed], candidate[seed] = a["y_pred"], b["y_pred"]
                truth = a["y_true"]

            paired = sorted(reference)
            if truth is None or len(paired) < 2:
                print(f"  {name}/{target}: NOT RUN "
                      f"({len(paired)} paired seed(s), need 2)")
                continue

            for metric_name in BOOTSTRAP_METRICS:
                result = crossed_bootstrap_difference(
                    truth, reference, candidate,
                    metric=metric_fn(metric_name),
                    n_bootstrap=n_bootstrap, seed=7)
                record = {
                    "contrast": name,
                    "isolates": isolates,
                    "reference_arm": reference_arm,
                    "candidate_arm": candidate_arm,
                    "target": target,
                    "metric": metric_name,
                    "n_seeds": result["n_seeds"],
                    "seeds": ",".join(str(s) for s in result["seeds"]),
                    "n_test": len(truth),
                    "delta": result["difference"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "p_value": result["p_value"],
                    "seed_sd": result["seed_sd"],
                    "sign_agreement": result["sign_agreement"],
                    "exceeds_seed_sd": result["exceeds_seed_sd"],
                }
                for seed, value in result["per_seed_difference"].items():
                    record[f"delta_seed{seed}"] = value
                records.append(record)

    if not records:
        print("\nnothing to compare yet -- no contrast has two paired seeds")
        return

    frame = pd.DataFrame(records)

    # Holm within each (contrast, metric) across held-out domains. Not across
    # contrasts: see the module docstring.
    frame["p_holm"] = float("nan")
    frame["holm_family"] = ""
    for (name, metric_name), group in frame.groupby(["contrast", "metric"]):
        adjusted = holm_adjust(group["p_value"].tolist())
        frame.loc[group.index, "p_holm"] = adjusted
        frame.loc[group.index, "holm_family"] = (
            f"{name}/{metric_name} across {len(group)} held-out domain(s)")

    path = tables / "configuration_decomposition.csv"
    frame.to_csv(path, index=False)
    print(f"saved -> {path.name}  ({len(frame)} rows)")

    # ---------------------------------------------------- additivity check
    # C - (A + B) on the per-seed means. Zero by identity when the seeds and
    # test sets match; a non-zero value is a bookkeeping fault, not a finding.
    print("\nadditivity check  C - (A + B), QWK, per-seed means:")
    for target in TARGETS:
        parts = {}
        for name in ("batch", "resolution", "combined"):
            hit = frame[(frame.contrast == name) & (frame.target == target)
                        & (frame.metric == "qwk")]
            if len(hit) == 1:
                parts[name] = float(hit.delta.iloc[0])
        if len(parts) == 3:
            residual = parts["combined"] - (parts["batch"] + parts["resolution"])
            flag = "ok" if abs(residual) < 1e-9 else "!! MISMATCHED SEEDS"
            print(f"  {target:8} {parts['combined']:+.4f} - ("
                  f"{parts['batch']:+.4f} {parts['resolution']:+.4f}) = "
                  f"{residual:+.2e}  {flag}")
        else:
            print(f"  {target:8} incomplete -- {sorted(parts)} of 3 contrasts")

    # ----------------------------------------------------------- the report
    print("\nQWK, by contrast:")
    header = (f"  {'contrast':11} {'target':8} {'seeds':5} {'delta':>8} "
              f"{'95% CI':>19} {'p':>7} {'p_holm':>7} {'sign':>5}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, _, _, isolates in CONTRASTS:
        for _, r in frame[(frame.contrast == name)
                          & (frame.metric == "qwk")].iterrows():
            interval = f"[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]"
            print(f"  {name:11} {r.target:8} {int(r.n_seeds):5} "
                  f"{r.delta:+8.4f} {interval:>19} {r.p_value:7.3f} "
                  f"{r.p_holm:7.3f} {int(r.sign_agreement)}/{int(r.n_seeds)}")

    print("\nin-domain resolution is NOT decomposed: the in-domain 512 px runs "
          "have no 224/b16 counterpart, so it remains a configuration effect.")


if __name__ == "__main__":
    main()
