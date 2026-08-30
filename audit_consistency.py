"""Check that every derived artifact still agrees with the authoritative record.

Two results were silently destroyed on 2026-08-22, both the same way: a summary
table was keyed on too few columns, so a new run replaced an old row instead of
joining it. Neither failed loudly. Both files still held the right *number* of
well-formed rows afterwards, which is exactly why nothing caught them --
there is no shape check, type check or schema check that these bugs violate.

What they do violate is agreement with the experiment registry, whose key is the
experiment id and therefore encodes protocol, sources, target, backbone,
resolution, method and seed. This script checks that agreement.

Five checks, in increasing order of cost:

1. **Key uniqueness** -- no two rows in a summary table share a key.
2. **Orphan runs** -- every completed run in the registry appears in its summary
   table. *This is the check that catches a silent overwrite*: the tell is a
   registry row with no corresponding table row.
3. **Registry agreement** -- table metrics equal the registry's for the same run.
4. **Prediction agreement** -- metrics recomputed from saved per-image
   predictions equal the registry's. Catches a corrupted or truncated
   predictions file.
5. **LaTeX agreement** -- each exported .tex equals what the exporter produces
   right now, so a stale or hand-edited table cannot reach the manuscript. A
   stale one is regenerated in place and reported.

Exit code is non-zero if anything fails, so this can gate a commit or a release.

Runs from saved artifacts. No GPU. Takes about five seconds.

Usage:
    python audit_consistency.py
    python audit_consistency.py --quick     # skip check 4 (re-scoring predictions)
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
TOLERANCE = 1e-6

# Columns that were added to the summary tables after some rows had been
# written. A row lacking them predates any run at another setting, so it is
# backfilled to the project default rather than reported as a fault.
DEFAULTS = {"backbone": "densenet121", "image_size": 224, "batch_size": 32,
            "domain_balanced": False, "irm_anneal_iters": 500,
            "trainable_blocks": -1, "learning_rate": 3e-4}


class Audit:
    """Collects failures rather than raising, so one run reports everything."""

    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.checks = 0
        self.skipped: list[str] = []

    def check(self, ok: bool, group: str, message: str) -> bool:
        self.checks += 1
        if not ok:
            self.failures.append((group, message))
        return ok

    def skip(self, message: str) -> None:
        self.skipped.append(message)

    def report(self) -> int:
        print("\n" + "=" * 96)
        if not self.failures:
            print(f"PASS -- {self.checks} checks, no disagreement found")
        else:
            print(f"FAIL -- {len(self.failures)} of {self.checks} checks failed")
            group = None
            for failed_group, message in self.failures:
                if failed_group != group:
                    print(f"\n  [{failed_group}]")
                    group = failed_group
                print(f"    {message}")
        if self.skipped:
            print(f"\n  skipped ({len(self.skipped)}):")
            for message in self.skipped:
                print(f"    {message}")
        print("=" * 96)
        return 1 if self.failures else 0


def _with_defaults(frame):
    """Backfill the late-added key columns."""
    frame = frame.copy()
    for column, value in DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = value
        else:
            frame[column] = frame[column].fillna(value)
    return frame


def _registry_backbone(name: str) -> str:
    """The registry writes convnext_tiny; experiment ids use convnext-tiny."""
    return str(name).replace("_", "-")


def specifications():
    """One entry per summary table: how to key it and how to name its runs."""
    from src.utils.registry import make_experiment_id

    def lodo_id(row):
        return make_experiment_id(
            protocol="lodo",
            sources=[d for d in ALL_DOMAINS if d != row["target"]],
            target=row["target"], backbone=row["backbone"],
            method=_method_tag(row), seed=int(row["seed"]),
        )

    def in_domain_id(row):
        return make_experiment_id(
            protocol="in_domain", sources=[row["domain"]], target=row["domain"],
            backbone=row["backbone"], method=_method_tag(row), seed=int(row["seed"]),
        )

    def single_source_id(row):
        return make_experiment_id(
            protocol="single_source", sources=[row["source"]], target=row["target"],
            backbone=row["backbone"], method=_method_tag(row), seed=int(row["seed"]),
        )

    def _method_tag(row) -> str:
        tag = f"{row['method']}-b{int(row['batch_size'])}"
        if int(row["image_size"]) != 224:
            tag += f"-r{int(row['image_size'])}"
        # Mirrors run_lodo._method_tag. Without this the audit rebuilds the
        # ordinary-sampler id for a domain-balanced row, then either matches the
        # wrong run or reports both as orphans.
        if bool(row.get("domain_balanced", False)):
            tag += "-dbal"
        # Mirrors run_lodo._method_tag. A non-default IRM anneal is a separate
        # configuration with its own id; without this the audit rebuilds
        # irm-b32 for an irm-b32-a1170 row and reports it as unregistered.
        anneal = row.get("irm_anneal_iters", 500)
        if row["method"] == "irm" and anneal == anneal and int(anneal) != 500:
            tag += f"-a{int(anneal)}"
        # Partial fine-tuning and a non-default rate, mirroring run_lodo again.
        # -1 means the whole network trained, which is what every run before the
        # flag existed did, so it adds nothing to the tag.
        blocks = row.get("trainable_blocks", -1)
        if blocks == blocks and int(blocks) >= 0:
            tag += f"-tb{int(blocks)}"
        rate = row.get("learning_rate", 3e-4)
        if rate == rate and float(rate) != 3e-4:
            tag += f"-lr{float(rate):g}"
        return tag

    def n_sources(registry_row) -> int:
        import json

        return len(json.loads(registry_row["source_domains"]))

    def is_subsample(registry_row) -> bool:
        """Sweep runs register as protocol=lodo, method=erm, three sources.

        They are indistinguishable from the LODO matrix by those fields alone,
        but they are summarised in subsample_sweep.csv rather than
        lodo_results.csv. Only the fraction tag in the experiment id separates
        them.
        """
        import re

        return bool(re.search(r"-f\d{3}_s\d+$", str(registry_row["experiment_id"])))

    def is_linear_probe(registry_row) -> bool:
        """Frozen-feature linear probes register as protocol=lodo, three sources.

        They are a different protocol from everything else in lodo_results.csv:
        the backbone is frozen and only a linear head is trained. Pooling them
        into that table would put a linear probe and a fine-tuned network in the
        same seed mean, which is how a Phase 5 summary would quietly acquire a
        row that answers a different question. They are summarised in
        linear_probe_results.csv and audited against it below.
        """
        return str(registry_row["method"]) == "linprobe"

    def probe_id(row):
        return make_experiment_id(
            protocol="lodo", sources=[d for d in ALL_DOMAINS if d != row["target"]],
            target=row["target"], backbone=row["backbone"],
            method=f"{row['method']}-b{int(row['batch_size'])}",
            seed=int(row["seed"]),
        )

    def subsample_id(row):
        return make_experiment_id(
            protocol="lodo", sources=[d for d in ALL_DOMAINS if d != row["target"]],
            target=row["target"], backbone=row["backbone"],
            method=f"{row['method']}-b{int(row['batch_size'])}"
                   f"-f{int(round(float(row['fraction']) * 100)):03d}",
            seed=int(row["seed"]),
        )

    return [
        {
            "file": "lodo_results*.csv", "protocol": "lodo",
            # The registry's protocol="lodo" also covers the Stage-C method
            # comparison, which trains on two sources and is summarised in
            # stage_c_*.csv. Without this scope every Stage-C run reads as a
            # row missing from the LODO matrix.
            # Every method run through run_lodo.py lands in this table, not just
            # ERM: the DG comparison (deep_coral, mixstyle, groupdro, irm) writes
            # here too. Scoping this to ERM would leave those runs matching no
            # spec at all, which the orphan check reports -- and, worse, would
            # leave their table rows unverified against the registry.
            #
            # Two-source runs are the Stage-C comparison, summarised in
            # stage_c_*.csv and audited there.
            #
            # Frozen-feature linear probes are excluded for the same reason and
            # audited against linear_probe_results.csv instead.
            "scope": lambda r: (n_sources(r) == 3 and not is_subsample(r)
                                and not is_linear_probe(r)),
            # batch_size is a key for the same reason domain_balanced is: it
            # appears in the experiment id but not in the row's "method"
            # column, so the Q1 batch-16 controls and the batch-32 runs are
            # indistinguishable by the other fields. It has to match the key
            # run_lodo.py merges on, or the audit calls a legitimate pair of
            # rows a duplicate.
            "key": ["target", "method", "seed", "backbone", "image_size",
                    "batch_size", "domain_balanced", "irm_anneal_iters",
                    "trainable_blocks", "learning_rate"],
            "experiment_id": lodo_id,
            "target_of": lambda row: row["target"],
            # table column -> registry column
            "registry": {"target_qwk": "test_qwk", "target_f1": "test_f1_macro",
                         "target_ece": "test_ece",
                         "target_severe": "test_severe_error_rate",
                         "n_train": "n_train", "n_test": "n_test"},
        },
        {
            "file": "in_domain_results*.csv", "protocol": "in_domain",
            "key": ["domain", "method", "seed", "backbone", "image_size"],
            "experiment_id": in_domain_id,
            "target_of": lambda row: row["domain"],
            "registry": {"test_qwk": "test_qwk", "test_f1": "test_f1_macro",
                         "test_ece": "test_ece",
                         "test_severe": "test_severe_error_rate",
                         "n_train": "n_train", "n_test": "n_test"},
        },
        {
            # The sweep, excluded from the LODO scope above and checked here
            # instead -- silencing a false positive without auditing the runs
            # somewhere would leave nine experiments unverified.
            "file": "subsample_sweep*.csv", "protocol": "lodo",
            "scope": is_subsample,
            "key": ["target", "method", "seed", "backbone", "image_size", "fraction"],
            "experiment_id": subsample_id,
            "target_of": lambda row: row["target"],
            "registry": {"target_qwk": "test_qwk", "target_f1": "test_f1_macro",
                         "target_ece": "test_ece",
                         "target_severe": "test_severe_error_rate",
                         "n_train": "n_train", "n_test": "n_test"},
        },
        {
            # Frozen-feature linear probes, excluded from the LODO scope above.
            # Auditing them somewhere is the point: silencing the orphan report
            # without checking them would leave nine experiments unverified.
            "file": "linear_probe_results*.csv", "protocol": "lodo",
            "scope": is_linear_probe,
            "key": ["target", "method", "seed", "backbone", "image_size"],
            "experiment_id": probe_id,
            "target_of": lambda row: row["target"],
            "registry": {"target_qwk": "test_qwk", "target_f1": "test_f1_macro",
                         "target_ece": "test_ece",
                         "target_severe": "test_severe_error_rate",
                         "n_train": "n_train", "n_test": "n_test"},
        },
        {
            "file": "single_source_results*.csv", "protocol": "single_source",
            "key": ["source", "target", "method", "seed", "backbone", "image_size"],
            "experiment_id": single_source_id,
            "target_of": lambda row: row["target"],
            "registry": {"target_qwk": "test_qwk", "target_f1": "test_f1_macro",
                         "target_ece": "test_ece",
                         "target_severe": "test_severe_error_rate",
                         "n_test": "n_test"},
        },
    ]


def audit_tables(audit: Audit, outputs, registry, *, quick: bool) -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.calibration import expected_calibration_error
    from src.evaluation.metrics import compute_all_metrics

    # The registry is an append-only log, so a re-run leaves two rows under one
    # experiment_id. Resolve to the current state before indexing; without this
    # .loc returns a two-row Series and every comparison against it raises.
    from src.utils.registry import latest_per_experiment

    indexed = latest_per_experiment(registry).set_index("experiment_id")

    for spec in specifications():
        # A glob, because run_lodo.py scopes its table by resolution:
        # lodo_results.csv and lodo_results_r512.csv are two files holding one
        # logical table. Reading only the first would report every 512 px run
        # as missing.
        paths = sorted((outputs / "tables").glob(spec["file"]))
        name = spec["file"]
        if not paths:
            audit.skip(f"{name}: NOT RUN")
            continue
        frame = _with_defaults(
            pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        )

        # -- 1. key uniqueness ------------------------------------------------
        duplicated = frame[frame.duplicated(spec["key"], keep=False)]
        audit.check(
            duplicated.empty, "key uniqueness",
            f"{name}: {len(duplicated)} row(s) share a key "
            f"{spec['key']} -- a mean over them double-counts a run"
            if not duplicated.empty else "",
        )

        # -- 2. orphan runs ---------------------------------------------------
        # A registry run of this protocol with no table row is the signature of
        # a silent overwrite: the run happened, its row is gone.
        ids_in_table = set()
        for _, row in frame.iterrows():
            try:
                ids_in_table.add(spec["experiment_id"](row))
            except Exception as exc:  # noqa: BLE001
                audit.check(False, "registry linkage",
                            f"{name}: cannot build an id for {dict(row[spec['key']])}: {exc}")

        expected = registry[registry["protocol"] == spec["protocol"]]
        if "scope" in spec:
            expected = expected[expected.apply(spec["scope"], axis=1)]
        for experiment_id in expected["experiment_id"]:
            audit.check(
                experiment_id in ids_in_table, "orphan runs",
                f"{name}: registry has {experiment_id} but the table has no such row "
                f"-- a completed run is missing from its summary table",
            )

        # -- 3 & 4. value agreement ------------------------------------------
        for _, row in frame.iterrows():
            try:
                experiment_id = spec["experiment_id"](row)
            except Exception:  # noqa: BLE001
                continue
            if experiment_id not in indexed.index:
                audit.check(False, "registry linkage",
                            f"{name}: row {dict(row[spec['key']])} maps to "
                            f"{experiment_id}, which is not in the registry")
                continue
            reference = indexed.loc[experiment_id]

            for table_column, registry_column in spec["registry"].items():
                if table_column not in frame.columns or registry_column not in registry.columns:
                    continue
                mine, theirs = row[table_column], reference[registry_column]
                if pd.isna(mine) or pd.isna(theirs):
                    continue
                audit.check(
                    abs(float(mine) - float(theirs)) < TOLERANCE, "registry agreement",
                    f"{name}: {experiment_id} column {table_column} is "
                    f"{float(mine):.6f} but the registry says {float(theirs):.6f} "
                    f"(delta {abs(float(mine) - float(theirs)):.2e})",
                )

            if quick:
                continue

            target = spec["target_of"](row)
            predictions = (outputs / "predictions"
                           / f"{experiment_id}__target_test[{target}]_predictions.csv")
            if not predictions.exists():
                audit.check(False, "prediction agreement",
                            f"{name}: {experiment_id} has a registry row but no "
                            f"predictions file")
                continue
            saved = pd.read_csv(predictions)
            recomputed = compute_all_metrics(saved["true_grade"].to_numpy(),
                                             saved["predicted_grade"].to_numpy())
            columns = sorted((c for c in saved.columns if c.startswith("probability_grade_")),
                             key=lambda c: int(c.rsplit("_", 1)[1]))
            probabilities = np.asarray(saved[columns].to_numpy(), dtype=np.float64)

            for metric, registry_column in [("qwk", "test_qwk"),
                                            ("f1_macro", "test_f1_macro"),
                                            ("severe_error_rate", "test_severe_error_rate")]:
                if registry_column not in registry.columns or pd.isna(reference[registry_column]):
                    continue
                audit.check(
                    abs(recomputed[metric] - float(reference[registry_column])) < TOLERANCE,
                    "prediction agreement",
                    f"{name}: {experiment_id} {metric} recomputed from predictions is "
                    f"{recomputed[metric]:.6f} but the registry says "
                    f"{float(reference[registry_column]):.6f}",
                )
            if "test_ece" in registry.columns and not pd.isna(reference["test_ece"]):
                audit.check(
                    abs(expected_calibration_error(probabilities,
                                                   saved["true_grade"].to_numpy())
                        - float(reference["test_ece"])) < TOLERANCE,
                    "prediction agreement",
                    f"{name}: {experiment_id} ECE recomputed from predictions "
                    f"disagrees with the registry",
                )


def audit_latex(audit: Audit, outputs) -> None:
    """An exported table must equal what the exporter would produce right now.

    Guards the last step before the manuscript. A .tex edited by hand, or
    exported from a CSV that has since been corrected, disagrees with the
    analysis and the disagreement is invisible on inspection --
    ``table_lodo.tex`` sat for a day carrying an across-architecture SD labelled
    as an across-seed SD.

    Checking that each number in the .tex appears somewhere in the source CSV
    does not work: most of these tables aggregate, so ``0.7383 $\\pm$ 0.0055``
    is a mean and an SD that appear in no cell. Regenerating and diffing tests
    the real property -- is this file what the pipeline produces? -- and needs
    no duplicate of the exporter's arithmetic.

    Regeneration overwrites a stale table, which is the correct outcome: the
    fix is applied and reported rather than merely announced.
    """
    import contextlib
    import importlib
    import io

    tables = outputs / "tables"
    before = {path.name: path.read_text(encoding="utf-8")
              for path in sorted(tables.glob("table_*.tex"))}
    if not before:
        audit.skip("no exported .tex tables to verify")
        return

    exporter = importlib.import_module("export_paper_tables")
    try:
        # The exporter narrates every file it writes, which here is noise
        # around the one line that matters -- whether anything changed.
        with contextlib.redirect_stdout(io.StringIO()):
            exporter.main()
    except Exception as exc:  # noqa: BLE001
        audit.check(False, "latex agreement",
                    f"export_paper_tables.py failed to run: {type(exc).__name__}: {exc}")
        return

    for name, original in before.items():
        current = (tables / name).read_text(encoding="utf-8")
        audit.check(
            current == original, "latex agreement",
            f"{name} did not match what export_paper_tables.py produces -- it was "
            f"stale and has now been regenerated. Re-check any manuscript text "
            f"quoting it.",
        )

    fresh = {path.name for path in tables.glob("table_*.tex")} - set(before)
    for name in sorted(fresh):
        audit.check(False, "latex agreement",
                    f"{name} did not exist before this run; the exporter has a "
                    f"table the repository was missing")


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    quick = "--quick" in sys.argv[1:]
    outputs = project_root() / "outputs"
    registry_path = outputs / "experiment_registry.csv"
    if not registry_path.exists():
        print(f"NOT RUN -- {registry_path} does not exist")
        return 1

    registry = pd.read_csv(registry_path)
    registry = registry[registry["status"] == "COMPLETE"]
    registry["backbone"] = registry["backbone"].map(_registry_backbone)
    # Distinct experiments, not log rows: a re-run under an unchanged id leaves
    # a superseded entry behind, and counting those would overstate the work.
    _distinct = registry["experiment_id"].nunique()
    _superseded = len(registry) - _distinct
    print(f"auditing {_distinct} completed runs against their summary tables"
          + (f" ({_superseded} superseded entries skipped)" if _superseded else "")
          + (" (quick: predictions not re-scored)" if quick else ""))

    audit = Audit()
    audit_tables(audit, outputs, registry, quick=quick)
    audit_latex(audit, outputs)
    return audit.report()


if __name__ == "__main__":
    raise SystemExit(main())
