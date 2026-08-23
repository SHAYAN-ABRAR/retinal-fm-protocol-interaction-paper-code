"""Rebuild the per-run performance figures from saved predictions.

Why this exists
---------------
Figures are cheap to produce and expensive to notice as stale. The
``perf_*``/``train_*``/``errors_*`` files written during Phase 3 were keyed to a
batch-16 run that has since been superseded twice -- by the batch-32 method
comparison and then by the full LODO matrix. They render correctly and look
current, so a figure pulled into a draft would silently contradict the tables
next to it.

Rather than delete them (results are never deleted in this project), stale files
are moved to ``outputs/figures/superseded/`` with a README recording what
replaced them and why.

Everything here is rebuilt from the saved per-image predictions and the stored
evaluation JSON. No model is re-run and no GPU is needed, so the figures cannot
drift from the numbers they are captioned with.

Usage:
    python regenerate_figures.py                 # LODO ERM seed 42, all targets
    python regenerate_figures.py --archive-only  # just move stale files aside
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BACKBONE = "densenet121"
BATCH_SIZE = 32
ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]

# Superseded prefixes -> what replaced them.
STALE_PREFIXES = {
    "perf_lodo_aptos-ddr__idrid_densenet121_erm-none_s42":
        "batch-16 Phase-3 baseline; superseded by the batch-32 LODO runs",
    "train_lodo_aptos-ddr__idrid_densenet121_erm-none_s42":
        "batch-16 Phase-3 baseline; superseded by the batch-32 LODO runs",
    "gap_lodo_aptos-ddr__idrid_densenet121_erm-none_s42":
        "batch-16 Phase-3 baseline; superseded by the batch-32 LODO runs",
    "errors_baseline_idrid":
        "built from the batch-16 baseline; superseded by the batch-32 LODO runs",
    "domain_matrix_grid_erm.png":
        "pooled several seeds and a superseded run into one cell; "
        "replaced by domain_matrix_grid_erm_s42.png",
    "domain_matrix_qwk_erm.png":
        "pooled several seeds and a superseded run into one cell; "
        "replaced by domain_matrix_qwk_erm_s42.png",
    "domain_matrix_ece_erm.png":
        "pooled several seeds and a superseded run into one cell; "
        "replaced by domain_matrix_ece_erm_s42.png",
}


def _experiment_id(target: str, method: str, seed: int) -> str:
    from src.utils.registry import make_experiment_id

    sources = [d for d in ALL_TARGETS if d != target]
    return make_experiment_id(
        protocol="lodo", sources=sources, target=target,
        backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
    )


def archive_stale(figures_dir) -> list[str]:
    """Move superseded figures aside, recording what replaced each one."""
    from pathlib import Path

    figures_dir = Path(figures_dir)
    archive = figures_dir / "superseded"
    archive.mkdir(exist_ok=True)

    moved: list[tuple[str, str]] = []
    for path in sorted(figures_dir.glob("*.png")):
        for prefix, reason in STALE_PREFIXES.items():
            if path.name == prefix or path.name.startswith(prefix):
                path.replace(archive / path.name)
                moved.append((path.name, reason))
                break

    if moved:
        lines = [
            "# Superseded figures",
            "",
            "These are kept, not deleted -- they were correct for the run that",
            "produced them. They are here because a newer run replaced that one,",
            "so they must not be pulled into the paper.",
            "",
        ]
        for name, reason in moved:
            lines.append(f"- `{name}` -- {reason}")
        (archive / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return [name for name, _ in moved]


def regenerate_for_target(target: str, method: str, seed: int) -> list[str]:
    import json

    import numpy as np
    import pandas as pd

    from src.evaluation.metrics import compute_all_metrics
    from src.utils.io import project_root
    from src.visualization.calibration_figures import scaled_probabilities
    from src.visualization.error_figures import generate_error_figures
    from src.visualization.performance_figures import (
        figure_confidence_histogram,
        figure_confusion_matrices,
        figure_per_class_metrics,
        figure_reliability_diagram,
        figure_roc_curves,
    )
    from src.visualization.style import apply_style, save_figure
    from src.visualization.training_figures import generate_training_figures

    apply_style()
    outputs = project_root() / "outputs"
    figures = outputs / "figures"
    experiment_id = _experiment_id(target, method, seed)

    report_path = outputs / "reports" / f"{experiment_id}_evaluation.json"
    if not report_path.exists():
        print(f"  {target}: NOT RUN -- no evaluation report")
        return []
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    temperature = float(payload["temperature"]["temperature"])

    written: list[str] = []
    for split, filename in (
        ("target_test", f"{experiment_id}__target_test[{target}]_predictions.csv"),
        ("source_val", f"{experiment_id}__source_val_predictions.csv"),
    ):
        path = outputs / "predictions" / filename
        if not path.exists():
            print(f"  {target}/{split}: NOT RUN -- no predictions file")
            continue

        frame = pd.read_csv(path)
        columns = sorted(
            (c for c in frame.columns if c.startswith("probability_grade_")),
            key=lambda c: int(c.rsplit("_", 1)[1]),
        )
        probabilities = frame[columns].to_numpy(dtype=np.float64)
        y_true = frame["true_grade"].to_numpy()
        y_pred = frame["predicted_grade"].to_numpy()
        scaled = scaled_probabilities(probabilities, temperature)

        # Recomputed from the predictions rather than read from the JSON, so a
        # figure can never disagree with the array it was drawn from.
        metrics = compute_all_metrics(y_true, y_pred, probabilities)
        suffix = f" -- {target} ({split})"
        prefix = f"perf_{experiment_id}_{split}"

        for figure, name in (
            (figure_confusion_matrices(metrics, title_suffix=suffix), "confusion"),
            (figure_per_class_metrics(metrics, title_suffix=suffix), "per_class"),
            (figure_roc_curves(y_true, probabilities, title_suffix=suffix), "roc"),
            (figure_reliability_diagram(
                probabilities, y_true, scaled_probabilities=scaled,
                title_suffix=suffix), "reliability"),
            (figure_confidence_histogram(
                probabilities, y_true, title_suffix=suffix), "confidence"),
        ):
            written.extend(p.name for p in save_figure(figure, f"{prefix}_{name}", figures))

    history_path = outputs / "logs" / f"{experiment_id}_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        written.extend(
            p.name for p in generate_training_figures(
                history, figures, prefix=f"train_{experiment_id}")
        )

    target_predictions = (
        outputs / "predictions" / f"{experiment_id}__target_test[{target}]_predictions.csv"
    )
    if target_predictions.exists():
        try:
            written.extend(
                p.name for p in generate_error_figures(
                    pd.read_csv(target_predictions), figures,
                    prefix=f"errors_{experiment_id}")
            )
        except Exception as exc:  # noqa: BLE001 - error galleries are optional
            print(f"  {target}: error figures skipped ({type(exc).__name__}: {exc})")

    return written


def main() -> None:
    from src.utils.io import project_root

    arguments = sys.argv[1:]
    archive_only = "--archive-only" in arguments
    if archive_only:
        arguments.remove("--archive-only")

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    method = _take("--method", "erm")
    seed = int(_take("--seed", "42"))
    targets = _take("--targets", ",".join(ALL_TARGETS)).split(",")

    figures = project_root() / "outputs" / "figures"
    moved = archive_stale(figures)
    print(f"archived {len(moved)} superseded figure(s) -> {figures / 'superseded'}")
    for name in moved:
        print(f"  moved {name}")

    if archive_only:
        return

    total = 0
    for target in targets:
        print(f"\nregenerating for target {target}:")
        written = regenerate_for_target(target, method, seed)
        total += len(written)
        for name in written:
            print(f"  {name}")

    print(f"\nwrote {total} figure(s) from saved predictions -- no model was re-run")


if __name__ == "__main__":
    main()
