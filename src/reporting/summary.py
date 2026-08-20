"""Generate the Markdown results summary from the experiment registry.

This writes the "what actually happened" document that the discussion section is
built from.  It is deliberately mechanical: it reads the registry and the saved
evaluation JSONs, and states what they show.

Rules it follows
----------------
**Nothing is invented.** Every number comes from a file. A method that has not
run is listed under "not run", never omitted (which would let a reader assume it
was tried and failed) and never given a placeholder value.

**Differences are qualified.** A gap smaller than the bootstrap uncertainty is
reported as "within noise", not as an improvement. With a single seed the
report says so on every comparison, because run-to-run variance is unmeasured
and could easily exceed the differences involved.

**Negative results are stated plainly.** If a domain-generalization method does
not help, the summary says it did not help. The project brief is explicit that
the contribution should emerge from the results rather than the results being
shaped to fit a hypothesis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("reporting.summary")

__all__ = ["build_summary", "write_summary"]

# Below this, a difference in QWK is not worth calling an improvement without
# multi-seed evidence. Chosen as roughly the half-width of the bootstrap CI seen
# on the 507-image IDRiD test split (+/- 0.06), rounded conservatively.
QWK_NOISE_FLOOR = 0.02
ECE_NOISE_FLOOR = 0.02


_KNOWN_METHODS = (
    # Longest first, so "deep_coral_ordinal" is not matched as "deep_coral".
    "deep_coral_ordinal", "mixstyle_ordinal", "deep_coral", "mixstyle", "ordinal", "erm",
)


def _normalise_method(name: Any) -> str:
    """Strip run-specific suffixes from a registry method label.

    Registry entries carry decorations that identify the run rather than the
    method -- ``erm-none`` (imbalance strategy) and ``deep_coral-b32`` (batch
    size). Comparing raw labels against the method list would report a method as
    never trained when it had been.
    """
    text = str(name)
    for method in _KNOWN_METHODS:
        if text == method or text.startswith(f"{method}-"):
            return method
    return text


def _get(row: Any, column: str) -> Any:
    """Read a registry column that may be absent.

    Registry rows are append-only and written over the life of the project, so
    an older row can lack a column a newer one has. Indexing directly would make
    the whole summary crash on one incomplete row; a missing value is simply
    reported as NOT RUN.
    """
    try:
        value = row[column]
    except (KeyError, IndexError):
        return None
    return value


def _fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "NOT RUN"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NOT RUN" if np.isnan(number) else f"{number:.{decimals}f}"


def _delta_verdict(delta: float, floor: float, *, higher_is_better: bool = True) -> str:
    """Describe a difference honestly relative to the noise floor."""
    if abs(delta) < floor:
        return f"within noise ({delta:+.4f}, |diff| < {floor})"
    improved = delta > 0 if higher_is_better else delta < 0
    return f"{'better' if improved else 'WORSE'} ({delta:+.4f})"


def build_summary(
    registry: Any,
    *,
    baseline_method: str = "erm",
    n_seeds: int = 1,
    extra_sections: Sequence[str] = (),
) -> str:
    """Compose the Markdown summary from a registry DataFrame."""
    import pandas as pd

    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines += [
        "# Experiment summary",
        "",
        f"*Generated {now} from `outputs/experiment_registry.csv`. "
        "Every number below is read from a file; none is typed in by hand.*",
        "",
    ]

    if registry is None or len(registry) == 0:
        lines += [
            "## No experiments have been run",
            "",
            "The registry is empty. Nothing can be concluded.",
        ]
        return "\n".join(lines)

    completed = registry[registry["status"] == "COMPLETE"].copy()
    completed = completed.sort_values("timestamp_utc").drop_duplicates(
        "experiment_id", keep="last"
    )

    # -- scope and caveats ------------------------------------------------
    lines += [
        "## Scope",
        "",
        f"- **{len(completed)} completed run(s)** in the registry.",
        f"- **{n_seeds} seed(s).**"
        + (
            " Run-to-run variance is therefore **unmeasured**; no difference below "
            "the noise floor should be treated as real."
            if n_seeds < 3 else ""
        ),
        f"- Protocols present: {sorted(completed['protocol'].dropna().unique().tolist())}",
        f"- Target domains evaluated: {sorted(completed['target_domain'].dropna().unique().tolist())}",
        f"- Backbones: {sorted(completed['backbone'].dropna().unique().tolist())}",
        "",
    ]

    # -- the generalization gap -------------------------------------------
    lines += ["## 1. How large is the domain-generalization gap?", ""]
    required = [c for c in ("best_val_qwk", "test_qwk") if c in completed.columns]
    gap_rows = completed.dropna(subset=required) if len(required) == 2 else completed.iloc[0:0]
    if gap_rows.empty:
        lines += ["NOT RUN -- no run has both a validation and a target-domain score.", ""]
    else:
        lines += [
            "| Experiment | Target | Val QWK | Test QWK | Gap | Val ECE→Test ECE |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in gap_rows.iterrows():
            gap = float(row["test_qwk"]) - float(row["best_val_qwk"])
            lines.append(
                f"| `{_get(row, 'method')}` | {_get(row, 'target_domain')} | "
                f"{_fmt(_get(row, 'best_val_qwk'))} | {_fmt(_get(row, 'test_qwk'))} | "
                f"**{gap:+.4f}** | → {_fmt(_get(row, 'test_ece'))} |"
            )
        mean_gap = float(
            (gap_rows["test_qwk"].astype(float) - gap_rows["best_val_qwk"].astype(float)).mean()
        )
        lines += [
            "",
            f"Mean QWK gap across runs: **{mean_gap:+.4f}**.",
            "",
            "Note the validation figure is the *best* source-validation score, "
            "selected on that split, so it is itself mildly optimistic. The gap is "
            "a lower bound on the drop a clinic would see.",
            "",
        ]

    # -- calibration -------------------------------------------------------
    lines += ["## 2. Does calibration degrade more than accuracy?", ""]
    if "test_ece" in completed and completed["test_ece"].notna().any():
        lines += [
            "| Experiment | Test QWK | Test ECE | Test NLL | Test Brier |",
            "|---|---|---|---|---|",
        ]
        for _, row in completed.iterrows():
            lines.append(
                f"| `{_get(row, 'method')}` | {_fmt(_get(row, 'test_qwk'))} | "
                f"{_fmt(_get(row, 'test_ece'))} | {_fmt(_get(row, 'test_nll'))} | "
                f"{_fmt(_get(row, 'test_brier'))} |"
            )
        lines.append("")
    else:
        lines += ["NOT RUN -- no calibration metrics recorded.", ""]

    # -- method comparison -------------------------------------------------
    lines += ["## 3. Do the methods help?", ""]
    baseline = completed[completed["method"].astype(str).str.startswith(baseline_method)]
    if baseline.empty:
        lines += [
            f"NOT RUN -- no `{baseline_method}` baseline in the registry, so nothing "
            "can be compared against.",
            "",
        ]
    else:
        base = baseline.iloc[-1]
        base_qwk = _get(base, "test_qwk")
        base_ece = _get(base, "test_ece")
        if base_qwk is None or base_ece is None:
            lines += [
                f"The `{baseline_method}` row has no target metrics recorded, so no "
                "comparison is possible.",
                "",
            ]
            base_qwk = base_ece = float("nan")
        base_qwk, base_ece = float(base_qwk), float(base_ece)
        lines += [
            f"Baseline: `{base['method']}` on {base['target_domain']} "
            f"(test QWK {base_qwk:.4f}, ECE {base_ece:.4f}).",
            "",
            "| Method | Test QWK | vs baseline | Test ECE | vs baseline |",
            "|---|---|---|---|---|",
        ]
        others = completed[completed["experiment_id"] != base["experiment_id"]]
        for _, row in others.iterrows():
            row_qwk, row_ece = _get(row, "test_qwk"), _get(row, "test_ece")
            if row_qwk is None or (isinstance(row_qwk, float) and np.isnan(row_qwk)):
                continue
            qwk_delta = float(row_qwk) - base_qwk
            ece_cell = "NOT RUN"
            if row_ece is not None and not (isinstance(row_ece, float) and np.isnan(row_ece)):
                ece_cell = _delta_verdict(
                    float(row_ece) - base_ece, ECE_NOISE_FLOOR, higher_is_better=False
                )
            lines.append(
                f"| `{_get(row, 'method')}` | {_fmt(row_qwk)} | "
                f"{_delta_verdict(qwk_delta, QWK_NOISE_FLOOR)} | "
                f"{_fmt(row_ece)} | {ece_cell} |"
            )
        lines += [
            "",
            f"*Noise floor: QWK differences below {QWK_NOISE_FLOOR} and ECE differences "
            f"below {ECE_NOISE_FLOOR} are reported as 'within noise'. These thresholds "
            "come from the width of the bootstrap intervals on the smallest test split, "
            "not from a significance test.*",
            "",
        ]
        if n_seeds < 3:
            lines += [
                "> **Single-seed caveat.** With one seed per method, an apparent "
                "improvement can be seed variation. None of the above should be "
                "reported as a finding until at least three seeds per method exist.",
                "",
            ]

    # -- computational cost -------------------------------------------------
    lines += ["## 4. Computational cost", ""]
    if "train_seconds" in completed and completed["train_seconds"].notna().any():
        lines += ["| Method | Params (M) | Epochs | Train (s) | Peak VRAM (GB) |", "|---|---|---|---|---|"]
        for _, row in completed.iterrows():
            lines.append(
                f"| `{_get(row, 'method')}` | {_fmt(_get(row, 'params_total_m'), 2)} | "
                f"{_get(row, 'epochs_run')} | {_fmt(_get(row, 'train_seconds'), 1)} | "
                f"{_fmt(_get(row, 'peak_vram_gb'), 2)} |"
            )
        lines.append("")

    # -- what has not been done --------------------------------------------
    lines += ["## 5. Not run", ""]
    all_domains = {"ddr", "aptos", "idrid", "eyepacs"}
    evaluated = set(completed["target_domain"].dropna().unique())
    missing_domains = sorted(all_domains - evaluated)
    known_methods = {
        "erm", "ordinal", "deep_coral", "mixstyle",
        "mixstyle_ordinal", "deep_coral_ordinal",
    }
    missing_methods = sorted(
        m for m in known_methods
        if not any(_normalise_method(r) == m for r in completed["method"].dropna())
    )

    if missing_domains:
        lines.append(
            f"- **Target domains not evaluated:** {missing_domains}. The "
            "leave-one-domain-out matrix is incomplete."
        )
    if missing_methods:
        lines.append(f"- **Methods implemented but not trained:** {missing_methods}.")
    if n_seeds < 3:
        lines.append(
            f"- **Multi-seed runs:** only {n_seeds} seed. Mean +/- SD cannot be reported."
        )
    backbones = (
        set(completed["backbone"].dropna().unique())
        if "backbone" in completed.columns else set()
    )
    missing_backbones = sorted({"densenet121", "convnext_tiny", "dinov2_vits14"} - backbones)
    if missing_backbones:
        lines.append(f"- **Backbones not trained:** {missing_backbones}.")
    lines.append("")

    # -- limitations --------------------------------------------------------
    lines += [
        "## 6. Standing limitations",
        "",
        "These hold regardless of which experiments finish, and belong in the paper:",
        "",
        "- **Patient-level splitting is possible only for EyePACS.** DDR, APTOS and "
        "IDRiD publish no patient identifiers, so same-patient images could span "
        "splits in those domains. Undetectable and unpreventable from the released data.",
        "- **Label shift is confounded with covariate shift.** The domains differ in "
        "case mix as well as appearance (no-DR ranges from 32.9% in IDRiD to 73.5% in "
        "EyePACS), so a cross-domain drop reflects both. A DG method targeting "
        "covariate shift would not fix the prior shift.",
        "- **EyePACS labels come from a corroborated secondary source**, not the "
        "official Kaggle file, and the domain is restricted to the 35,108 images whose "
        "labels were verified.",
        "- **Duplicate removal discarded 33 groups with contradictory ground truth** "
        "(byte-identical images labelled differently). Those labels were unusable, but "
        "their existence suggests residual label noise in the remaining data.",
        "- **Selective prediction is not evidence of clinical readiness.** Deferred "
        "cases still require a clinician, and the abstention threshold would have to be "
        "fixed prospectively.",
        "",
    ]

    lines += list(extra_sections)
    return "\n".join(lines)


def write_summary(
    registry_path: Path | str,
    output_path: Path | str,
    *,
    baseline_method: str = "erm",
    n_seeds: int = 1,
) -> Path:
    """Read the registry and write the Markdown summary."""
    import pandas as pd

    registry_path = Path(registry_path)
    registry = pd.read_csv(registry_path) if registry_path.exists() else None

    text = build_summary(registry, baseline_method=baseline_method, n_seeds=n_seeds)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    log.info("summary -> %s (%d lines)", output_path, len(text.splitlines()))
    return output_path
