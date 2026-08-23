"""Publication-ready CSV and LaTeX tables, built from the experiment registry.

Everything here reads `outputs/experiment_registry.csv` and the per-experiment
artefacts beside it.  Nothing is typed in by hand, so a table cannot drift out of
sync with the run that produced it.

Formatting rules
----------------
**Bold marks the best value only where "best" is well defined**, and only when
the winner is separated from the runner-up by more than the reported
uncertainty. Bolding a 0.001 difference implies a distinction the data does not
support; :func:`bold_best` takes a tolerance and refuses to bold a tie.

**Rounding never flatters.** Values are rounded once, at the declared number of
decimals, using round-half-to-even. There is no per-column rounding chosen to
make a difference look larger.

**Missing is missing.** An experiment that has not run appears as ``NOT RUN``,
never as a blank that could be mistaken for zero, and never as an interpolated
value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("reporting.tables")

__all__ = [
    "NOT_RUN",
    "NOT_AVAILABLE",
    "bold_best",
    "to_latex",
    "save_table",
    "table_dataset_characteristics",
    "table_method_comparison",
    "table_calibration",
    "table_selective_prediction",
    "table_per_class",
    "table_computational_cost",
]

# Two distinct kinds of absence, never conflated:
#   NOT_RUN       -- an experiment we have not performed yet
#   NOT_AVAILABLE -- a quantity the data does not contain and never will
#                    (e.g. patient ids for DDR/APTOS/IDRiD)
NOT_RUN = "NOT RUN"
NOT_AVAILABLE = "n/a"

# Metrics where a *lower* value is better.
LOWER_IS_BETTER = {
    "ece", "ece_scaled", "adaptive_ece", "nll", "brier", "mae_grade",
    "severe_error_rate", "aurc", "test_ece", "test_nll", "test_brier",
    "test_mae_grade", "test_severe_error_rate", "risk",
}

_LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(text: Any) -> str:
    out = str(text)
    for character, replacement in _LATEX_ESCAPES.items():
        out = out.replace(character, replacement)
    return out


def bold_best(
    values: Sequence[float],
    *,
    lower_is_better: bool = False,
    tolerance: float = 0.0,
    decimals: int = 4,
) -> list[str]:
    """Format a column, bolding the best value unless it ties within ``tolerance``.

    ``tolerance`` should be set from the reported uncertainty -- typically the
    half-width of the bootstrap CI. With the default 0 the best value is always
    bolded, which is only appropriate when uncertainty is shown separately.
    """
    numeric = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    if np.all(np.isnan(numeric)):
        return [NOT_RUN] * len(values)

    best = np.nanmin(numeric) if lower_is_better else np.nanmax(numeric)
    formatted: list[str] = []
    for value in numeric:
        if np.isnan(value):
            formatted.append(NOT_RUN)
            continue
        text = f"{value:.{decimals}f}"
        is_best = abs(value - best) <= tolerance if tolerance > 0 else value == best
        # A tie inside the tolerance is not a win: leave every tied entry plain.
        n_within = int(np.sum(np.abs(numeric - best) <= tolerance)) if tolerance > 0 else 1
        formatted.append(f"\\textbf{{{text}}}" if is_best and n_within == 1 else text)
    return formatted


def to_latex(
    frame: Any,
    *,
    caption: str,
    label: str,
    decimals: int = 4,
    bold_columns: Sequence[str] = (),
    tolerances: dict[str, float] | None = None,
    column_format: str | None = None,
    note: str | None = None,
) -> str:
    """Render a DataFrame as a booktabs LaTeX table."""
    import pandas as pd

    tolerances = tolerances or {}
    display = frame.copy()

    for column in display.columns:
        if column in bold_columns and pd.api.types.is_numeric_dtype(display[column]):
            display[column] = bold_best(
                display[column].tolist(),
                lower_is_better=column.lower() in LOWER_IS_BETTER,
                tolerance=tolerances.get(column, 0.0),
                decimals=decimals,
            )
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = [
                NOT_RUN if v is None or (isinstance(v, float) and np.isnan(v))
                else f"{v:.{decimals}f}" if isinstance(v, float) else str(v)
                for v in display[column]
            ]
        else:
            display[column] = [
                NOT_RUN if v is None or (isinstance(v, float) and np.isnan(v)) else _escape(v)
                for v in display[column]
            ]

    header = " & ".join(_escape(c) for c in display.columns) + r" \\"
    body = "\n".join(" & ".join(str(v) for v in row) + r" \\" for row in display.itertuples(index=False))
    fmt = column_format or ("l" + "r" * (len(display.columns) - 1))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{fmt}}}",
        r"\toprule",
        header,
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines.append(rf"\vspace{{2pt}}\\\footnotesize{{{note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_table(
    frame: Any,
    name: str,
    tables_dir: Path | str,
    *,
    caption: str,
    label: str,
    decimals: int = 4,
    bold_columns: Sequence[str] = (),
    tolerances: dict[str, float] | None = None,
    note: str | None = None,
) -> dict[str, Path]:
    """Write both the CSV (for analysis) and the LaTeX (for the paper)."""
    directory = ensure_dir(tables_dir)
    csv_path = directory / f"{name}.csv"
    tex_path = directory / f"{name}.tex"

    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        to_latex(
            frame, caption=caption, label=label, decimals=decimals,
            bold_columns=bold_columns, tolerances=tolerances, note=note,
        ),
        encoding="utf-8",
    )
    log.info("table %s -> %s, %s", name, csv_path.name, tex_path.name)
    return {"csv": csv_path, "tex": tex_path}


# ---------------------------------------------------------------------------
# specific tables
# ---------------------------------------------------------------------------

def table_dataset_characteristics(manifest: Any) -> Any:
    """Table 1: what each domain contains.

    ``Patients`` is a *string* column on purpose. Three of the four datasets
    publish no patient identifiers at all, and that is a permanent property of
    the data -- rendering it as a missing number would let it be confused with
    "NOT RUN", which means an experiment we have simply not done yet.
    """
    import pandas as pd

    from ..data.schema import N_GRADES
    from ..visualization.style import DOMAIN_LABELS, DOMAIN_ORDER

    rows = []
    order = {name: i for i, name in enumerate(DOMAIN_ORDER)}
    for domain, group in sorted(
        manifest.groupby("domain", sort=False), key=lambda kv: order.get(kv[0], 99)
    ):
        counts = group["grade"].value_counts().reindex(range(N_GRADES), fill_value=0)
        has_patients = bool(group["patient_id"].notna().any())
        rows.append({
            "Domain": DOMAIN_LABELS.get(domain, domain),
            "Images": len(group),
            "Patients": f"{group['patient_id'].nunique():,}" if has_patients else NOT_AVAILABLE,
            **{f"Grade {g}": int(counts[g]) for g in range(N_GRADES)},
            "No-DR %": round(100.0 * counts[0] / len(group), 1),
            "Patient-level split": "Yes" if has_patients else "No",
        })
    return pd.DataFrame(rows)


def table_method_comparison(registry: Any, *, protocol: str = "lodo") -> Any:
    """Table 4/7: one row per method, source-validation and target metrics."""
    import pandas as pd

    subset = registry[registry["protocol"] == protocol].copy()
    if subset.empty:
        return pd.DataFrame([{"Method": NOT_RUN}])

    subset = subset.sort_values("timestamp_utc").drop_duplicates("experiment_id", keep="last")
    return pd.DataFrame({
        "Method": subset["method"],
        "Target": subset["target_domain"],
        "Val QWK": subset["best_val_qwk"],
        "Test QWK": subset["test_qwk"],
        "Test macro F1": subset["test_f1_macro"],
        "Test AUROC": subset["test_auroc_macro"],
        "Test MAE": subset["test_mae_grade"],
        "Test ECE": subset["test_ece"],
        "Test NLL": subset["test_nll"],
    }).reset_index(drop=True)


def table_calibration(evaluations: Sequence[dict[str, Any]]) -> Any:
    """Table 5: calibration before and after temperature scaling."""
    import pandas as pd

    rows = []
    for evaluation in evaluations:
        for split, result in evaluation["results"].items():
            rows.append({
                "Experiment": evaluation.get("experiment_id", NOT_RUN),
                "Split": split,
                "ECE": result.get("calibration", {}).get("ece"),
                "ECE (T-scaled)": result.get("calibration_scaled", {}).get("ece"),
                "Adaptive ECE": result.get("calibration", {}).get("adaptive_ece"),
                "NLL": result.get("calibration", {}).get("nll"),
                "NLL (T-scaled)": result.get("calibration_scaled", {}).get("nll"),
                "Brier": result.get("calibration", {}).get("brier"),
                "Conf - Acc": result.get("calibration", {}).get("confidence_minus_accuracy"),
            })
    return pd.DataFrame(rows)


def table_selective_prediction(coverage_rows: Sequence[dict[str, Any]], *, method: str) -> Any:
    """Table 6: performance at each coverage level."""
    import pandas as pd

    return pd.DataFrame([{
        "Method": method,
        "Coverage": f"{row['coverage']:.0%}",
        "Kept": row["n_retained"],
        "Accuracy": row["accuracy"],
        "QWK": row["qwk"],
        "Macro F1": row["f1_macro"],
        "Severe error": row["severe_error_rate"],
    } for row in coverage_rows])


def table_per_class(metrics: dict[str, Any], *, label: str = "") -> Any:
    """Table 8: per-class precision, recall, specificity and F1."""
    import pandas as pd

    from ..data.schema import GRADE_NAMES

    return pd.DataFrame([{
        "Experiment": label,
        "Grade": f"{row['grade']} {GRADE_NAMES[row['grade']]}",
        "Support": row["support"],
        "Precision": row["precision"],
        "Recall": row["recall"],
        "Specificity": row["specificity"],
        "F1": row["f1"],
    } for row in metrics["per_class"]])


def table_computational_cost(registry: Any) -> Any:
    """Table 9: parameters, training time and peak memory per method."""
    import pandas as pd

    subset = registry.sort_values("timestamp_utc").drop_duplicates("experiment_id", keep="last")
    return pd.DataFrame({
        "Method": subset["method"],
        "Backbone": subset["backbone"],
        "Params (M)": subset["params_total_m"],
        "Trainable (M)": subset["params_trainable_m"],
        "Epochs": subset["epochs_run"],
        "Train time (s)": subset["train_seconds"],
        "s / epoch": (subset["train_seconds"] / subset["epochs_run"].clip(lower=1)).round(1),
        "Peak VRAM (GB)": subset["peak_vram_gb"],
    }).reset_index(drop=True)
