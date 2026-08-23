"""Figures for the two clinically load-bearing results.

Discrimination and calibration figures live elsewhere. These two answer the
questions a clinical reader asks first:

1. **What does a two-step misgrade cost when the model is deployed off-domain?**
   The severe-error rate (|error| >= 2) is the error that sends a referable
   patient home, and it is not recoverable by recalibration -- temperature
   scaling is monotonic and cannot move an argmax.
2. **Can the model be made safe by declining the cases it is unsure about?**
   The risk-coverage curve answers that directly: abstain on the least-confident
   fraction and see how far the error rate on the rest falls.

Both read saved CSVs. A domain with no result renders as a NOT RUN panel rather
than being dropped, so a missing experiment stays visible in the figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from .style import DOMAIN_ORDER, apply_style, domain_color, domain_label, save_figure

log = get_logger("viz.clinical")

__all__ = [
    "figure_severe_error_deployment",
    "figure_risk_coverage",
    "generate_clinical_figures",
]

# Abstaining below this coverage stops being a screening tool and becomes a
# referral queue, so the curves are annotated here rather than at their tails.
REFERENCE_COVERAGE = 0.7


def _not_run(axis, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=10,
              color="#B00020", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def figure_severe_error_deployment(comparison: Any) -> Any:
    """Severe-error rate in-domain vs cross-domain, on matched images.

    ``comparison`` is the frame written by ``analyse_severe_error.py``. Rows
    whose verdict is not established are labelled as such, so an unsupported
    difference cannot be read off the figure as a result.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()
    figure, (bars, effect) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    present = [d for d in DOMAIN_ORDER if d in set(comparison["target"])]
    if not present:
        _not_run(bars, "NOT RUN")
        _not_run(effect, "NOT RUN")
        return figure

    positions = np.arange(len(present))
    width = 0.36

    for offset, (column, label, alpha) in enumerate([
        ("in_domain_severe", "Trained on this domain", 1.0),
        ("lodo_severe_mean", "Never saw this domain", 0.55),
    ]):
        heights, errors = [], []
        for domain in present:
            row = comparison[comparison["target"] == domain].iloc[0]
            heights.append(float(row[column]))
            sd = row.get("lodo_severe_sd", float("nan"))
            errors.append(
                float(sd) if column.endswith("mean") and np.isfinite(sd) else 0.0
            )
        bars.bar(positions + (offset - 0.5) * width, heights, width,
                 yerr=errors, capsize=3,
                 color=[domain_color(d) for d in present], alpha=alpha,
                 edgecolor="white", linewidth=0.8,
                 error_kw={"elinewidth": 1.0, "ecolor": "#333333"})

    for index, domain in enumerate(present):
        row = comparison[comparison["target"] == domain].iloc[0]
        established = row["verdict"] == "REAL (both bars)"
        top = max(float(row["in_domain_severe"]), float(row["lodo_severe_mean"]))
        text = f"{float(row['relative_increase']):+.0%}" if established else "not established"
        bars.text(index, top + 0.018, text, ha="center", fontsize=9,
                  fontweight="bold" if established else "normal",
                  color="#333333" if established else "#B00020")

    bars.set_xticks(positions)
    bars.set_xticklabels([domain_label(d) for d in present])
    bars.set_ylabel("Severe-error rate  (|error| $\\geq$ 2 grades)")
    # Deliberately not "cross-domain doubles severe errors": that holds on the
    # two domains where both bars pass, and IDRiD moves the other way. A title
    # a reader could quote must be true of the whole figure.
    bars.set_title("Where the deployment cost is established, severe errors more than double")
    bars.set_ylim(0, max(comparison["lodo_severe_mean"].max(),
                         comparison["in_domain_severe"].max()) * 1.30)
    # Colour encodes the domain and fill opacity encodes the condition, so the
    # legend keys must be neutral. Taking them from the bars would tint both
    # entries with whichever domain happens to be plotted first.
    from matplotlib.patches import Patch

    bars.legend(handles=[
        Patch(facecolor="#555555", edgecolor="white", label="Trained on this domain"),
        Patch(facecolor="#555555", alpha=0.55, edgecolor="white",
              label="Never saw this domain"),
    ], loc="upper left")

    # Right panel: the difference with its interval, which is where the claim
    # actually lives. A bar chart alone invites reading every gap as real.
    for index, domain in enumerate(present):
        row = comparison[comparison["target"] == domain].iloc[0]
        established = row["verdict"] == "REAL (both bars)"
        low, high = float(row["ci_lower"]), float(row["ci_upper"])
        colour = domain_color(domain) if established else "#999999"
        effect.plot([low, high], [index, index], color=colour,
                    linewidth=3 if established else 2,
                    solid_capstyle="round", zorder=2)
        effect.scatter([float(row["delta_severe"])], [index], color=colour,
                       s=70 if established else 45, zorder=3,
                       edgecolor="white", linewidth=1.2)
        if not established:
            effect.text(high + 0.012, index, "spans zero", va="center",
                        fontsize=8, color="#B00020")

    effect.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
    effect.set_yticks(range(len(present)))
    effect.set_yticklabels([domain_label(d) for d in present])
    effect.invert_yaxis()
    effect.set_xlabel("Change in severe-error rate (cross-domain $-$ in-domain)")
    effect.set_title("Paired bootstrap, 95% CI")
    effect.grid(axis="y", alpha=0)
    # The "spans zero" labels sit to the right of each interval, so the axis
    # needs room for them or they are clipped at the frame.
    low = float(comparison["ci_lower"].min())
    high = float(comparison["ci_upper"].max())
    span = high - low
    effect.set_xlim(low - 0.08 * span, high + 0.34 * span)

    figure.tight_layout()
    return figure


def figure_risk_coverage(curves: Any, summary: Any = None) -> Any:
    """Risk-coverage curves per domain, averaged over seeds.

    Coverage is the fraction of cases the model still answers; risk is the
    error rate among those. A useful abstention mechanism bends the curve down
    sharply as coverage falls. These barely bend, which is the finding.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()
    figure, axis = plt.subplots(figsize=(7.2, 5.0))

    present = [d for d in DOMAIN_ORDER if d in set(curves["target"])]
    if not present:
        _not_run(axis, "NOT RUN -- no risk-coverage curves found")
        return figure

    # Seeds have different test-set orderings, so the curves are resampled onto
    # a shared coverage grid before averaging. Averaging raw curves point-by-
    # point would silently compare different coverage levels.
    grid = np.linspace(0.05, 1.0, 200)
    annotations: list[tuple[str, float, float]] = []

    for domain in present:
        subset = curves[curves["target"] == domain]
        resampled = []
        for seed in sorted(subset["seed"].unique()):
            one = subset[subset["seed"] == seed].sort_values("coverage")
            resampled.append(
                np.interp(grid, one["coverage"].to_numpy(), one["risk"].to_numpy())
            )
        stacked = np.vstack(resampled)
        mean = stacked.mean(axis=0)
        axis.plot(grid, mean, color=domain_color(domain), linewidth=2.0,
                  label=domain_label(domain), zorder=3)
        if len(stacked) > 1:
            sd = stacked.std(axis=0, ddof=1)
            axis.fill_between(grid, mean - sd, mean + sd,
                              color=domain_color(domain), alpha=0.18, zorder=2)

        if summary is not None and domain in set(summary["target"]):
            row = summary[summary["target"] == domain].iloc[0]
            at_reference = float(row[f"risk@{REFERENCE_COVERAGE}_mean"])
            axis.scatter([REFERENCE_COVERAGE], [at_reference],
                         color=domain_color(domain), s=42, zorder=4,
                         edgecolor="white", linewidth=1.2)
            annotations.append((domain, float(row["risk@1.0_mean"]), at_reference))

    axis.axvline(REFERENCE_COVERAGE, color="#666666", linewidth=1.0,
                 linestyle="--", zorder=1)

    axis.set_xlabel("Coverage - fraction of cases the model still grades")
    axis.set_ylabel("Risk - error rate among the cases it grades")
    axis.set_title("Abstention does not rescue an off-domain model")
    axis.set_xlim(0.05, 1.02)
    axis.set_ylim(bottom=0, top=axis.get_ylim()[1] * 1.16)
    axis.legend(title="Unseen domain", loc="lower right")
    axis.text(REFERENCE_COVERAGE - 0.012, axis.get_ylim()[1] * 0.99,
              f"abstain on {1 - REFERENCE_COVERAGE:.0%}", rotation=90,
              va="top", ha="right", fontsize=8, color="#666666")

    # DDR and APTOS almost coincide at 70% coverage, as do IDRiD and EyePACS,
    # so per-curve labels overlap into illegibility. The numbers go in a block
    # instead, which also lets the "no abstention" baseline sit beside them --
    # the reduction means nothing without the rate it reduces from.
    if annotations:
        axis.text(0.055, 0.975,
                  f"error rate at 100% -> {REFERENCE_COVERAGE:.0%} coverage",
                  transform=axis.transAxes, fontsize=8.5, va="top",
                  color="#444444", fontweight="bold")
        for index, (domain, full, reduced) in enumerate(annotations):
            axis.text(0.055, 0.930 - index * 0.048,
                      f"{domain_label(domain):<11s} {full:.3f} -> {reduced:.3f}"
                      f"   ({1 - reduced / full:>4.0%})",
                      transform=axis.transAxes, fontsize=8.5, va="top",
                      family="monospace", color=domain_color(domain))

    figure.tight_layout()
    return figure


def generate_clinical_figures(outputs_dir: Path | str) -> dict[str, list[Path]]:
    """Write both figures from whatever results exist. Missing ones are skipped."""
    import pandas as pd

    outputs = Path(outputs_dir)
    tables = outputs / "tables"
    figures = outputs / "figures"
    written: dict[str, list[Path]] = {}

    comparison = tables / "severe_error_comparison.csv"
    if comparison.exists():
        written["severe_error_deployment"] = save_figure(
            figure_severe_error_deployment(pd.read_csv(comparison)),
            "fig_severe_error_deployment", figures, formats=("png", "pdf"),
        )
    else:
        log.warning("severe_error_comparison.csv NOT RUN -- run analyse_severe_error.py")

    curves = tables / "risk_coverage_curves_erm.csv"
    summary = tables / "selective_prediction_erm.csv"
    if curves.exists():
        written["risk_coverage"] = save_figure(
            figure_risk_coverage(
                pd.read_csv(curves),
                pd.read_csv(summary) if summary.exists() else None,
            ),
            "fig_risk_coverage", figures, formats=("png", "pdf"),
        )
    else:
        log.warning("risk_coverage_curves_erm.csv NOT RUN -- run analyse_selective.py")

    return written
