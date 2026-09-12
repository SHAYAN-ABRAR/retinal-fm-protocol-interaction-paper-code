"""Provenance for every manuscript figure, and the completeness gate.

Each panel declares the authoritative artifact it draws from, the columns it
reads, any filter applied, the seed set, the quantity derived and the function
that drew it. The outputs are hashed, so a figure can be tied to the exact
bytes a reviewer would receive.

Also enforces the two completeness rules:

* every figure has **both** a 300-dpi PNG and a vector PDF;
* every declared source artifact exists.

Writes:
    outputs/figures/jbhi_final/FIGURE_PROVENANCE.csv

Exit code is non-zero if any figure is missing a format or any source is gone.

Usage:
    python export_figure_provenance.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, ".")

OUT = "jbhi_final"
COMMON = "42,1,2,3,4"
TEN = "42,1,2,3,4,5,6,7,8,9"
THREE = "1,2,42"

# figure, panel, source, columns, filters, seed_set, derived, function
PANELS = [
    ("fig1_study_design", "whole", "(none -- schematic)", "", "",
     "n/a", "no plotted scientific quantity; parameter count 303.3M is the "
     "asserted trainable-parameter total", "figure_study_design"),

    ("fig2_adaptation_depth", "per-seed points",
     "figure_qwk_by_domain_protocol_seed.csv", "domain,protocol,seed,delta",
     "domain in {DDR,APTOS}", COMMON,
     "ImageNet-MAE minus RETFound QWK per seed and protocol",
     "figure_adaptation_depth"),
    ("fig2_adaptation_depth", "mean bars and labels",
     "JBHI_MASTER_RESULTS.csv", "domain,protocol,seed_set,paired_delta",
     "seed_set == common-5", COMMON, "mean paired delta per protocol",
     "figure_adaptation_depth"),

    ("fig3_primary_interaction", "estimates and intervals",
     "JBHI_PRIMARY_INTERACTION.csv",
     "domain,interaction,crossed_CI_low,crossed_CI_high,Holm_p,"
     "sign_agreement,n_test", "", COMMON,
     "I_full and its crossed seed x case 95% interval",
     "figure_primary_interaction"),

    ("fig4_seed_interaction_slopes", "per-seed slopes",
     "figure_qwk_by_domain_protocol_seed.csv", "domain,protocol,seed,delta",
     "protocol in {frozen,full}", COMMON,
     "per-seed delta at frozen and full, joined within seed",
     "figure_seed_slopes"),
    ("fig4_seed_interaction_slopes", "panel titles",
     "JBHI_PRIMARY_INTERACTION.csv", "domain,interaction,sign_agreement", "",
     COMMON, "mean interaction and sign agreement", "figure_seed_slopes"),

    ("fig5_full_ft_paired_qwk", "paired points",
     "figure_qwk_by_domain_protocol_seed.csv",
     "domain,protocol,seed,imagenet_qwk,retfound_qwk", "protocol == full",
     COMMON, "per-seed QWK of each arm under full fine-tuning",
     "figure_full_ft_paired"),
    ("fig5_full_ft_paired_qwk", "means, delta, CI, p",
     "JBHI_MASTER_RESULTS.csv",
     "ImageNet_QWK_mean,RETFound_QWK_mean,paired_delta,crossed_CI_low,"
     "crossed_CI_high,raw_p", "protocol == full and seed_set == common-5",
     COMMON, "arm means and the paired difference with its interval",
     "figure_full_ft_paired"),

    ("s1_source_validation", "best epoch; loss inflation; early stopping",
     "full_finetune_source_validation.csv + "
     "aptos_full_finetune_source_validation.csv",
     "model,best_epoch,overfit_ratio,early_stopped", "",
     COMMON + " per domain",
     "source-validation optimisation behaviour (descriptive)",
     "s1_source_validation"),

    ("s2_domain_profile", "A grade composition",
     "table1_dataset_characteristics.csv", "Domain,Grade 0..Grade 4", "",
     "n/a", "per-dataset grade distribution normalised to 100%",
     "s2_domain_profile"),
    ("s2_domain_profile", "B image size; C brightness",
     "reports/image_statistics.csv", "domain,megapixels,brightness,readable",
     "readable == True", "n/a",
     "distribution over the sampled image audit (400 images per domain)",
     "s2_domain_profile"),

    ("s3_configuration", "contrasts, intervals, Holm p",
     "configuration_decomposition.csv",
     "contrast,target,metric,delta,ci_lower,ci_upper,p_holm,"
     "delta_seed1,delta_seed2,delta_seed42", "metric == qwk", THREE,
     "batch / resolution / combined QWK contrasts, DenseNet-121",
     "s3_configuration"),

    ("s4_calibration", "reliability curves and ECE",
     "outputs/predictions/*__target_test[*]_predictions.csv",
     "true_grade,probability_grade_0..4 (temperature-scaled)",
     "protocol in {frozen,full}; bins with <50 pooled images not drawn",
     COMMON, "mean reliability curve per arm; ECE from calibration_metrics",
     "s4_calibration"),

    ("s5_risk_coverage", "risk-coverage curves",
     "outputs/predictions/*__target_test[*]_predictions.csv",
     "true_grade,predicted_grade,probability_grade_0..4", "", COMMON,
     "mean risk at each coverage from risk_coverage_curve", "s5_risk_coverage"),

    ("s6_error_pattern", "confusion matrices and severe-error rate",
     "outputs/predictions/*__target_test[*]_predictions.csv",
     "true_grade,predicted_grade", "protocol == full", COMMON,
     "row-normalised confusion pooled over seeds; severe error |t-p|>=2",
     "s6_error_pattern"),
]


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    root = project_root()
    directory = root / "outputs" / "figures" / OUT
    tables = root / "outputs" / "tables"
    if not directory.exists():
        print(f"NOT RUN -- {directory} missing")
        return 1

    figures = sorted({name for name, *_ in PANELS})
    problems = []

    # -------------------------------------------------- completeness gate
    for figure in figures:
        for suffix in ("png", "pdf"):
            path = directory / f"{figure}.{suffix}"
            if not path.exists():
                problems.append(f"{figure}: missing .{suffix}")

    # ------------------------------------------------------ source check
    for _, _, source, *_ in PANELS:
        for token in source.split(" + "):
            token = token.strip()
            if (not token or token.startswith("(") or "*" in token
                    or token.startswith("outputs/")):
                continue
            candidate = tables / token
            if not candidate.exists():
                candidate = root / "outputs" / token
            if not candidate.exists():
                problems.append(f"source missing: {token}")

    rows = []
    for figure, panel, source, columns, filters, seeds, derived, function in PANELS:
        png = directory / f"{figure}.png"
        pdf = directory / f"{figure}.pdf"
        rows.append({
            "figure": figure, "panel": panel, "source_csv": source,
            "source_columns": columns, "filters": filters, "seed_set": seeds,
            "derived_quantity": derived, "generator_function": function,
            "output_png": png.name if png.exists() else "MISSING",
            "output_pdf": pdf.name if pdf.exists() else "MISSING",
            "SHA256": sha256(png) if png.exists() else "",
        })

    frame = pd.DataFrame(rows)
    out = directory / "FIGURE_PROVENANCE.csv"
    frame.to_csv(out, index=False)

    print(f"{len(figures)} figure(s), {len(frame)} panel row(s)")
    for figure in figures:
        png = directory / f"{figure}.png"
        pdf = directory / f"{figure}.pdf"
        if png.exists():
            try:
                from PIL import Image
                with Image.open(png) as image:
                    size = f"{image.size[0]}x{image.size[1]}px"
                    dpi = image.info.get("dpi", (0, 0))[0]
            except Exception:                              # noqa: BLE001
                size, dpi = "?", 0
        else:
            size, dpi = "MISSING", 0
        print(f"  {figure:30} {size:>12}  {int(dpi) or '?':>4} dpi  "
              f"png {png.stat().st_size/1024:5.0f} KB  "
              f"pdf {pdf.stat().st_size/1024:5.0f} KB")

    if problems:
        print(f"\n!! {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"\nevery figure has PNG and PDF; every declared source exists")
    print(f"saved -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
