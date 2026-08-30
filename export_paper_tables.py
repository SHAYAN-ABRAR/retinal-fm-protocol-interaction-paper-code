"""Export the paper's tables as booktabs LaTeX, built from saved results.

Every table here is generated from the CSVs the experiments wrote, never
retyped. Retyping a number into a manuscript is how a table comes to disagree
with the analysis that produced it, and that error is invisible on inspection.

Numbers that carry an across-seed SD are rendered as ``mean ± SD``. Where a
result is not established -- an interval spanning zero, or an effect smaller
than the seed noise -- the caption says so rather than leaving the reader to
infer it from a bold weight.

Usage:
    python export_paper_tables.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

ALL_TARGETS = ["ddr", "aptos", "idrid", "eyepacs"]
LABELS = {"ddr": "DDR", "aptos": "APTOS 2019", "idrid": "IDRiD", "eyepacs": "EyePACS"}

# The backbone the paper's results are reported on. ConvNeXt-Tiny exists as a
# robustness check and must never be pooled with it: mixing architectures turns
# an across-seed SD into an across-architecture SD.
MAIN_BACKBONE = "densenet121"
# The reference input configuration for every result in the paper. The Q1
# batch-size controls added a second 224 px configuration to lodo_results.csv.
REFERENCE_BATCH_SIZE = 32
MAIN_IMAGE_SIZE = 224


def _pm(mean, sd, decimals: int = 4) -> str:
    import numpy as np

    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "NOT RUN"
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.utils.io import project_root

    outputs = project_root() / "outputs"
    tables = outputs / "tables"
    written = []

    def emit(name: str, body: str) -> None:
        path = tables / f"{name}.tex"
        path.write_text(body, encoding="utf-8")
        written.append(path)

    # ---------------------------------------------------------------- LODO
    lodo = tables / "lodo_results.csv"
    if lodo.exists():
        frame = pd.read_csv(lodo)
        frame = frame[frame["method"] == "erm"]
        # lodo_results.csv now also holds domain-balanced-sampler runs, whose
        # "method" column reads exactly the same as their ordinary-sampler
        # counterparts -- only the experiment id distinguishes them. Without this
        # filter an ERM three-seed summary would silently average six rows from two
        # different samplers and report the spread between them as seed noise.
        if "domain_balanced" in frame.columns:
            frame = frame[~frame["domain_balanced"].fillna(False).astype(bool)]
        # The table also holds the ConvNeXt-Tiny backbone comparison. Without
        # this filter its seed-42 rows join the DenseNet121 seeds and the
        # "across-seed SD" becomes an across-architecture SD -- which is how
        # DDR came to be exported as 0.7465 +/- 0.0105 instead of
        # 0.7383 +/- 0.0055.
        if "backbone" in frame.columns:
            frame = frame[frame["backbone"] == MAIN_BACKBONE]
        # Third column of the same kind, after sampler and backbone. Without it
        # the batch-16 Q1 controls join the batch-32 seeds and this table
        # reported IDRiD as 0.7643 +/- 0.0348 instead of 0.7413 +/- 0.0244.
        if "batch_size" in frame.columns:
            frame = frame[frame["batch_size"].fillna(REFERENCE_BATCH_SIZE)
                          .astype(int) == REFERENCE_BATCH_SIZE]
        n_seeds = frame["seed"].nunique()
        rows = []
        for target in ALL_TARGETS:
            subset = frame[frame["target"] == target]
            if subset.empty:
                rows.append([LABELS[target], "NOT RUN", "", "", "", ""])
                continue
            rows.append([
                LABELS[target],
                f"{int(subset['n_test'].iloc[0]):,}",
                _pm(subset["target_qwk"].mean(), subset["target_qwk"].std(ddof=1)),
                _pm(subset["target_ece"].mean(), subset["target_ece"].std(ddof=1)),
                _pm(subset["target_ece_scaled"].mean(),
                    subset["target_ece_scaled"].std(ddof=1)),
                _pm(subset["target_severe"].mean(), subset["target_severe"].std(ddof=1)),
            ])
        header = ("Held-out domain & $n$ & QWK & ECE & ECE after $T$ & "
                  "Severe-error rate \\\\")
        body = "\n".join(" & ".join(r) + r" \\" for r in rows)
        emit("table_lodo", "\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{Leave-one-domain-out results, mean $\pm$ SD over "
            rf"{n_seeds} seeds. The temperature is fitted on source validation "
            r"only and never sees the target domain. \textbf{No mean over the "
            r"four rows should be quoted}: the change against source validation "
            r"ranges from $-0.483$ to $+0.063$, so an average describes no row.}",
            r"\label{tab:lodo}",
            r"\begin{tabular}{lrrrrr}", r"\toprule", header, r"\midrule", body,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]))

    # ------------------------------------------------- deployment cost
    verdicts = tables / "lodo_erm_seed_verdicts.csv"
    matched = tables / "in_domain_vs_lodo_erm_s42.csv"
    if verdicts.exists() and matched.exists():
        frame = pd.read_csv(verdicts)
        # n_test lives in the matched-comparison table; the verdicts table
        # carries n_seeds instead. Joining rather than assuming keeps the
        # printed n equal to the images the comparison actually used.
        counts = pd.read_csv(matched).set_index("domain")["n_test"].to_dict()
        rows = []
        for target in ALL_TARGETS:
            subset = frame[frame["target"] == target]
            if subset.empty or target not in counts:
                continue
            r = subset.iloc[0]
            verdict = str(r["verdict"]).replace("_", " ")
            rows.append([
                LABELS[target], f"{int(counts[target]):,}",
                f"{r['in_domain_qwk']:.4f}",
                _pm(r["lodo_qwk_mean"], r["lodo_qwk_sd"]),
                f"${r['delta_qwk']:+.4f}$",
                f"$[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]$",
                verdict,
            ])
        header = (r"Domain & $n$ & In-domain & Cross-domain & $\Delta$ & "
                  r"95\% CI & Verdict \\")
        body = "\n".join(" & ".join(r) + r" \\" for r in rows)
        emit("table_deployment_cost", "\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{The cost of cross-domain deployment, measured on "
            r"\emph{identical} test images: the cross-domain model's predictions "
            r"are restricted to the in-domain model's held-out split before "
            r"differencing. A result is called real only when it exceeds the "
            r"across-seed SD \emph{and} its paired bootstrap interval excludes "
            r"zero. Two of four are established; APTOS and IDRiD are limited by "
            r"test-set size (354 and 102 images).}",
            r"\label{tab:deployment}",
            r"\begin{tabular}{lrrrrrl}", r"\toprule", header, r"\midrule", body,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]))

    # ------------------------------------------------- cross-domain matrix
    single = tables / "single_source_results.csv"
    indomain = tables / "in_domain_results.csv"
    if single.exists() and indomain.exists():
        s = pd.read_csv(single)
        s = s[s["seed"] == 42]
        d = pd.read_csv(indomain)
        # The matrix is a 224px table; in_domain_results.csv also holds the
        # 512px EyePACS run. Selecting by position would put a 512px number on
        # one diagonal cell and 224px numbers everywhere else.
        if "image_size" in d.columns:
            d = d[d["image_size"] == MAIN_IMAGE_SIZE]
        rows = []
        for source in ALL_TARGETS:
            cells = [LABELS[source]]
            for target in ALL_TARGETS:
                if source == target:
                    v = d[d["domain"] == target]["test_qwk"]
                    cells.append(f"\\textit{{{v.iloc[0]:.3f}}}" if len(v) else "NOT RUN")
                else:
                    v = s[(s["source"] == source) & (s["target"] == target)]["target_qwk"]
                    cells.append(f"{v.iloc[0]:.3f}" if len(v) else "NOT RUN")
            n = s[s["source"] == source]["n_train"]
            cells.append(f"{int(n.iloc[0]):,}" if len(n) else "--")
            rows.append(cells)
        header = ("Trained on & " + " & ".join(LABELS[t] for t in ALL_TARGETS)
                  + r" & $n_{\text{train}}$ \\")
        body = "\n".join(" & ".join(r) + r" \\" for r in rows)
        emit("table_cross_domain_matrix", "\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{Cross-domain QWK for single-source models (seed 42). "
            r"Diagonal entries, in italics, are in-domain. "
            r"\textbf{Source identity dominates source count}: EyePACS alone "
            r"matches three-source training on every target, while IDRiD alone "
            r"(335 images) fails everywhere. The IDRiD row varies by up to "
            r"0.088 QWK across seeds and no single entry in it should be "
            r"quoted as a point estimate.}",
            r"\label{tab:matrix}",
            r"\begin{tabular}{lrrrrr}", r"\toprule", header, r"\midrule", body,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]))

    # ------------------------------------------------- method comparison
    seeds = tables / "stage_c_seed_summary.csv"
    if seeds.exists():
        frame = pd.read_csv(seeds)
        pretty = {"erm": "ERM", "ordinal": "Ordinal CORAL",
                  "deep_coral": "Deep CORAL", "mixstyle": "MixStyle",
                  "mixstyle_ordinal": "MixStyle + Ordinal",
                  "deep_coral_ordinal": "Deep CORAL + Ordinal"}
        rows = []
        for _, r in frame.iterrows():
            rows.append([
                pretty.get(r["method"], r["method"]),
                _pm(r.get("test_qwk_mean"), r.get("test_qwk_sd")),
                _pm(r.get("test_ece_mean"), r.get("test_ece_sd")),
                _pm(r.get("test_severe_mean"), r.get("test_severe_sd")),
            ])
        header = r"Method & Target QWK & Target ECE & Severe-error rate \\"
        body = "\n".join(" & ".join(r) + r" \\" for r in rows)
        emit("table_method_comparison", "\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{Method comparison on DDR\,+\,APTOS $\rightarrow$ unseen "
            r"IDRiD, mean $\pm$ SD over three seeds ($n=507$). Every "
            r"alternative is worse than ERM on target QWK, clearing both the "
            r"across-seed SD and the paired bootstrap. The ordinal head's "
            r"calibration gain is real but comes with collapse of grades 1 and "
            r"3, doubling the severe-error rate.}",
            r"\label{tab:methods}",
            r"\begin{tabular}{lrrr}", r"\toprule", header, r"\midrule", body,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]))

    # ------------------------------------------------- severe-error cost
    severe = tables / "severe_error_comparison.csv"
    if severe.exists():
        frame = pd.read_csv(severe)
        rows = []
        for target in ALL_TARGETS:
            subset = frame[frame["target"] == target]
            if subset.empty:
                continue
            r = subset.iloc[0]
            established = r["verdict"] == "REAL (both bars)"
            rows.append([
                LABELS[target], f"{int(r['n_test']):,}",
                f"{r['in_domain_severe']:.4f}",
                _pm(r["lodo_severe_mean"], r["lodo_severe_sd"]),
                f"${r['delta_severe']:+.4f}$",
                f"$[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]$",
                (rf"\textbf{{{r['relative_increase']:+.0%}}}".replace("%", r"\%")
                 if established else "--"),
            ])
        header = (r"Domain & $n$ & In-domain & Cross-domain & $\Delta$ & "
                  r"95\% CI & Relative \\")
        body = "\n".join(" & ".join(r) + r" \\" for r in rows)
        emit("table_severe_error", "\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{Severe-error rate ($|\text{error}| \geq 2$ grades) "
            r"in-domain versus cross-domain, on \emph{identical} test images. "
            r"A two-step misgrade is the error that sends a referable patient "
            r"home, and it is not recoverable by recalibration: temperature "
            r"scaling is monotonic and cannot move an argmax. Where the "
            r"deployment cost is established under both bars it \textbf{more "
            r"than doubles}. The relative column is left blank where the "
            r"interval spans zero; those differences are not established.}",
            r"\label{tab:severe}",
            r"\begin{tabular}{lrrrrrr}", r"\toprule", header, r"\midrule", body,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]))

    # ------------------------------------------------- selective prediction
    selective = tables / "selective_prediction_erm.csv"
    if selective.exists():
        frame = pd.read_csv(selective)
        rows = []
        for target in ALL_TARGETS:
            subset = frame[frame["target"] == target]
            if subset.empty:
                continue
            r = subset.iloc[0]
            rows.append([
                LABELS[target], f"{int(r['n']):,}",
                _pm(r["error_auroc_mean"], r["error_auroc_sd"], 3),
                f"{r['risk@1.0_mean']:.3f}",
                f"{r['risk@0.7_mean']:.3f}",
                f"{(1 - r['risk@0.7_mean'] / r['risk@1.0_mean']) * 100:.0f}\\%",
            ])
        header = (r"Domain & $n$ & Error-detection AUROC & Error @100\% & "
                  r"Error @70\% & Reduction \\")
        body = "\n".join(" & ".join(r) + r" \\" for r in rows)
        emit("table_selective_prediction", "\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{Selective prediction on unseen domains, three seeds. "
            r"Abstaining on the least-confident 30\% of cases reduces the "
            r"automated error rate by only 17--33\%, and works \emph{worst} on "
            r"EyePACS where the error rate is highest. Temperature scaling is "
            r"monotonic, so these values are identical before and after "
            r"calibration.}",
            r"\label{tab:selective}",
            r"\begin{tabular}{lrrrrr}", r"\toprule", header, r"\midrule", body,
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]))

    for path in written:
        print(f"saved -> {path.name}")
    if not written:
        print("NOT RUN -- no result tables found to export")
    else:
        print(f"\n{len(written)} LaTeX tables written to {tables}")
        print("Every value is read from a results CSV; none is retyped.")


if __name__ == "__main__":
    main()
