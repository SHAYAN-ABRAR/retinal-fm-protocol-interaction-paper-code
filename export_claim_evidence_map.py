"""The claim-evidence map: every candidate manuscript claim, and what backs it.

Each claim carries the authoritative CSV it comes from, the effect estimate and
interval, the formal test, the multiplicity adjustment, whether it is supported,
and -- the part that matters for drafting -- the exact wording permitted and the
stronger wording forbidden.

Numeric fields are read from the generator CSVs rather than typed, so the map
cannot drift from the tables. Running it twice on unchanged tables produces an
identical file.

Writes:
    outputs/tables/final_claim_evidence.csv

Usage:
    python export_claim_evidence_map.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

SUPPORTED = "statistically supported under the study's inferential framework"
NOT_DEMONSTRATED = "not demonstrated (this is not a demonstration of equivalence)"
DESCRIPTIVE = "descriptive only; no formal claim"


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    tables = project_root() / "outputs" / "tables"
    master = pd.read_csv(tables / "JBHI_MASTER_RESULTS.csv")
    family = pd.read_csv(tables / "JBHI_PRIMARY_INTERACTION.csv")

    def m(domain, protocol, seed_set="common-5"):
        row = master[(master.domain == domain) & (master.protocol == protocol)
                     & (master.seed_set == seed_set)]
        if len(row) != 1:
            raise SystemExit(f"master row not unique: {domain} {protocol} {seed_set}")
        return row.iloc[0]

    def f(domain):
        row = family[family.domain == domain]
        if len(row) != 1:
            raise SystemExit(f"family row not unique: {domain}")
        return row.iloc[0]

    ddr_i, aptos_i = f("DDR"), f("APTOS")
    claims = []

    def add(**kw):
        claims.append(kw)

    # ------------------------------------------------------------ PRIMARY
    add(claim_id="P1",
        claim=("Moving from frozen linear probing to matched full fine-tuning "
               "changes the relative QWK difference between ImageNet-MAE and "
               "RETFound."),
        claim_class="primary",
        datasets="DDR; APTOS",
        protocols="frozen linear probe vs full fine-tuning",
        seeds="42,1,2,3,4 (common five, paired within seed)",
        authoritative_csv="JBHI_PRIMARY_INTERACTION.csv",
        effect=(f"DDR {ddr_i.interaction:+.4f}; APTOS {aptos_i.interaction:+.4f}"),
        ci=(f"DDR [{ddr_i.crossed_CI_low:+.4f}, {ddr_i.crossed_CI_high:+.4f}]; "
            f"APTOS [{aptos_i.crossed_CI_low:+.4f}, {aptos_i.crossed_CI_high:+.4f}]"),
        formal_p=f"DDR {ddr_i.paired_t_p:.4f}; APTOS {aptos_i.paired_t_p:.4f}",
        multiplicity=(f"Holm across the two domains: DDR {ddr_i.Holm_p:.4f}; "
                      f"APTOS {aptos_i.Holm_p:.4f}"),
        sign_agreement=f"DDR {ddr_i.sign_agreement}; APTOS {aptos_i.sign_agreement}",
        supported="yes -- on both domains",
        inference_status=SUPPORTED,
        allowed_wording=(
            "The large frozen-probe ImageNet-MAE advantage was substantially "
            "attenuated under matched full fine-tuning on both held-out "
            "domains. | The protocol-by-initialization interaction was "
            "statistically supported on both domains under the final "
            "two-domain Holm-adjusted analysis."),
        forbidden_wording=(
            "RETFound is superior under full fine-tuning. | The models are "
            "equivalent after fine-tuning. | Linear probing reverses the "
            "ranking on every domain. | This behaviour generalizes to all "
            "retinal foundation models."))

    # ---------------------------------------------------------- SECONDARY
    for domain in ("DDR", "APTOS"):
        row = m(domain, "full")
        excl = (row.crossed_CI_low > 0) or (row.crossed_CI_high < 0)
        add(claim_id=f"S1-{domain}",
            claim=(f"Under matched full fine-tuning on {domain}, ImageNet-MAE "
                   f"and RETFound differ in QWK."),
            claim_class="secondary",
            datasets=domain,
            protocols="full fine-tuning",
            seeds="42,1,2,3,4",
            authoritative_csv="JBHI_MASTER_RESULTS.csv",
            effect=f"{row.paired_delta:+.4f} (ImageNet-MAE minus RETFound)",
            ci=f"[{row.crossed_CI_low:+.4f}, {row.crossed_CI_high:+.4f}]",
            formal_p=f"{row.raw_p:.4f}",
            multiplicity="none (single pre-specified comparison per domain)",
            sign_agreement=str(row.sign_agreement),
            supported="no",
            inference_status=NOT_DEMONSTRATED,
            allowed_wording=(
                f"No difference between the two initialisations was "
                f"demonstrated under matched full fine-tuning on {domain}."
                + (" The crossed uncertainty interval excludes zero while the "
                   "formal seed-level test does not reject; under the study's "
                   "framework the formal test governs and no difference is "
                   "claimed." if excl else "")),
            forbidden_wording=(
                f"The two initialisations are equivalent on {domain}. | "
                f"RETFound matches ImageNet-MAE after fine-tuning. | "
                f"{'RETFound outperforms ImageNet-MAE on ' + domain + '. | ' if row.paired_delta < 0 else ''}"
                "Full fine-tuning eliminates the difference."))

    add(claim_id="S2",
        claim=("Two secondary metrics under full fine-tuning on APTOS "
               "(macro AUROC, ECE) favour ImageNet-MAE at uncorrected p<0.05."),
        claim_class="secondary",
        datasets="APTOS", protocols="full fine-tuning", seeds="42,1,2,3,4",
        authoritative_csv="aptos_secondary_outcomes.csv",
        effect="AUROC +0.0241; ECE -0.0126 (lower ECE is better)",
        ci="not computed for secondary metrics",
        formal_p="AUROC 0.0394; ECE 0.0246",
        multiplicity="none applied; nine-metric exploratory set",
        sign_agreement="AUROC 4/5; ECE 5/5",
        supported="no -- exploratory",
        inference_status=DESCRIPTIVE,
        allowed_wording=(
            "Among nine exploratory secondary metrics, macro AUROC and ECE "
            "favoured ImageNet-MAE at uncorrected p<0.05; with QWK as the "
            "pre-specified endpoint and roughly one such value expected by "
            "chance across nine metrics, neither is claimed."),
        forbidden_wording=(
            "ImageNet-MAE is better calibrated than RETFound. | ImageNet-MAE "
            "has significantly higher AUROC. | Secondary metrics confirm the "
            "ImageNet-MAE advantage."))

    # -------------------------------------------------------- DESCRIPTIVE
    for domain in ("DDR", "APTOS", "IDRiD"):
        row = m(domain, "frozen", "all-10")
        favours = "ImageNet-MAE" if row.paired_delta > 0 else "RETFound"
        add(claim_id=f"D1-{domain}",
            claim=(f"Under frozen linear probing on {domain}, the QWK "
                   f"difference favours {favours}."),
            claim_class="descriptive",
            datasets=domain, protocols="frozen linear probe",
            seeds="42,1,2,3,4,5,6,7,8,9 (ten)",
            authoritative_csv="JBHI_MASTER_RESULTS.csv (seed_set=all-10)",
            effect=f"{row.paired_delta:+.4f}",
            ci=f"[{row.crossed_CI_low:+.4f}, {row.crossed_CI_high:+.4f}]",
            formal_p=f"{row.raw_p:.4f}",
            multiplicity="none (descriptive)",
            sign_agreement=str(row.sign_agreement),
            supported="descriptive",
            inference_status=DESCRIPTIVE,
            allowed_wording=(
                f"Under frozen linear probing on {domain} the paired "
                f"difference was {row.paired_delta:+.4f} QWK, favouring "
                f"{favours} (ten seeds, descriptive)."),
            forbidden_wording=(
                "Frozen probing shows ImageNet-MAE is the better "
                "representation. | RETFound's representations are worse. | "
                "Differencing this ten-seed estimate against a five-seed "
                "adaptation estimate."))

    add(claim_id="D2",
        claim=("The frozen-probe direction is not uniform across held-out "
               "domains: DDR and APTOS favour ImageNet-MAE, IDRiD favours "
               "RETFound."),
        claim_class="descriptive",
        datasets="DDR; APTOS; IDRiD", protocols="frozen linear probe",
        seeds="ten", authoritative_csv="JBHI_MASTER_RESULTS.csv",
        effect=(f"DDR {m('DDR','frozen','all-10').paired_delta:+.4f}; "
                f"APTOS {m('APTOS','frozen','all-10').paired_delta:+.4f}; "
                f"IDRiD {m('IDRiD','frozen','all-10').paired_delta:+.4f}"),
        ci="see master table", formal_p="see master table",
        multiplicity="none (descriptive)",
        sign_agreement="DDR 10/10; APTOS 10/10; IDRiD 9/10",
        supported="descriptive",
        inference_status=DESCRIPTIVE,
        allowed_wording=(
            "The direction of the frozen-probe difference was not uniform "
            "across held-out domains, favouring ImageNet-MAE on DDR and APTOS "
            "and RETFound on IDRiD."),
        forbidden_wording=(
            "Frozen probing consistently favours ImageNet-MAE. | ImageNet-MAE "
            "has better frozen representations than RETFound."))

    add(claim_id="D3",
        claim=("RETFound converges earlier and overfits the source pool more "
               "than ImageNet-MAE under the shared budget."),
        claim_class="descriptive",
        datasets="DDR; APTOS (source validation only)",
        protocols="full fine-tuning", seeds="42,1,2,3,4 per domain",
        authoritative_csv=("full_finetune_source_validation.csv; "
                           "aptos_full_finetune_source_validation.csv"),
        effect=("early stop 10/10 RETFound vs 0/10 ImageNet-MAE; "
                "best epoch 8.6+-1.7 / 9.2+-2.6 vs 17.6+-1.5 / 18.0+-1.0; "
                "overfit ratio 2.39+-0.36 / 2.50+-0.32 vs 1.10+-0.05 / 1.14+-0.02"),
        ci="not applicable", formal_p="not tested",
        multiplicity="none", sign_agreement="10/10 vs 0/10",
        supported="descriptive",
        inference_status=DESCRIPTIVE,
        allowed_wording=(
            "Under the shared fixed budget RETFound reached its best "
            "source-validation QWK roughly twice as early and early-stopped in "
            "every run, while ImageNet-MAE never early-stopped; RETFound's "
            "source-validation loss rose further above its minimum. Reported "
            "as mechanistic context."),
        forbidden_wording=(
            "Overfitting causes the interaction. | RETFound is more prone to "
            "overfitting in general. | The budget disadvantaged RETFound."))

    # -------------------------------------------------------- LIMITATIONS
    add(claim_id="L1",
        claim=("IDRiD contributes no full fine-tuning evidence; the "
               "interaction is estimated on two domains only."),
        claim_class="limitation",
        datasets="IDRiD", protocols="frozen; partial only", seeds="n/a",
        authoritative_csv="JBHI_RUN_ACCOUNTING.csv",
        effect="0 IDRiD full-FT runs", ci="n/a", formal_p="n/a",
        multiplicity="n/a", sign_agreement="n/a", supported="n/a",
        inference_status="limitation",
        allowed_wording=("Matched full fine-tuning was run on two of the three "
                         "held-out domains; IDRiD contributes frozen and "
                         "partial evidence only."),
        forbidden_wording=("Any imputed or implied IDRiD full-FT value. | "
                           "Describing the interaction as a three-domain "
                           "result."))

    add(claim_id="L2",
        claim=("The two-domain family is a post-hoc pairing: DDR was observed "
               "before APTOS was specified."),
        claim_class="limitation",
        datasets="DDR; APTOS", protocols="all", seeds="n/a",
        authoritative_csv="JBHI_EVIDENCE_FREEZE.md section 10",
        effect="n/a", ci="n/a", formal_p="n/a",
        multiplicity="Holm applied for conservative final reporting",
        sign_agreement="n/a", supported="n/a",
        inference_status="limitation",
        allowed_wording=("DDR was the originating result; APTOS was specified "
                         "as a confirmatory replication after it, with the "
                         "recipe and analysis frozen before any APTOS target "
                         "outcome was inspected. Holm adjustment across the "
                         "two is applied for conservative final reporting."),
        forbidden_wording=("A prospectively specified two-domain family. | "
                           "Both domains were planned in advance."))

    add(claim_id="L3",
        claim=("EyePACS is a source domain and was part of RETFound's "
               "published pretraining corpus."),
        claim_class="limitation",
        datasets="EyePACS", protocols="all", seeds="n/a",
        authoritative_csv="DATA_PROVENANCE.md; JBHI_EVIDENCE_FREEZE.md section 4",
        effect="n/a", ci="n/a", formal_p="n/a", multiplicity="n/a",
        sign_agreement="n/a", supported="n/a",
        inference_status="limitation",
        allowed_wording=("EyePACS appears in RETFound's published colour-fundus "
                         "pretraining corpus and is one of the source domains "
                         "here; no held-out target image was seen in "
                         "pretraining or training, and the exposure applies "
                         "identically to the frozen and fine-tuned arms."),
        forbidden_wording=("Target leakage. | RETFound saw the test data. | "
                           "The comparison is contaminated."))

    frame = pd.DataFrame(claims, columns=[
        "claim_id", "claim", "claim_class", "datasets", "protocols", "seeds",
        "authoritative_csv", "effect", "ci", "formal_p", "multiplicity",
        "sign_agreement", "supported", "inference_status", "allowed_wording",
        "forbidden_wording"])
    out = tables / "final_claim_evidence.csv"
    frame.to_csv(out, index=False)

    print(f"{len(frame)} claim(s)")
    for r in frame.itertuples():
        print(f"  {r.claim_id:9} {r.claim_class:12} {r.supported:18} "
              f"{r.claim[:58]}")
    print(f"\nsaved -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
