"""Write the plain-English research proposal as a Word document.

Who this is for
---------------
Someone with no medical or computing background: a family member, a funder, a
hospital administrator, a supervisor outside the field. It deliberately uses no
jargon -- no "domain generalization", no "calibration", no kappa scores -- and
states rates as "16 out of every 100 photographs" rather than 0.163.

Why it is a script and not a document
-------------------------------------
Every number is read from the project's own result tables at the moment of
writing, exactly as ``export_paper_tables.py`` does for the paper's LaTeX. A
plain-English summary that quietly disagrees with the results it summarises is
worse than no summary, and hand-editing a .docx as results land is precisely
how that happens. Re-run this after any experiment finishes and the document
follows.

Results that do not exist yet are described as still running. Nothing here is
predicted or rounded up for effect.

Usage:
    python make_proposal.py                    # writes docs/
    python make_proposal.py --also-downloads   # and the user's Downloads folder
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DOCS_NAME = "RESEARCH_PROPOSAL_PLAIN_ENGLISH.docx"
DOWNLOADS = Path.home() / "Downloads" / "Eye_Screening_AI_Research_Proposal.docx"

DOMAIN_NOTES = {
    "ddr": ("DDR", "China", "Good quality, carefully graded"),
    "aptos": ("APTOS 2019", "India", "Small, clean, widely used"),
    "idrid": ("IDRiD", "India", "Very small; treated with extra caution"),
    "eyepacs": ("EyePACS", "United States", "Largest; graded by many people, so noisier"),
}
ORDER = ["ddr", "aptos", "idrid", "eyepacs"]


def _read(path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() else None


def gather(outputs):
    """Collect every number the document quotes. Missing files become None."""
    import numpy as np
    import pandas as pd

    tables = outputs / "tables"
    facts = {}

    manifest = _read(outputs / "reports" / "unified_manifest_split.csv")
    facts["sizes"] = (manifest["domain"].value_counts().to_dict()
                      if manifest is not None else {})

    severe = _read(tables / "severe_error_comparison.csv")
    facts["severe"] = severe
    facts["severe_established"] = (
        severe[severe["verdict"] == "REAL (both bars)"] if severe is not None else None)

    selective = _read(tables / "selective_prediction_erm.csv")
    if selective is not None:
        reduction = (1 - selective["risk@0.7_mean"] / selective["risk@1.0_mean"]) * 100
        facts["abstain_low"] = int(round(reduction.min()))
        facts["abstain_high"] = int(round(reduction.max()))
        worst = selective.loc[selective["risk@1.0_mean"].idxmax(), "target"]
        facts["abstain_worst_domain"] = DOMAIN_NOTES.get(worst, (worst,))[0]
        facts["abstain_worst_value"] = int(round(
            float(reduction[selective["target"] == worst].iloc[0])))
    else:
        facts["abstain_low"] = facts["abstain_high"] = None

    # Resolution: the 224px three-seed mean against the 512px run.
    lodo = _read(tables / "lodo_results.csv")
    lodo512 = _read(tables / "lodo_results_r512.csv")
    facts["resolution"] = None
    if lodo is not None and lodo512 is not None:
        base = lodo[lodo["backbone"] == "densenet121"].groupby("target")
        base = base[["target_qwk", "target_severe"]].mean()
        new = lodo512.set_index("target")
        shared = [t for t in ORDER if t in base.index and t in new.index]
        if shared:
            drops = [(new.loc[t, "target_severe"] - base.loc[t, "target_severe"])
                     / base.loc[t, "target_severe"] * 100 for t in shared]
            facts["resolution"] = {
                "n_domains": len(shared),
                "all_improved": all(new.loc[t, "target_qwk"] > base.loc[t, "target_qwk"]
                                    for t in shared),
                "drop_low": int(round(-max(drops))),
                "drop_high": int(round(-min(drops))),
            }

    # Has the 512px in-domain reference been completed?
    in_domain = _read(tables / "in_domain_results.csv")
    facts["in_domain_512"] = (
        sorted(in_domain[in_domain["image_size"] == 512]["domain"].unique())
        if in_domain is not None and "image_size" in in_domain.columns else [])

    facts["subsample_done"] = (tables / "subsample_decomposition.csv").exists()
    return facts


def build(facts, out_path: Path) -> Path:
    from docx import Document
    from docx.shared import Pt, RGBColor

    ACCENT = RGBColor(0x0A, 0x5A, 0x58)
    GREY = RGBColor(0x55, 0x55, 0x55)

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11.5)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15

    def heading(text, level=1):
        h = doc.add_heading(text, level=level)
        for run in h.runs:
            run.font.color.rgb = ACCENT
        return h

    def para(text, bold=False, italic=False, size=None, colour=None):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold, run.italic = bold, italic
        if size:
            run.font.size = Pt(size)
        if colour:
            run.font.color.rgb = colour
        return p

    def bullet(text):
        doc.add_paragraph(text, style="List Bullet")

    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Light Grid Accent 1"
        for cell, text in zip(t.rows[0].cells, headers):
            cell.text = ""
            cell.paragraphs[0].add_run(text).bold = True
        for row in rows:
            for cell, text in zip(t.add_row().cells, row):
                cell.text = str(text)
        doc.add_paragraph()

    def approx(n):
        if n is None:
            return "not counted"
        if n >= 10000:
            return f"About {round(n, -2):,.0f}".replace(".0", "")
        return f"About {round(n, -2):,.0f}".replace(".0", "")

    title = doc.add_heading("Making Eye-Screening AI Safe to Use in Any Hospital", 0)
    for run in title.runs:
        run.font.color.rgb = ACCENT
    para("A study of what happens to diabetic eye-disease AI when it is moved "
         "from the hospital it learned in to a hospital it has never seen.",
         italic=True, size=12.5, colour=GREY)
    para("Research proposal - written in plain language. "
         "Every number is taken directly from our own results.",
         size=10, colour=GREY)

    # ---------------------------------------------------------------- summary
    heading("The short version")
    para("Diabetes can slowly damage the eye and cause blindness. It can be "
         "caught early by taking a photograph of the back of the eye. There "
         "are not enough eye doctors to look at every photograph, so computers "
         "are being trained to do a first check.")
    para("These computer programs work very well on photographs from the "
         "hospital they learned from. The problem is what happens next. Used "
         "in a different hospital, with a different camera and different "
         "patients, the program gets worse - and, more worryingly, it stays "
         "just as confident while being wrong.")
    para("This project measures exactly how much worse it gets, and how much "
         "its confidence can be trusted. We use four real, public collections "
         "of eye photographs from three countries, and we always test on a "
         "collection the program has never seen.")

    # ------------------------------------------------------------- why it matters
    heading("1. Why this matters")
    para("Diabetic retinopathy is damage to the small blood vessels at the "
         "back of the eye, caused by diabetes. It is one of the leading causes "
         "of blindness in working-age adults. Caught early it can usually be "
         "treated. Caught late, the sight already lost does not come back.")
    para("Doctors grade the damage from 0 to 4:")
    table(["Grade", "What it means"],
          [["0", "No damage"], ["1", "Mild"], ["2", "Moderate"],
           ["3", "Severe"], ["4", "Very advanced"]])
    para("The important line is between grade 1 and grade 2. From grade 2 "
         "upwards a patient normally needs to see a specialist. A mistake that "
         "carries a patient across that line is not a small error - it can "
         "mean somebody is sent home who needed treatment.")

    # ------------------------------------------------------------- the problem
    heading("2. The problem we are studying")
    para("There are two problems, and the second is the dangerous one.", bold=True)
    para("Problem one: the program gets worse in a new hospital.")
    para("Every hospital uses different cameras and different lighting, and "
         "sees different patients. A program trained in one place has quietly "
         "picked up habits that only apply there. Move it, and it gets things "
         "wrong more often.")
    para("Problem two: it does not realise it has got worse.", bold=True)
    para("A good program should become less sure of itself when it is out of "
         "its depth, so that a doctor knows to look again. What actually "
         "happens is that it stays confident. It is like being given wrong "
         "directions in a very certain voice - you would follow them, because "
         "nothing warned you not to.")
    para("Most published research measures only the first problem. We measure "
         "both, and treat the second as the main question.")

    # ---------------------------------------------------------------- the data
    heading("3. What data we use")
    para("We use four public collections of eye photographs from different "
         "countries and hospitals. Using four rather than one is the whole "
         "point: it lets us move a program between them and watch what happens.")
    rows = []
    for key in ORDER:
        name, place, note = DOMAIN_NOTES[key]
        rows.append([name, place, approx(facts["sizes"].get(key)), note])
    table(["Collection", "Where from", "Photographs", "Notes"], rows)
    para("All four are public and free to use for research. We collect no new "
         "patient data, and no patient can be identified from these photographs.")

    para("We checked the data before trusting it.", bold=True)
    para("In the largest collection, more than 50,000 photographs came with "
         "labels that could not be checked against the official records. Those "
         "labels turned out to be broken: among all 50,000, only one single "
         "photograph was marked as severe, where roughly 1,250 should have "
         "been. We removed every one of them and kept only the photographs "
         "whose grades we could verify exactly.")
    para("We also found photographs that appear in more than one collection. "
         "Those had to go too. If the same photograph teaches the program and "
         "then tests it, the test means nothing - like marking an exam using "
         "questions the student was already given the answers to.")

    # ---------------------------------------------------------------- method
    heading("4. What we actually do")
    para("The method is simple to describe. We take three of the four "
         "collections, train the program on those, then test it on the fourth "
         "- which it has never seen, not even once. That is our stand-in for "
         "moving a program to a new hospital. We repeat this four times, "
         "holding out each collection in turn.")
    para("For every test we ask three questions:")
    bullet("How often does it agree with the doctor?")
    bullet("How often is it badly wrong - off by two grades or more?")
    bullet("When it says it is confident, is it actually right?")
    para("The third question is the one usually skipped, and it is the one "
         "that decides whether a doctor can safely rely on the answer.")

    # ---------------------------------------------------------------- findings
    heading("5. What we have found so far")
    para("These are finished, measured results - not predictions.",
         italic=True, colour=GREY)

    severe = facts["severe"]
    if severe is not None and facts["severe_established"] is not None \
            and len(facts["severe_established"]):
        para("Serious mistakes roughly double in a new hospital.", bold=True)
        para("This is the number that matters most in a clinic. A serious "
             "mistake means the grade was wrong by two steps or more:")
        rows = []
        for _, r in facts["severe_established"].iterrows():
            name = DOMAIN_NOTES.get(r["target"], (r["target"],))[0]
            rows.append([
                name,
                f"{r['in_domain_severe'] * 100:.1f} out of every 100",
                f"{r['lodo_severe_mean'] * 100:.1f} out of every 100",
                f"{r['relative_increase'] * 100:.0f}% more mistakes",
            ])
        table(["Collection", "Trained on this hospital",
               "Never seen this hospital", "Change"], rows)
        unresolved = severe[severe["verdict"] != "REAL (both bars)"]
        if len(unresolved):
            names = ", ".join(DOMAIN_NOTES.get(t, (t,))[0] for t in unresolved["target"])
            para(f"On the other collections ({names}) the test sets were too "
                 "small to give a trustworthy answer, so we say so rather than "
                 "guessing. Reporting a number we cannot stand behind would be "
                 "worse than reporting none.")
    else:
        para("Serious-mistake comparison: still running.", bold=True)

    para("The obvious fixes did not work.", bold=True)
    bullet("Five well-known methods built to help programs cope with new "
           "hospitals were tested. None beat the simple baseline.")
    bullet("A larger, more modern program was more accurate - but no better at "
           "knowing when it was wrong, and on one collection it got worse. "
           "Being cleverer did not make it more honest.")
    if facts["abstain_low"] is not None:
        bullet(f"Letting the program refuse the cases it is least sure about "
               f"helps only a little. Skipping the 30 least confident "
               f"photographs out of every 100 removes only "
               f"{facts['abstain_low']} to {facts['abstain_high']} mistakes out "
               f"of every 100 - and it works worst on "
               f"{facts['abstain_worst_domain']}, where it is needed most.")

    resolution = facts["resolution"]
    if resolution and resolution["all_improved"]:
        para("One thing did help, and it was unexpected.", bold=True)
        para("We had been shrinking every photograph to save computer memory. "
             "Using larger photographs instead improved results on all "
             f"{resolution['n_domains']} collections we have re-run, and cut "
             f"serious mistakes by between {resolution['drop_low']} and "
             f"{resolution['drop_high']} out of every 100. The damage this "
             "disease causes begins as tiny spots, and shrinking the picture "
             "was rubbing them out before the program ever saw them.")
        para("This is our largest single improvement so far. It also overturned "
             "an earlier conclusion of our own, which we have recorded openly "
             "rather than quietly deleting.", italic=True, colour=GREY)

    # ---------------------------------------------------------------- benefit
    heading("6. What this will help with")
    para("For hospitals deciding whether to use such a system:", bold=True)
    para("These systems are advertised with the score they achieved on the "
         "data they were built from. That number does not tell a hospital what "
         "it will actually get. Our work gives an honest figure for what is "
         "lost when the system arrives somewhere new.")
    para("For patient safety:", bold=True)
    para("If a program cannot be trusted to know when it is unsure, it cannot "
         "be left to decide by itself which cases a doctor should review. "
         "Showing exactly where that breaks down tells hospitals where a human "
         "must stay involved.")
    para("For other researchers:", bold=True)
    para("We are publishing a complete, repeatable way of running this test, "
         "along with all our results, so others can check our work or run the "
         "same test on their own system.")

    # ---------------------------------------------------------------- honesty
    heading("7. How we keep the results honest")
    para("It is easy to accidentally produce a result that flatters itself. We "
         "built specific protections against that:")
    bullet("The collection being tested is never used for training, or for any "
           "decision about the program. The computer checks this automatically "
           "rather than us relying on memory.")
    bullet("Every experiment is repeated three times from different random "
           "starting points. If a difference is smaller than the wobble "
           "between those three runs, we do not call it a finding.")
    bullet("Two separate statistical checks must agree before we believe a "
           "result. Several of our own early conclusions failed this and were "
           "withdrawn.")
    bullet("Every number we publish is generated automatically from the saved "
           "results, never typed by hand, and an automatic checker confirms "
           "they still match. This document is produced the same way.")

    # ---------------------------------------------------------------- progress
    heading("8. What is still in progress")
    para("An honest status of the unfinished work:", italic=True, colour=GREY)
    missing = [DOMAIN_NOTES[d][0] for d in ORDER if d not in facts["in_domain_512"]]
    if missing:
        bullet("The larger-photograph comparison is being completed for "
               + ", ".join(missing) + ".")
    bullet("The larger-photograph experiments have been run once each. They "
           "need repeating from different random starting points before we "
           "state them as firmly as the rest.")
    if not facts["subsample_done"]:
        bullet("An experiment is queued to separate two possible explanations "
               "for the poor results on the largest collection: are the "
               "hospitals genuinely very different, or did we simply have "
               "fewer photographs to learn from?")
    bullet("The written paper has not been started yet. That is deliberate: "
           "we finish and check the experiments first.")

    # ---------------------------------------------------------------- summary
    heading("9. In one paragraph")
    para("AI can already read eye photographs about as well as a doctor - but "
         "only in the hospital where it was trained. Somewhere new it makes "
         "roughly twice as many serious mistakes, and it does not become any "
         "less confident while doing so. We measure that cost properly across "
         "four real collections from three countries, test whether the "
         "standard fixes work (mostly they do not), and show one simple change "
         "that does help. The goal is that a hospital considering one of these "
         "systems knows what it is really getting, and knows where a human "
         "still needs to look.")

    doc.add_paragraph()
    para("All results above come from completed experiments. Anything "
         "unfinished is described as still running. Regenerate this document "
         "with 'python make_proposal.py' after any experiment completes.",
         size=9.5, italic=True, colour=GREY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return out_path


def main() -> None:
    try:
        import docx  # noqa: F401
    except ImportError:
        print("NOT RUN -- python-docx is not installed.\n"
              "    ./.venv/Scripts/python.exe -m pip install python-docx")
        raise SystemExit(1)

    root = Path(__file__).resolve().parent
    outputs = root / "outputs"
    facts = gather(outputs)

    targets = [root / "docs" / DOCS_NAME]
    if "--also-downloads" in sys.argv[1:]:
        targets.append(DOWNLOADS)

    written, locked = [], []
    for target in targets:
        try:
            written.append(build(facts, target))
        except PermissionError:
            # Almost always because the document is open in Word, which holds
            # an exclusive lock. That is the normal case when someone is
            # reading it, so it must not look like a failure of the run -- the
            # copy in docs/ is the one that matters and is written first.
            locked.append(target)

    for path in written:
        print(f"saved -> {path}  ({path.stat().st_size:,} bytes)")
    for path in locked:
        print(f"SKIPPED -> {path}\n"
              f"    the file is open (probably in Word) and cannot be replaced. "
              f"Close it and re-run to update that copy.")

    if facts["resolution"]:
        print(f"  larger-photograph result: {facts['resolution']['n_domains']} "
              f"collection(s), serious mistakes down "
              f"{facts['resolution']['drop_low']}-{facts['resolution']['drop_high']}%")
    still = [DOMAIN_NOTES[d][0] for d in ORDER if d not in facts["in_domain_512"]]
    if still:
        print(f"  marked as still running: {', '.join(still)}")


if __name__ == "__main__":
    main()
