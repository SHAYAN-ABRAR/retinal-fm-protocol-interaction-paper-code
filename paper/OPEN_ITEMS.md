# Open items for the manuscript

Status of `paper/main.tex` (draft v0.1). Nothing here is a blocker on the
science; these are writing and assembly tasks.

## Blocked on running experiments

- [ ] **IDRiD fine-tune, seeds 3 and 4** — running now. The Limitations section
      currently concedes that IDRiD's fine-tuned verdict rests on three seeds,
      which is the sample size §II-C argues is insufficient. When these land,
      update Table `finetune_comparison.csv`, Fig. 1, §IV-A and the limitation.
- [ ] **Frozen probes, seeds 5–9** — queued behind IDRiD. Takes the surviving
      arm to ten seeds; update §IV-A and Fig. 1.

## Tables to emit (do not transcribe)

- [ ] Table I — dataset characteristics. Exists as
      `outputs/tables/table1_dataset_characteristics.tex`; wire the input.
- [ ] Table II — foundation-model comparison, frozen and fine-tuned side by
      side. **Generator does not exist yet**; add to `export_paper_tables.py`
      reading `linear_probe_comparison_*.csv` and `finetune_comparison.csv`.
- [ ] Table III — DG method comparison. Add a generator reading
      `method_comparison_lodo_{eyepacs,ddr}.csv`.
- [ ] Table IV — deployment cost. `table_deployment_cost.tex` exists but is the
      512 px seed-42 version; the paper quotes the 224 px three-seed paired
      numbers. Regenerate.

## Writing

- [ ] Related work section — currently absent. Needs DomainBed
      (`gulrajani2021domainbed`), medical foundation models, and the
      linear-probe-vs-fine-tune literature.
- [ ] Author list, affiliations, funding, ethics statement.
- [ ] IEEEtran `\num{}` requires `siunitx`; either add the package or spell the
      counts out.
- [ ] Decide whether calibration results (ECE, temperature scaling) get their
      own subsection or fold into the deployment-cost paragraph.

## Discipline

`paper/check_numbers.py` verifies every four-decimal value in `main.tex`
against a generator table and exits non-zero on any that matches none. Run it
before each commit. It currently passes.

Two values in the phase reports were previously typed from memory and were
wrong; that is what this script exists to prevent.
