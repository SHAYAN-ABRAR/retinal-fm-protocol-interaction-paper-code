"""Is the 20-epoch budget adequate for both initialisations?

**Source validation only.** The held-out target plays no part in this judgement.
Deciding whether the recipe is long enough by looking at DDR would tune the
protocol on the domain the study exists to keep untouched -- the same mistake as
selecting a checkpoint on the target, one level up.

The question has two failure modes, not one:

* **Truncation** -- the best epoch sits at the end of the budget while the metric
  is still climbing materially. The cap is then an arbitrary stopping point and
  the reported model is not the model the recipe would produce.
* **Over-running** -- the best epoch sits far from the end and the tail is spent
  memorising. Selection on best-source-validation makes this harmless for the
  reported number, but it means one fixed budget is not equally well matched to
  both arms, which is a sensitivity worth stating rather than discovering later.

So the script reports, per arm: the best epoch and its position in the budget,
the improvement over the final third, the train/validation loss gap as an
overfitting signature, and whether early stopping fired.

Runs from saved history. No GPU.

Usage:
    python analyse_ft_convergence.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

RUNS = [
    ("ImageNet-MAE", "lodo_aptos-eyepacs-idrid__ddr_vit-large-mae-in1k"
                     "_erm-b16-ftfull-lr0.0001_s42"),
    ("RETFound-CFP", "lodo_aptos-eyepacs-idrid__ddr_retfound-cfp"
                     "_erm-b16-ftfull-lr0.0001_s42"),
]
MONITOR = "qwk"
BUDGET = 20
# A tail gain smaller than this is a plateau rather than continued learning.
# Set against the across-seed SD of the partial fine-tuning arm (~0.026): a
# budget is not truncating if extending it would move the metric by less than
# seed noise already does.
MATERIAL = 0.026


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root

    outputs = project_root() / "outputs"
    rows = []
    curves = {}

    for label, experiment_id in RUNS:
        path = outputs / "logs" / f"{experiment_id}_history.csv"
        if not path.exists():
            print(f"NOT RUN -- {label}: no history at {path.name}")
            continue
        history = pd.read_csv(path)
        column = next((c for c in (f"val_{MONITOR}", MONITOR, "val_qwk")
                       if c in history.columns), None)
        if column is None:
            print(f"!! {label}: no {MONITOR} column in {list(history.columns)}")
            continue

        values = history[column].to_numpy(dtype=float)
        curves[label] = values
        best_index = int(values.argmax())
        last_index = len(values) - 1

        # How much of the final third of the *run* was worth having: the gain
        # from the best value before that third to the overall best.
        third = max(1, len(values) // 3)
        before_tail = float(values[:len(values) - third].max())
        tail_gain = float(values.max() - before_tail)

        train_column = next((c for c in ("train_loss", "loss")
                             if c in history.columns), None)
        val_loss_column = next((c for c in ("val_loss",)
                                if c in history.columns), None)
        rows.append({
            "arm": label,
            "epochs_run": len(values),
            "budget": BUDGET,
            "stopped_early": len(values) < BUDGET,
            "best_epoch": best_index,
            "best_value": float(values[best_index]),
            "final_value": float(values[last_index]),
            "epochs_after_best": last_index - best_index,
            "position_in_budget": round(best_index / (BUDGET - 1), 3),
            "tail_gain": tail_gain,
            "tail_gain_material": bool(tail_gain > MATERIAL),
            "train_loss_first": (float(history[train_column].iloc[0])
                                 if train_column else float("nan")),
            "train_loss_last": (float(history[train_column].iloc[-1])
                                if train_column else float("nan")),
            "val_loss_min": (float(history[val_loss_column].min())
                             if val_loss_column else float("nan")),
            "val_loss_last": (float(history[val_loss_column].iloc[-1])
                              if val_loss_column else float("nan")),
        })

    if not rows:
        print("nothing to analyse")
        return 1

    frame = pd.DataFrame(rows)
    frame["overfit_ratio"] = frame.val_loss_last / frame.val_loss_min
    out = outputs / "tables" / "ft_convergence_source_validation.csv"
    frame.to_csv(out, index=False)

    print("Source-validation convergence, seed 42. The held-out target is not "
          "consulted.\n")
    for _, r in frame.iterrows():
        print(f"  {r['arm']}")
        print(f"    epochs run           {int(r.epochs_run)} of {int(r.budget)}"
              f"{'  (early stopped)' if r.stopped_early else ''}")
        print(f"    best {MONITOR:<16} {r.best_value:.4f} at epoch "
              f"{int(r.best_epoch)}  "
              f"({r.position_in_budget:.0%} through the budget)")
        print(f"    epochs after best    {int(r.epochs_after_best)}")
        print(f"    final {MONITOR:<15} {r.final_value:.4f}")
        print(f"    gain in final third  {r.tail_gain:+.4f}"
              f"  ({'material' if r.tail_gain_material else 'below seed noise'})")
        print(f"    train loss           {r.train_loss_first:.4f} -> "
              f"{r.train_loss_last:.4f}")
        print(f"    val loss             min {r.val_loss_min:.4f}, "
              f"last {r.val_loss_last:.4f}  "
              f"({r.overfit_ratio:.2f}x its minimum)")
        print()

    print("  verdict per arm:")
    for _, r in frame.iterrows():
        truncated = (r.epochs_after_best <= 1) and r.tail_gain_material
        if truncated:
            verdict = ("TRUNCATED -- best sits at the cap and the tail gain "
                       "exceeds seed noise; extend the budget")
        elif r.epochs_after_best >= 5 and not r.tail_gain_material:
            verdict = ("MORE THAN ADEQUATE -- plateaued well before the cap; "
                       "the tail is spent overfitting")
        else:
            verdict = "ADEQUATE -- plateaued inside the budget"
        print(f"    {r['arm']:<14} {verdict}")

    print(f"\n  saved -> {out.name}")

    if len(frame) == 2:
        a, b = frame.iloc[0], frame.iloc[1]
        print(f"\n  The two arms plateau at different points "
              f"({int(a.best_epoch)} vs {int(b.best_epoch)}) and overfit to "
              f"different degrees ({a.overfit_ratio:.2f}x vs "
              f"{b.overfit_ratio:.2f}x).")
        print("  Best-source-validation selection makes this harmless for the "
              "reported model,")
        print("  but one fixed budget is not equally well matched to both "
              "initialisations, and")
        print("  that belongs in the limitations rather than being discovered "
              "by a reader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
