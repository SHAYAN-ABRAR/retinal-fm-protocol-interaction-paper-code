"""In-domain training: the ceiling the LODO numbers must be measured against.

Every gap reported in Phase 5 compares a LODO model's target score against its
own *source validation* -- different images from different domains. That says
how much worse the unseen domain is than the training distribution, but not the
number a clinician needs: **how much worse is a model that never saw this domain
than one trained on it?**

This script supplies the second term. Each run trains and tests inside a single
domain, so its test score is the best that domain supports under this backbone
and budget.

    DDR       8,697 train /  1,865 val /  1,862 test
    APTOS     2,809 train /    341 val /    354 test
    IDRiD       335 train /     70 val /    102 test
    EyePACS  24,574 train /  5,266 val /  5,268 test

Comparing fairly
----------------
The LODO runs were tested on **all** of each target (12,424 DDR images); an
in-domain model can only be tested on that domain's held-out split (1,862). A
direct comparison would be unfair -- the LODO score includes images the
in-domain model trained on. ``compare_to_lodo`` restricts the LODO predictions
to the same test image ids before differencing, using the saved per-image
predictions, so no retraining is needed.

IDRiD is expected to fail here. 335 training images is not a viable training
set, and demonstrating that is itself an argument for cross-domain training --
so it is run and reported rather than quietly skipped.

Usage:
    python run_in_domain.py                        # all four domains, seed 42
    python run_in_domain.py --domains idrid,aptos
    python run_in_domain.py --compare-only         # no training; compare saved runs
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

IMAGE_SIZE = 224
BACKBONE = "densenet121"
BATCH_SIZE = 32
EPOCHS = 20
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]

# The LODO experiment ids whose predictions the comparison restricts.
LODO_IDS = {
    "ddr": "lodo_aptos-eyepacs-idrid__ddr_densenet121_erm-b32_s42",
    "aptos": "lodo_ddr-eyepacs-idrid__aptos_densenet121_erm-b32_s42",
    "idrid": "lodo_aptos-ddr-eyepacs__idrid_densenet121_erm-b32_s42",
    "eyepacs": "lodo_aptos-ddr-idrid__eyepacs_densenet121_erm-b32_s42",
}


def run_one(domain: str, method_name: str, seed: int) -> dict | None:
    import torch

    from src.data.augmentations import (
        AugmentationConfig,
        build_eval_transform,
        build_train_transform,
    )
    from src.data.loaders import LoaderConfig, build_loaders
    from src.data.preprocessing import PreprocessConfig
    from src.data.splits import build_experiment_split
    from src.data.unified_dataset import load_manifest
    from src.evaluation.evaluate import evaluate_experiment
    from src.models.backbones import BackboneConfig, count_parameters, resolve_input_size
    from src.training.checkpointing import load_checkpoint
    from src.training.methods import MethodConfig, build_method
    from src.training.trainer import TrainConfig, Trainer
    from src.utils.io import project_root, write_json
    from src.utils.registry import make_experiment_id, register_experiment
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    root = project_root()
    outputs = root / "outputs"

    manifest = load_manifest(outputs / "reports" / "manifest_cached_224.csv")
    experiment = build_experiment_split(
        manifest, protocol="in_domain", sources=[domain], target=domain
    )

    experiment_id = make_experiment_id(
        protocol="in_domain", sources=[domain], target=domain,
        backbone=BACKBONE, method=f"{method_name}-b{BATCH_SIZE}", seed=seed,
    )
    print(f"\n{'=' * 78}\n=== {experiment_id}\n    {experiment.sizes()}\n{'=' * 78}", flush=True)
    for note in experiment.notes:
        print(f"    {note}", flush=True)

    if len(experiment.train) < 1000:
        print(f"    !! only {len(experiment.train)} training images. This is reported "
              "as a measured limit of the domain, not as a tuned result.", flush=True)

    augment = AugmentationConfig(image_size=IMAGE_SIZE)
    loader_config = LoaderConfig(batch_size=BATCH_SIZE, num_workers=2, seed=seed)
    loaders = build_loaders(
        experiment,
        loader_config=loader_config,
        train_transform=build_train_transform(augment),
        eval_transform=build_eval_transform(augment),
        preprocess=None,
        expected_size=IMAGE_SIZE,
    )

    backbone = BackboneConfig(name=BACKBONE, image_size=resolve_input_size(BACKBONE, IMAGE_SIZE))
    class_counts = (
        experiment.train["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
    )
    method = MethodConfig(name=method_name)
    built = build_method(method, backbone, class_counts=class_counts, device="cuda")
    total, trainable = count_parameters(built.model)

    train_config = TrainConfig(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=3e-4, weight_decay=1e-4,
        warmup_epochs=1, scheduler="cosine", amp=True, grad_clip_norm=1.0,
        early_stopping_patience=6, monitor="qwk", seed=seed,
    )
    trainer = Trainer(
        built.model, built.loss_fn, train_config,
        device="cuda",
        checkpoint_dir=outputs / "checkpoints",
        experiment_id=experiment_id,
        extra_config={
            **built.description,
            "preprocess": PreprocessConfig(image_size=IMAGE_SIZE).describe(),
            "augmentation": augment.describe(),
            "loaders": loader_config.describe(),
            "sources": [domain],
            "target": domain,
        },
        feature_loss=built.feature_loss,
        batch_hook=built.batch_hook,
        to_probabilities=built.to_probabilities,
    )
    trainer.resume()

    started = time.perf_counter()
    trainer.fit(loaders["train"], loaders["val"])
    seconds = round(time.perf_counter() - started, 1)

    history = trainer.history_frame()
    history_path = outputs / "logs" / f"{experiment_id}_history.csv"
    history.to_csv(history_path, index=False)

    checkpoint = trainer.checkpoints.best_path
    eval_built = build_method(method, backbone, class_counts=class_counts, device="cuda")
    eval_model = eval_built.model.to("cuda")
    load_checkpoint(checkpoint, model=eval_model, map_location="cuda")

    evaluation = evaluate_experiment(
        eval_model, loaders, experiment, experiment_id=experiment_id,
        device="cuda", predictions_dir=outputs / "predictions",
        to_probabilities=eval_built.to_probabilities,
        predict_fn=eval_built.predict_fn,
    )
    target_result = evaluation["results"]["target_test"]
    source_result = evaluation["results"]["source_val"]

    write_json(outputs / "reports" / f"{experiment_id}_evaluation.json", {
        "experiment_id": experiment_id,
        "method": built.description,
        "temperature": evaluation["temperature"],
        "results": {k: {"metrics": v.metrics, "calibration": v.calibration,
                        "calibration_scaled": v.calibration_scaled}
                    for k, v in evaluation["results"].items()},
    })

    summary = trainer.checkpoints.summary()
    register_experiment(outputs / "experiment_registry.csv", {
        "experiment_id": experiment_id, "status": "COMPLETE", "protocol": "in_domain",
        "source_domains": [domain], "target_domain": domain,
        "backbone": BACKBONE, "head": built.description.get("head"),
        "method": method_name, "loss": built.description.get("loss"),
        "imbalance_strategy": method.imbalance_strategy,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "accumulation_steps": 1, "effective_batch_size": BATCH_SIZE,
        "learning_rate": 3e-4, "weight_decay": 1e-4,
        "epochs_planned": EPOCHS, "epochs_run": len(history), "seed": seed,
        "deterministic": False,
        "n_train": len(experiment.train), "n_val": len(experiment.val),
        "n_test": len(experiment.test),
        "best_epoch": summary["best_epoch"],
        "best_val_qwk": summary.get("best_val_qwk"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_checkpoint": str(checkpoint),
        "test_qwk": target_result.metrics["qwk"],
        "test_f1_macro": target_result.metrics["f1_macro"],
        "test_accuracy": target_result.metrics["accuracy"],
        "test_balanced_accuracy": target_result.metrics["balanced_accuracy"],
        "test_mae_grade": target_result.metrics["mae_grade"],
        "test_within_1_grade": target_result.metrics["within_1_grade"],
        "test_severe_error_rate": target_result.metrics["severe_error_rate"],
        "test_auroc_macro": target_result.metrics["auroc_macro"],
        "test_ece": target_result.calibration["ece"],
        "test_nll": target_result.calibration["nll"],
        "test_brier": target_result.calibration["brier"],
        "train_seconds": seconds,
        "peak_vram_gb": float(history["peak_vram_gb"].max()) if len(history) else None,
        "params_total_m": round(total / 1e6, 2),
        "params_trainable_m": round(trainable / 1e6, 2),
        "predictions_path": target_result.predictions_path,
        "history_path": str(history_path),
        "config_json": {**train_config.describe(), **built.description},
        "notes": "In-domain ceiling: train/val/test all from this domain",
    })

    row = {
        "domain": domain, "method": method_name, "seed": seed,
        "n_train": len(experiment.train), "n_test": len(experiment.test),
        "val_qwk": source_result.metrics["qwk"],
        "test_qwk": target_result.metrics["qwk"],
        "test_f1": target_result.metrics["f1_macro"],
        "test_ece": target_result.calibration["ece"],
        "test_ece_scaled": target_result.calibration_scaled.get("ece"),
        "test_severe": target_result.metrics["severe_error_rate"],
        "seconds": seconds,
    }
    print(f"RESULT {domain}: {row}", flush=True)

    del eval_model, eval_built, trainer, built
    torch.cuda.empty_cache()
    return row


def compare_to_lodo(seed: int = 42, method: str = "erm") -> None:
    """The cost of deployment: in-domain vs LODO, on identical test images."""
    import pandas as pd

    from src.evaluation.bootstrap import paired_bootstrap_difference
    from src.evaluation.metrics import quadratic_weighted_kappa
    from src.utils.io import project_root
    from src.utils.registry import make_experiment_id

    outputs = project_root() / "outputs"
    predictions = outputs / "predictions"
    rows = []

    print(f"\n{'=' * 96}\nCOST OF CROSS-DOMAIN DEPLOYMENT (identical test images)\n{'=' * 96}")
    header = (f"{'domain':<9} {'n_test':>7} {'in-domain':>10} {'LODO':>8} "
              f"{'delta':>8} {'95% CI':>22} {'verdict':>12}")
    print(header)
    print("-" * len(header))

    for domain in ALL_DOMAINS:
        in_id = make_experiment_id(
            protocol="in_domain", sources=[domain], target=domain,
            backbone=BACKBONE, method=f"{method}-b{BATCH_SIZE}", seed=seed,
        )
        in_path = predictions / f"{in_id}__target_test[{domain}]_predictions.csv"
        lodo_path = predictions / f"{LODO_IDS[domain]}__target_test[{domain}]_predictions.csv"

        if not in_path.exists():
            print(f"{domain:<9} {'NOT RUN -- in-domain model has not been trained':>60}")
            continue
        if not lodo_path.exists():
            print(f"{domain:<9} {'NOT RUN -- no LODO predictions to compare against':>60}")
            continue

        in_frame = pd.read_csv(in_path)
        lodo_frame = pd.read_csv(lodo_path)

        # Restrict the LODO run to the in-domain test images. Without this the
        # LODO model is scored on images the in-domain model trained on.
        shared = set(in_frame["image_id"]) & set(lodo_frame["image_id"])
        if len(shared) != len(in_frame):
            print(f"  !! {domain}: {len(in_frame) - len(shared)} in-domain test image(s) "
                  "absent from the LODO predictions; comparing the intersection")
        in_frame = in_frame[in_frame["image_id"].isin(shared)].sort_values("image_id")
        lodo_frame = lodo_frame[lodo_frame["image_id"].isin(shared)].sort_values("image_id")

        truth_a = in_frame["true_grade"].to_numpy()
        truth_b = lodo_frame["true_grade"].to_numpy()
        if not (truth_a == truth_b).all():
            raise AssertionError(
                f"{domain}: the two runs disagree on ground truth for the same image ids. "
                "One of the prediction files is stale; re-evaluate before comparing."
            )

        in_pred = in_frame["predicted_grade"].to_numpy()
        lodo_pred = lodo_frame["predicted_grade"].to_numpy()
        in_qwk = quadratic_weighted_kappa(truth_a, in_pred)
        lodo_qwk = quadratic_weighted_kappa(truth_a, lodo_pred)

        # Argument order matters: paired_bootstrap_difference returns
        # metric(B) - metric(A), so passing (in_pred, lodo_pred) makes the
        # interval describe LODO minus in-domain -- the same quantity, with the
        # same sign, as the delta column beside it. Reversing these would print
        # a CI whose sign contradicts the delta it is meant to bound.
        difference = paired_bootstrap_difference(
            truth_a, in_pred, lodo_pred, metric="qwk", n_bootstrap=2000, seed=seed,
        )
        lower, upper = difference["ci_lower"], difference["ci_upper"]
        excludes_zero = bool(difference["significant"])
        # The paired interval resamples images. It says nothing about training
        # variance, so a "REAL" verdict here still needs the seed bar before it
        # goes in the paper -- see the note printed below the table.
        verdict = "REAL" if excludes_zero else "not resolved"
        print(f"{domain:<9} {len(shared):>7} {in_qwk:>10.4f} {lodo_qwk:>8.4f} "
              f"{lodo_qwk - in_qwk:>+8.4f} "
              f"{f'[{lower:+.4f}, {upper:+.4f}]':>22} {verdict:>12}")
        rows.append({
            "domain": domain, "n_test": len(shared),
            "in_domain_qwk": in_qwk, "lodo_qwk": lodo_qwk,
            "delta_qwk": lodo_qwk - in_qwk,
            "ci_lower": lower, "ci_upper": upper,
            "ci_excludes_zero": excludes_zero,
        })

    if rows:
        path = outputs / "tables" / f"in_domain_vs_lodo_{method}_s{seed}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\nsaved -> {path}")
        print("\nNote: single seed. ERM's across-seed SD on target QWK is 0.032; "
              "a delta smaller than that is not established by this table alone.")


def main() -> None:
    import pandas as pd

    from src.utils.hardware import assert_cuda_ready
    from src.utils.io import project_root

    arguments = sys.argv[1:]
    compare_only = "--compare-only" in arguments
    if compare_only:
        arguments.remove("--compare-only")

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    method = _take("--method", "erm")
    seed = int(_take("--seed", "42"))
    domains = _take("--domains", ",".join(ALL_DOMAINS)).split(",")

    if compare_only:
        compare_to_lodo(seed=seed, method=method)
        return

    assert_cuda_ready()
    print(f"in-domain: method={method}, seed={seed}, domains={domains}\n"
          f"  backbone={BACKBONE}, batch={BATCH_SIZE}, epochs={EPOCHS}", flush=True)

    rows = []
    for domain in domains:
        try:
            result = run_one(domain, method, seed)
            if result is not None:
                rows.append(result)
        except Exception as exc:  # noqa: BLE001 - one failure must not lose the rest
            import traceback

            print(f"!! domain {domain} FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    if rows:
        path = project_root() / "outputs" / "tables" / "in_domain_results.csv"
        frame = pd.DataFrame(rows)
        if path.exists():
            frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
            frame = frame.drop_duplicates(["domain", "method", "seed"], keep="last")
        frame.to_csv(path, index=False)
        pd.set_option("display.width", 220)
        print("\n" + "=" * 78)
        print(frame.round(4).to_string(index=False))
        print(f"\nsaved -> {path}")

    compare_to_lodo(seed=seed, method=method)


if __name__ == "__main__":
    main()
