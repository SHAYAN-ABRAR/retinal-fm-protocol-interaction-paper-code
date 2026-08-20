"""Stage-C method comparison: DDR + APTOS -> IDRiD, six methods, N seeds.

Everything except the method is held identical -- same splits, same seed, same
preprocessing, augmentation, optimiser, schedule, epochs and **batch size** -- so
differences are attributable to the method rather than the wiring.

Why batch 32 rather than the 16 used for the first baseline
-----------------------------------------------------------
Deep CORAL estimates a per-domain feature covariance *within each batch*. With
DDR+APTOS at their natural 76/24 mix, a batch of 16 leaves fewer than 4 APTOS
samples **43% of the time**, which makes the covariance estimate degenerate.
At batch 32 that falls to 3%. Batch size is therefore fixed at 32 for every
method, including a re-run of ERM, so the comparison stays clean.

Seeds
-----
The train/val/test **split is fixed** across seeds (it comes from the cached
manifest, built once with SplitConfig(seed=42)). Only training randomness --
weight init, shuffling, augmentation draws, MixStyle sampling -- varies. That is
deliberate: the question is whether a method difference survives *training*
variance, and re-splitting per seed would confound that with split variance.
Keeping the split fixed also keeps the test set identical, so paired comparisons
stay valid across seeds.

Usage:
    python run_method_comparison.py                  # all methods, seed 42
    python run_method_comparison.py --seeds 1,2      # all methods, seeds 1 and 2
    python run_method_comparison.py --seeds 1 erm ordinal
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

DEFAULT_SEEDS = (42,)
IMAGE_SIZE = 224
BACKBONE = "densenet121"
BATCH_SIZE = 32
EPOCHS = 20


def run_one(method_name: str, SEED: int) -> dict:
    import pandas as pd
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
    from src.training.methods import MethodConfig, build_method
    from src.training.trainer import TrainConfig, Trainer
    from src.utils.io import project_root, write_json
    from src.utils.registry import make_experiment_id, register_experiment
    from src.utils.seed import set_global_seed

    set_global_seed(SEED)
    root = project_root()
    outputs = root / "outputs"

    manifest = load_manifest(outputs / "reports" / "manifest_cached_224.csv")
    experiment = build_experiment_split(
        manifest, protocol="lodo", sources=["ddr", "aptos"], target="idrid"
    )
    augment = AugmentationConfig(image_size=IMAGE_SIZE)
    loader_config = LoaderConfig(batch_size=BATCH_SIZE, num_workers=2, seed=SEED)
    loaders = build_loaders(
        experiment,
        loader_config=loader_config,
        train_transform=build_train_transform(augment),
        eval_transform=build_eval_transform(augment),
        preprocess=None,
        expected_size=IMAGE_SIZE,
    )

    backbone = BackboneConfig(
        name=BACKBONE, image_size=resolve_input_size(BACKBONE, IMAGE_SIZE)
    )
    class_counts = (
        experiment.train["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
    )
    method = MethodConfig(name=method_name)
    built = build_method(method, backbone, class_counts=class_counts, device="cuda")
    total, trainable = count_parameters(built.model)

    experiment_id = make_experiment_id(
        protocol=experiment.protocol, sources=experiment.source_domains,
        target=experiment.target_domain, backbone=BACKBONE,
        method=f"{method_name}-b{BATCH_SIZE}", seed=SEED,
    )
    print(f"\n{'=' * 78}\n=== {experiment_id}\n{'=' * 78}", flush=True)

    train_config = TrainConfig(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=3e-4, weight_decay=1e-4,
        warmup_epochs=1, scheduler="cosine", amp=True, grad_clip_norm=1.0,
        early_stopping_patience=6, monitor="qwk", seed=SEED,
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
            "sources": experiment.source_domains,
            "target": experiment.target_domain,
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

    # Did the DG components actually fire? A Deep CORAL run whose alignment term
    # was always zero is ERM under another name, and must not be reported as DG.
    component_stats = built.statistics()
    print(f"component statistics: {component_stats}", flush=True)

    checkpoint = trainer.checkpoints.best_path
    from src.training.checkpointing import load_checkpoint

    # Rebuild rather than reuse, so evaluation starts from the saved best-QWK
    # weights rather than the final-epoch ones still in `built.model`.
    eval_built = build_method(method, backbone, class_counts=class_counts, device="cuda")
    eval_model = eval_built.model.to("cuda")
    load_checkpoint(checkpoint, model=eval_model, map_location="cuda")

    # to_probabilities / predict_fn are REQUIRED for the ordinal head: it emits
    # K-1 cumulative logits, and the default softmax path produces a K-1 vector.
    # Omitting them is what crashed the first sweep on all three ordinal methods.
    evaluation = evaluate_experiment(
        eval_model, loaders, experiment, experiment_id=experiment_id,
        device="cuda", predictions_dir=outputs / "predictions",
        to_probabilities=eval_built.to_probabilities,
        predict_fn=eval_built.predict_fn,
    )
    source = evaluation["results"]["source_val"]
    target = evaluation["results"]["target_test"]

    write_json(outputs / "reports" / f"{experiment_id}_evaluation.json", {
        "experiment_id": experiment_id,
        "method": built.description,
        "component_statistics": component_stats,
        "temperature": evaluation["temperature"],
        "results": {k: {"metrics": v.metrics, "calibration": v.calibration,
                        "calibration_scaled": v.calibration_scaled}
                    for k, v in evaluation["results"].items()},
    })

    summary = trainer.checkpoints.summary()
    register_experiment(outputs / "experiment_registry.csv", {
        "experiment_id": experiment_id, "status": "COMPLETE", "protocol": "lodo",
        "source_domains": experiment.source_domains, "target_domain": "idrid",
        "backbone": BACKBONE, "head": built.description.get("head"),
        "method": method_name, "loss": built.description.get("loss"),
        "imbalance_strategy": method.imbalance_strategy,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "accumulation_steps": 1, "effective_batch_size": BATCH_SIZE,
        "learning_rate": 3e-4, "weight_decay": 1e-4,
        "epochs_planned": EPOCHS, "epochs_run": len(history), "seed": SEED,
        "deterministic": False,
        "n_train": len(experiment.train), "n_val": len(experiment.val),
        "n_test": len(experiment.test),
        "best_epoch": summary["best_epoch"],
        "best_val_qwk": summary.get("best_val_qwk"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_checkpoint": str(checkpoint),
        "test_qwk": target.metrics["qwk"], "test_f1_macro": target.metrics["f1_macro"],
        "test_accuracy": target.metrics["accuracy"],
        "test_balanced_accuracy": target.metrics["balanced_accuracy"],
        "test_mae_grade": target.metrics["mae_grade"],
        "test_within_1_grade": target.metrics["within_1_grade"],
        "test_severe_error_rate": target.metrics["severe_error_rate"],
        "test_auroc_macro": target.metrics["auroc_macro"],
        "test_ece": target.calibration["ece"], "test_nll": target.calibration["nll"],
        "test_brier": target.calibration["brier"],
        "train_seconds": seconds,
        "peak_vram_gb": float(history["peak_vram_gb"].max()),
        "params_total_m": round(total / 1e6, 2),
        "params_trainable_m": round(trainable / 1e6, 2),
        "predictions_path": target.predictions_path,
        "history_path": str(history_path),
        "config_json": {**train_config.describe(), **built.description},
        "notes": f"Stage C method comparison, batch {BATCH_SIZE}, 1 seed; "
                 f"temperature fitted on source validation only; "
                 f"components={component_stats}",
    })

    row = {
        "method": method_name,
        "seed": SEED,
        "val_qwk": source.metrics["qwk"], "test_qwk": target.metrics["qwk"],
        "val_f1": source.metrics["f1_macro"], "test_f1": target.metrics["f1_macro"],
        "val_ece": source.calibration["ece"], "test_ece": target.calibration["ece"],
        "test_ece_scaled": target.calibration_scaled.get("ece"),
        "test_mae": target.metrics["mae_grade"],
        "test_severe": target.metrics["severe_error_rate"],
        "seconds": seconds,
    }
    print(f"RESULT {method_name}: {row}", flush=True)

    del eval_model, trainer, built
    torch.cuda.empty_cache()
    return row


def main() -> None:
    import pandas as pd

    from src.utils.hardware import assert_cuda_ready
    from src.utils.io import project_root

    assert_cuda_ready()

    arguments = sys.argv[1:]
    seeds = list(DEFAULT_SEEDS)
    if "--seeds" in arguments:
        index = arguments.index("--seeds")
        seeds = [int(s) for s in arguments[index + 1].split(",")]
        arguments = arguments[:index] + arguments[index + 2:]

    methods = arguments or [
        "erm", "ordinal", "deep_coral", "mixstyle", "mixstyle_ordinal", "deep_coral_ordinal"
    ]
    print(
        f"running {len(methods)} methods x {len(seeds)} seed(s) at batch {BATCH_SIZE}\n"
        f"  methods: {methods}\n  seeds  : {seeds}",
        flush=True,
    )

    rows = []
    for seed in seeds:
        for name in methods:
            try:
                rows.append(run_one(name, seed))
            except Exception as exc:  # noqa: BLE001 - one failure must not lose the rest
                import traceback

                print(f"!! {name} seed {seed} FAILED: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    if rows:
        path = project_root() / "outputs" / "tables" / "stage_c_method_seeds.csv"
        frame = pd.DataFrame(rows)
        if path.exists():
            frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
            frame = frame.drop_duplicates(["method", "seed"], keep="last")
        frame.to_csv(path, index=False)
        pd.set_option("display.width", 220)
        print("\n" + "=" * 78)
        print(frame.round(4).to_string(index=False))
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
