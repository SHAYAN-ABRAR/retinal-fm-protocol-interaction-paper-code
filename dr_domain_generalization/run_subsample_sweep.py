"""How much of the EyePACS gap is training-set size, and how much is domain shift?

Phase 7 §6 estimates the split from two points under a log-linear assumption
and says so explicitly: *"This is a bound from two points under a log-linear
assumption at one seed, not a measurement. A subsample sweep at fixed sources
would settle it for about an hour of compute."* This is that sweep.

The confound
------------
EyePACS is the hardest target (LODO QWK 0.4147 against an in-domain 0.7063),
but it is also the target with the smallest source pool: holding it out leaves
only 11,841 training images, a third of what the other targets get. Small pool
and large shift are perfectly confounded in the LODO design, so the gap cannot
be attributed without varying one of them.

The design
----------
Hold the source **domains** fixed (DDR + APTOS + IDRiD) and vary only the
number of training images. Everything else -- validation split, target test
split, architecture, schedule, resolution -- is identical to the Phase 5 LODO
run at 224 px, so the 100% point *is* that run and does not need repeating.

Subsampling is stratified by (domain, grade). Taking a plain random subset
would let the domain mix and the grade prior drift with size, and the sweep
would then measure composition change rather than size.

Reading the result
------------------
Fit QWK against log(n) across the sweep and extrapolate to the in-domain budget
of 24,574. Whatever that predicts is the part of the gap attributable to size;
the remainder is shift. If the curve is flat, size explains nothing and the gap
is shift. If it is steep, Phase 7's "one third size" bound was optimistic in
the other direction.

Cost: ~0.6 h per seed for three fractions at 224 px.

Usage:
    python run_subsample_sweep.py                       # 3 fractions x 3 seeds
    python run_subsample_sweep.py --fractions 0.25,0.5
    python run_subsample_sweep.py --seeds 42
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

IMAGE_SIZE = 224
BACKBONE = "densenet121"
BATCH_SIZE = 32
EPOCHS = 20
TARGET = "eyepacs"
SOURCES = ["ddr", "aptos", "idrid"]
# 1.0 is deliberately absent: it is the Phase 5 LODO run, already done at three
# seeds. Re-running it would spend an hour to reproduce a number we hold.
FRACTIONS = [0.25, 0.5, 0.75]
SEEDS = [42, 1, 2]


def _method_tag(fraction: float) -> str:
    """Fraction is part of the run's identity, so it belongs in the id.

    Without it every fraction would write to the same checkpoint directory and
    the same predictions file, and each run would destroy the last -- the
    failure this project has already had twice, in in_domain_results.csv and
    lodo_results.csv.
    """
    return f"erm-b{BATCH_SIZE}-f{int(round(fraction * 100)):03d}"


def stratified_subsample(frame, fraction: float, seed: int):
    """A fraction of the training rows, holding domain and grade mix fixed.

    Sampling uniformly at random would let the domain proportions drift, and
    IDRiD (335 images against DDR's 8,697) would thin out of the small
    fractions. The sweep would then confound size with composition, which is
    the confound it exists to break.

    Every (domain, grade) cell keeps at least one image, so a rare cell is
    never emptied entirely. That biases the smallest fractions very slightly
    upward in the rare cells; the alternative -- letting grade 4 of IDRiD
    disappear at 25% -- would change what is being measured.

    Written as an explicit loop rather than ``groupby().apply()``: from pandas
    2.2 the grouping columns are excluded from the frame the callable receives,
    so the obvious one-liner silently returns rows without ``domain``.
    """
    import pandas as pd

    parts = []
    for _, group in frame.groupby(["domain", "grade"], sort=False):
        n = max(1, int(round(len(group) * fraction)))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_one(fraction: float, seed: int) -> dict | None:
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
    from src.training.checkpointing import load_checkpoint
    from src.training.methods import MethodConfig, build_method
    from src.training.trainer import TrainConfig, Trainer
    from src.utils.io import project_root, write_json
    from src.utils.registry import make_experiment_id, register_experiment
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    outputs = project_root() / "outputs"

    manifest = load_manifest(outputs / "reports" / f"manifest_cached_{IMAGE_SIZE}.csv")
    experiment = build_experiment_split(
        manifest, protocol="lodo", sources=SOURCES, target=TARGET
    )

    full_train = len(experiment.train)
    experiment.train = stratified_subsample(experiment.train, fraction, seed)
    experiment.notes.append(
        f"train subsampled to {fraction:.0%}: {full_train} -> {len(experiment.train)}, "
        f"stratified by (domain, grade)"
    )

    # The target domain must not appear in training at any fraction. Shrinking
    # a split cannot introduce leakage, but this is the assertion the project
    # makes everywhere else and a silent change here would be invisible.
    assert TARGET not in set(experiment.train["domain"]), \
        f"{TARGET} leaked into the training split"
    assert TARGET not in set(experiment.val["domain"]), \
        f"{TARGET} leaked into the validation split"

    experiment_id = make_experiment_id(
        protocol="lodo", sources=SOURCES, target=TARGET,
        backbone=BACKBONE, method=_method_tag(fraction), seed=seed,
    )
    print(f"\n{'=' * 78}\n=== {experiment_id}\n    {experiment.sizes()}\n{'=' * 78}",
          flush=True)
    for note in experiment.notes:
        print(f"    {note}", flush=True)

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

    backbone = BackboneConfig(
        name=BACKBONE, image_size=resolve_input_size(BACKBONE, IMAGE_SIZE))
    class_counts = (
        experiment.train["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
    )
    method = MethodConfig(name="erm")
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
            "sources": SOURCES, "target": TARGET,
            "subsample_fraction": fraction,
            "n_train_full": full_train,
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
    history.to_csv(outputs / "logs" / f"{experiment_id}_history.csv", index=False)

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
    source_result = evaluation["results"]["source_val"]
    target_result = evaluation["results"]["target_test"]

    write_json(outputs / "reports" / f"{experiment_id}_evaluation.json", {
        "experiment_id": experiment_id,
        "subsample_fraction": fraction,
        "method": built.description,
        "temperature": evaluation["temperature"],
        "results": {k: {"metrics": v.metrics, "calibration": v.calibration,
                        "calibration_scaled": v.calibration_scaled}
                    for k, v in evaluation["results"].items()},
    })

    summary = trainer.checkpoints.summary()
    register_experiment(outputs / "experiment_registry.csv", {
        "experiment_id": experiment_id, "status": "COMPLETE", "protocol": "lodo",
        "source_domains": SOURCES, "target_domain": TARGET,
        "backbone": BACKBONE, "head": built.description.get("head"),
        "method": "erm", "loss": built.description.get("loss"),
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
        "test_ece": target_result.calibration["ece"],
        "test_nll": target_result.calibration["nll"],
        "test_brier": target_result.calibration["brier"],
        "train_seconds": seconds,
        "peak_vram_gb": float(history["peak_vram_gb"].max()) if len(history) else None,
        "params_total_m": round(total / 1e6, 2),
        "params_trainable_m": round(trainable / 1e6, 2),
        "predictions_path": target_result.predictions_path,
        "config_json": {**train_config.describe(), **built.description},
        "notes": f"subsample sweep at {fraction:.0%} of the LODO source pool",
    })

    row = {
        "target": TARGET, "method": "erm", "seed": seed,
        "backbone": BACKBONE, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "fraction": fraction,
        "n_train": len(experiment.train), "n_train_full": full_train,
        "n_test": len(experiment.test),
        "source_qwk": source_result.metrics["qwk"],
        "target_qwk": target_result.metrics["qwk"],
        "target_f1": target_result.metrics["f1_macro"],
        "target_ece": target_result.calibration["ece"],
        "target_ece_scaled": target_result.calibration_scaled.get("ece"),
        "target_severe": target_result.metrics["severe_error_rate"],
        "seconds": seconds,
    }
    print(f"RESULT f={fraction:.2f} seed={seed}: {row}", flush=True)

    del eval_model, eval_built, trainer, built
    torch.cuda.empty_cache()
    return row


def main() -> None:
    import pandas as pd

    from src.utils.hardware import assert_cuda_ready
    from src.utils.io import project_root
    from src.utils.registry import merge_results_table

    arguments = sys.argv[1:]

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    fractions = [float(f) for f in _take("--fractions", ",".join(map(str, FRACTIONS))).split(",")]
    seeds = [int(s) for s in _take("--seeds", ",".join(map(str, SEEDS))).split(",")]

    assert_cuda_ready()
    print(f"subsample sweep: target={TARGET}, sources={SOURCES}\n"
          f"  fractions={fractions}, seeds={seeds}, {IMAGE_SIZE}px, batch {BATCH_SIZE}",
          flush=True)

    path = project_root() / "outputs" / "tables" / "subsample_sweep.csv"

    def save(new_rows):
        frame = pd.DataFrame(new_rows)
        if path.exists():
            frame = merge_results_table(
                pd.read_csv(path), frame,
                ["target", "method", "seed", "backbone", "image_size", "fraction"],
            )
            frame = frame.sort_values(["fraction", "seed"]).reset_index(drop=True)
        frame.to_csv(path, index=False)

    rows = []
    for seed in seeds:
        for fraction in fractions:
            try:
                result = run_one(fraction, seed)
                if result is not None:
                    rows.append(result)
                    save([result])       # written per run, not once at the end
            except Exception as exc:  # noqa: BLE001
                import traceback

                print(f"!! fraction {fraction} seed {seed} FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    if rows:
        pd.set_option("display.width", 220)
        print("\n" + "=" * 78)
        print(pd.read_csv(path).round(4).to_string(index=False))
        print(f"\nsaved -> {path}")
        print("\nNext: python analyse_subsample.py")


if __name__ == "__main__":
    main()
