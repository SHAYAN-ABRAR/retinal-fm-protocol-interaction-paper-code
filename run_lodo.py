"""The four leave-one-domain-out experiments (brief section 20).

Each run trains on three domains and evaluates on the fourth, which it has never
seen. The held-out domain contributes **test data only** -- never training,
validation, early stopping, model selection or temperature fitting. That is
asserted in ``src/data/splits.py``, not left to convention.

    Experiment 1: DDR + APTOS + EyePACS  ->  IDRiD     (36,080 train /    507 test)
    Experiment 2: DDR + IDRiD + EyePACS  ->  APTOS     (33,606 train /  3,504 test)
    Experiment 3: DDR + APTOS + IDRiD    ->  EyePACS   (11,841 train / 35,108 test)
    Experiment 4: APTOS + IDRiD + EyePACS ->  DDR      (27,718 train / 12,424 test)

Cost
----
Roughly 3 hours for one method across all four targets at 20 epochs, dominated
by the ~36k-image training sets. Run it for ERM first: the Stage-C comparison
found no method beating ERM, so establishing the ERM matrix is the priority.

Usage:
    python run_lodo.py                                # ERM, all four targets, seed 42
    python run_lodo.py --method mixstyle --seeds 42
    python run_lodo.py --targets idrid,eyepacs
    python run_lodo.py --method groupdro --domain-balanced --targets eyepacs
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

IMAGE_SIZE = 224
# Set by main() via --backbone. A module global rather than a parameter so the
# experiment id, the registry row and the figures cannot disagree about which
# network produced a result.
BACKBONE = "densenet121"
BATCH_SIZE = 32
EPOCHS = 20
# Domain-balanced batches. Off by default so every existing run is reproducible
# by rerunning this file; set by main() via --domain-balanced.
DOMAIN_BALANCED = False
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]


# Set by --irm-anneal-iters. None means the published default (500).
IRM_ANNEAL_ITERS: int | None = None
# Set by --trainable-blocks. None means the whole backbone trains.
TRAINABLE_BLOCKS: int | None = None
# Set by --full-finetune. Distinct from --trainable-blocks 24, which is NOT
# full fine-tuning: freeze_backbone() freezes everything and then unfreezes only
# the modules _find_blocks() returns, which for a timm ViT leaves cls_token,
# pos_embed, patch_embed.proj.{weight,bias} and norm.{weight,bias} frozen --
# six tensors, verified. Passing trainable_blocks=None instead never freezes
# anything, and assert_full_finetune() checks that afterwards rather than
# trusting it.
FULL_FINETUNE = False
# Set by --gradient-checkpointing. Recorded because it changes memory and step
# time but must not change results; if it ever does, that is a bug worth seeing.
GRADIENT_CHECKPOINTING = False


def enable_gradient_checkpointing(model) -> None:
    """Turn on timm gradient checkpointing, or refuse.

    timm exposes ``set_grad_checkpointing`` on the models that support it. If
    the selected backbone does not, this raises rather than returning quietly:
    a silent no-op would let a full fine-tune be launched on a memory budget
    that assumed checkpointing was active, and it would OOM hours later with no
    indication of why.
    """
    backbone = getattr(model, "backbone", model)
    setter = getattr(backbone, "set_grad_checkpointing", None)
    if setter is None or not callable(setter):
        raise RuntimeError(
            f"{type(backbone).__name__} exposes no set_grad_checkpointing(); "
            "gradient checkpointing was requested but cannot be enabled. "
            "Re-measure VRAM before running without it."
        )
    setter(True)


def adaptation_mode() -> str:
    """The protocol, in words, for the registry and the paper."""
    if FULL_FINETUNE:
        return "full_finetune"
    if TRAINABLE_BLOCKS is None:
        return "full_network"          # every pre-ViT run: no freezing applied
    if TRAINABLE_BLOCKS == 0:
        return "linear_probe"
    return f"partial_finetune_{TRAINABLE_BLOCKS}"


def recorded_trainable_blocks() -> int:
    """-1 for the whole network, never None.

    ``trainable_blocks`` is part of the dedup key in both the registry and the
    results table, and NaN never equals NaN. Two runs differing only in a key
    column that is blank therefore merge into one row, and the second silently
    replaces the first -- the same failure that already cost this project three
    results under ``-dbal``, ``-a1170`` and ``-tb4-lr0.0001``.

    ``KEY_COLUMN_DEFAULTS`` states the convention ("-1 encodes all blocks; NaN
    is not a key") and backfills historical rows, but new full-network runs were
    still writing None, so the backfill was papering over a live source of NaN
    rather than a purely historical one. The Q1 batch-size controls wrote four
    such rows before this was caught.
    """
    return -1 if TRAINABLE_BLOCKS is None else TRAINABLE_BLOCKS


def assert_full_finetune(model) -> tuple[int, int]:
    """Refuse to call a run 'full fine-tuning' unless everything trains.

    Returns (total, trainable) parameter counts. Raises if any backbone tensor
    is frozen, because a run labelled full_finetune that silently froze the
    patch embedding would answer a different question from the one the paper
    says it answers, and nothing downstream would reveal it.
    """
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise RuntimeError(
            f"--full-finetune requested but {len(frozen)} parameter tensor(s) "
            f"are frozen: {frozen[:8]}{' ...' if len(frozen) > 8 else ''}. "
            "This is exactly the failure mode --trainable-blocks 24 has."
        )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total != trainable:
        raise RuntimeError(
            f"full fine-tuning: {trainable:,} of {total:,} parameters trainable")
    return total, trainable
# Set by --learning-rate. A 304 M ViT-L is not fine-tuned at a 7 M CNN's rate.
LEARNING_RATE: float = 3e-4


def _method_tag(method_name: str) -> str:
    """The method component of the experiment id.

    Resolution is appended only when it differs from the 224 px default, so
    every existing run keeps its id. Without this a 512 px run at batch 32 would
    produce the same id as the 224 px run at batch 32 and silently overwrite its
    checkpoint and predictions -- destroying a finished result with no error.
    """
    tag = f"{method_name}-b{BATCH_SIZE}"
    if IMAGE_SIZE != 224:
        tag += f"-r{IMAGE_SIZE}"
    # A non-default anneal is a different configuration, not a re-run of the
    # same one: it must not share an id with the run that used the published
    # default, or one would overwrite the other's checkpoint and predictions.
    if method_name == "irm" and IRM_ANNEAL_ITERS is not None:
        tag += f"-a{IRM_ANNEAL_ITERS}"
    # Partial fine-tuning and a non-default learning rate are different
    # configurations, not re-runs, and must not share an id with the full-network
    # run at 3e-4 -- one would overwrite the other's checkpoint and predictions.
    if FULL_FINETUNE:
        # Unambiguous and cannot collide with any tb* id.
        tag += "-ftfull"
    elif TRAINABLE_BLOCKS is not None:
        tag += f"-tb{TRAINABLE_BLOCKS}"
    if LEARNING_RATE != 3e-4:
        tag += f"-lr{LEARNING_RATE:g}"
    if DOMAIN_BALANCED:
        # Without this a domain-balanced run and an ordinary one share an id and
        # the second silently destroys the first's checkpoint and predictions --
        # and the pair only exists in order to be compared.
        tag += "-dbal"
    return tag


def run_one(target: str, method_name: str, seed: int) -> dict | None:
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
    root = project_root()
    outputs = root / "outputs"

    sources = [d for d in ALL_DOMAINS if d != target]
    manifest = load_manifest(outputs / "reports" / f"manifest_cached_{IMAGE_SIZE}.csv")
    experiment = build_experiment_split(
        manifest, protocol="lodo", sources=sources, target=target
    )

    experiment_id = make_experiment_id(
        protocol="lodo", sources=sources, target=target,
        backbone=BACKBONE, method=_method_tag(method_name), seed=seed,
    )
    print(f"\n{'=' * 78}\n=== {experiment_id}\n    {experiment.sizes()}\n{'=' * 78}", flush=True)
    for note in experiment.notes:
        print(f"    {note}", flush=True)

    augment = AugmentationConfig(image_size=IMAGE_SIZE)
    loader_config = LoaderConfig(
        batch_size=BATCH_SIZE, num_workers=2, seed=seed,
        domain_balanced_batches=DOMAIN_BALANCED,
    )
    loaders = build_loaders(
        experiment,
        loader_config=loader_config,
        train_transform=build_train_transform(augment),
        eval_transform=build_eval_transform(augment),
        preprocess=None,
        expected_size=IMAGE_SIZE,
    )

    # The probe script guards this; run_lodo did not. RETFound's CFP model was
    # pretrained on EyePACS, so an EyePACS target would report leakage as
    # generalization -- and nothing in the results row would show it. Raises
    # rather than warns, for the same reason it does there.
    if BACKBONE.startswith("retfound"):
        from src.models.retfound import assert_target_not_pretrained

        assert_target_not_pretrained(target)

    backbone = BackboneConfig(
        name=BACKBONE,
        image_size=resolve_input_size(BACKBONE, IMAGE_SIZE),
        # None means build_model never calls freeze_backbone at all, which is
        # what full fine-tuning requires. Passing 24 would freeze and then
        # partially unfreeze, leaving six tensors frozen.
        trainable_blocks=None if FULL_FINETUNE else TRAINABLE_BLOCKS,
    )
    class_counts = (
        experiment.train["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
    )
    method = MethodConfig(name=method_name)
    if IRM_ANNEAL_ITERS is not None:
        method.irm_anneal_iters = IRM_ANNEAL_ITERS
    built = build_method(method, backbone, class_counts=class_counts, device="cuda")
    if FULL_FINETUNE:
        assert_full_finetune(built.model)
    if GRADIENT_CHECKPOINTING:
        enable_gradient_checkpointing(built.model)
    total, trainable = count_parameters(built.model)
    print(f"    adaptation: {adaptation_mode()} | {trainable / 1e6:.1f}M of "
          f"{total / 1e6:.1f}M trainable "
          f"({100 * trainable / total:.1f}%) | "
          f"grad-checkpointing={GRADIENT_CHECKPOINTING}", flush=True)

    train_config = TrainConfig(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
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
            "sources": sources,
            "target": target,
        },
        feature_loss=built.feature_loss,
        batch_hook=built.batch_hook,
        objective_fn=built.objective_fn,
        to_probabilities=built.to_probabilities,
    )
    trainer.resume()

    started = time.perf_counter()
    trainer.fit(loaders["train"], loaders["val"])
    seconds = round(time.perf_counter() - started, 1)

    history = trainer.history_frame()
    history_path = outputs / "logs" / f"{experiment_id}_history.csv"
    history.to_csv(history_path, index=False)
    print(f"component statistics: {built.statistics()}", flush=True)

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
        "method": built.description,
        "component_statistics": built.statistics(),
        "temperature": evaluation["temperature"],
        "results": {k: {"metrics": v.metrics, "calibration": v.calibration,
                        "calibration_scaled": v.calibration_scaled}
                    for k, v in evaluation["results"].items()},
    })

    summary = trainer.checkpoints.summary()
    register_experiment(outputs / "experiment_registry.csv", {
        "experiment_id": experiment_id, "status": "COMPLETE", "protocol": "lodo",
        "source_domains": sources, "target_domain": target,
        "backbone": BACKBONE, "head": built.description.get("head"),
        "method": method_name, "loss": built.description.get("loss"),
        "imbalance_strategy": method.imbalance_strategy,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "domain_balanced": DOMAIN_BALANCED,
        "irm_anneal_iters": method.irm_anneal_iters,
        # These three were written to the summary row but not to the registry,
        # so 27 of the 33 runs whose experiment id says "tb4" carried NaN in the
        # registry's trainable_blocks column. The registry is the authoritative
        # record, so an id and its own record disagreed about what was trained.
        # adaptation_mode is the human-readable form and exists so a reader
        # never has to infer the protocol from a block count.
        "adaptation_mode": adaptation_mode(),
        "trainable_blocks": recorded_trainable_blocks(),
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "params_total_m": round(total / 1e6, 2),
        "params_trainable_m": round(trainable / 1e6, 2),
        "accumulation_steps": 1, "effective_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": 1e-4,
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
        "notes": f"Full leave-one-domain-out, batch {BATCH_SIZE}; "
                 "temperature fitted on source validation only",
    })

    row = {
        "target": target, "method": method_name, "seed": seed,
        "image_size": IMAGE_SIZE, "backbone": BACKBONE, "batch_size": BATCH_SIZE,
        "domain_balanced": DOMAIN_BALANCED,
        # The row's "method" column holds "irm", not the experiment tag, so a
        # non-default anneal is invisible here without this -- and the audit,
        # which rebuilds the experiment id from these columns, would look for
        # irm-b32 and find only the diverged run under that id.
        "irm_anneal_iters": method.irm_anneal_iters,
        # Same reason as irm_anneal_iters: the row's "method" column holds "erm",
        # so partial fine-tuning and a non-default rate are invisible here, and
        # the audit rebuilds erm-b16 for an erm-b16-tb4-lr0.0001 row.
        "trainable_blocks": recorded_trainable_blocks(),
        "learning_rate": LEARNING_RATE,
        "n_train": len(experiment.train), "n_test": len(experiment.test),
        "source_qwk": source_result.metrics["qwk"],
        "target_qwk": target_result.metrics["qwk"],
        "source_f1": source_result.metrics["f1_macro"],
        "target_f1": target_result.metrics["f1_macro"],
        "source_ece": source_result.calibration["ece"],
        "target_ece": target_result.calibration["ece"],
        "target_ece_scaled": target_result.calibration_scaled.get("ece"),
        "target_severe": target_result.metrics["severe_error_rate"],
        # The table has always had a temperature column and this row has never
        # filled it, so every run wrote NaN there and only rows touched by a
        # hand-rebuild held a value. A scaled ECE beside an empty temperature
        # reads as scaling that did not happen; it always had, and the number
        # was in the evaluation report the whole time.
        "temperature": (evaluation["temperature"] or {}).get("temperature"),
        "seconds": seconds,
    }
    print(f"RESULT {target}: {row}", flush=True)

    del eval_model, eval_built, trainer, built
    torch.cuda.empty_cache()
    return row


def main() -> None:
    import pandas as pd

    from src.utils.hardware import assert_cuda_ready
    from src.utils.io import project_root
    from src.utils.registry import merge_results_table

    assert_cuda_ready()

    arguments = sys.argv[1:]

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    method = _take("--method", "erm")
    global IRM_ANNEAL_ITERS
    _anneal = _take("--irm-anneal-iters", "")
    IRM_ANNEAL_ITERS = int(_anneal) if _anneal else None
    global TRAINABLE_BLOCKS, LEARNING_RATE, FULL_FINETUNE, GRADIENT_CHECKPOINTING
    _blocks = _take("--trainable-blocks", "")
    TRAINABLE_BLOCKS = int(_blocks) if _blocks else None
    LEARNING_RATE = float(_take("--learning-rate", str(LEARNING_RATE)))
    if "--full-finetune" in arguments:
        arguments.remove("--full-finetune")
        FULL_FINETUNE = True
        if TRAINABLE_BLOCKS is not None:
            raise SystemExit(
                "--full-finetune and --trainable-blocks are mutually exclusive: "
                "--trainable-blocks 24 is NOT full fine-tuning (it leaves the "
                "patch embedding, position embedding, cls token and final norm "
                "frozen). Pass only --full-finetune."
            )
    if "--gradient-checkpointing" in arguments:
        arguments.remove("--gradient-checkpointing")
        GRADIENT_CHECKPOINTING = True
    seeds = [int(s) for s in _take("--seeds", "42").split(",")]
    targets = _take("--targets", ",".join(ALL_DOMAINS)).split(",")

    global BACKBONE, BATCH_SIZE, IMAGE_SIZE
    BACKBONE = _take("--backbone", BACKBONE)
    # ConvNeXt-Tiny is 4x the parameters of DenseNet121. Batch 32 still fits in
    # 8 GB (measured 2.8 GB peak for DenseNet), but the batch size is exposed so
    # a larger backbone can be stepped down without editing the file.
    BATCH_SIZE = int(_take("--batch-size", str(BATCH_SIZE)))
    IMAGE_SIZE = int(_take("--image-size", str(IMAGE_SIZE)))

    global DOMAIN_BALANCED
    if "--domain-balanced" in arguments:
        arguments.remove("--domain-balanced")
        DOMAIN_BALANCED = True

    from src.models.backbones import SUPPORTED_BACKBONES

    if BACKBONE not in SUPPORTED_BACKBONES:
        raise SystemExit(
            f"unknown backbone {BACKBONE!r}; choose from {sorted(SUPPORTED_BACKBONES)}"
        )

    print(
        f"leave-one-domain-out: method={method}, seeds={seeds}, targets={targets}\n"
        f"  backbone={BACKBONE}, batch={BATCH_SIZE}, epochs={EPOCHS}"
        f"{', domain-balanced batches' if DOMAIN_BALANCED else ''}",
        flush=True,
    )

    # Resolution-scoped, for the same reason the experiment id is: every
    # analyse_* script reads this file and filters on method alone, so one
    # shared table would silently average two resolutions together.
    suffix = "" if IMAGE_SIZE == 224 else f"_r{IMAGE_SIZE}"
    path = project_root() / "outputs" / "tables" / f"lodo_results{suffix}.csv"

    def save(new_rows) -> "pd.DataFrame":
        frame = pd.DataFrame(new_rows)
        if path.exists():
            # backbone belongs in the key. It was once absent, and the
            # ConvNeXt-Tiny seed-42 run replaced the DenseNet121 seed-42 rows:
            # the three-seed means then mixed two architectures and reported
            # DDR's across-seed SD as 0.0105 instead of 0.0055, nearly doubling
            # a bar the two-bar criterion depends on.
            frame = merge_results_table(
                pd.read_csv(path), frame,
                # domain_balanced belongs here for the same reason backbone and
                # image_size do: the row's "method" column holds "erm", not the
                # tag, so an ERM run with domain-balanced batches and one without
                # are indistinguishable by the other four fields and the second
                # would overwrite the first -- destroying the control it exists
                # to be compared against.
                # irm_anneal_iters joins the key for the same reason: two IRM
                # runs differing only in the anneal are one row otherwise, and
                # the second silently replaces the first.
                # batch_size joins the key for the fourth instance of this same
                # bug. The Q1 controls train ERM at 224 px with batch 16 against
                # the existing batch 32 runs; with batch_size absent from the
                # key the two are one row, and the b16 runs silently replaced
                # the b32 rows for eyepacs 42/1/2 and idrid 42. The -dbal rows
                # survived only because domain_balanced happens to be a key.
                # Nothing was lost -- the registry is append-only and the
                # predictions and reports are filed under the full experiment
                # id, which carries the batch tag -- but the summary table is
                # what the paper's generators read, so it must not be lossy.
                ["target", "method", "seed", "backbone", "image_size",
                 "batch_size", "domain_balanced", "irm_anneal_iters",
                 "trainable_blocks", "learning_rate"],
            )
            frame = frame.sort_values(
                ["backbone", "seed", "target"]).reset_index(drop=True)
        frame.to_csv(path, index=False)
        return frame

    rows = []
    for seed in seeds:
        for target in targets:
            try:
                result = run_one(target, method, seed)
                if result is not None:
                    rows.append(result)
                    # Written after every target rather than once at the end.
                    # A four-target run is ~10 h at 512 px; a crash or a power
                    # cut on the last one used to leave three finished models
                    # with no summary row at all. The registry and predictions
                    # always survived, so nothing was ever lost -- but the
                    # table had to be rebuilt by hand, and until it was, this
                    # run looked like it had never happened.
                    save([result])
            except Exception as exc:  # noqa: BLE001 - one failure must not lose the rest
                import traceback

                print(f"!! target {target} seed {seed} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                traceback.print_exc()

                # A crashed run used to leave nothing behind at all, so the
                # configuration was simply absent from the registry and read as
                # "never attempted". That is the one thing this project must not
                # do: IRMv1 diverged to NaN on all three DDR seeds, which is a
                # result about IRM, and silence would have hidden it.
                #
                # status is not COMPLETE, so audit_consistency and every
                # analyse_* script skip these rows; they exist to record that
                # the run happened and what it did.
                from src.utils.registry import (
                    make_experiment_id, register_experiment)

                diverged = "nan" in str(exc).lower() or "infin" in str(exc).lower()
                register_experiment(
                    project_root() / "outputs" / "experiment_registry.csv",
                    {
                        "experiment_id": make_experiment_id(
                            protocol="lodo",
                            sources=[d for d in ALL_DOMAINS if d != target],
                            target=target, backbone=BACKBONE,
                            method=_method_tag(method), seed=seed,
                        ),
                        "status": "DIVERGED" if diverged else "FAILED",
                        "protocol": "lodo",
                        "source_domains": [d for d in ALL_DOMAINS if d != target],
                        "target_domain": target, "backbone": BACKBONE,
                        "method": method, "image_size": IMAGE_SIZE,
                        "batch_size": BATCH_SIZE, "seed": seed,
                        "domain_balanced": DOMAIN_BALANCED,
                        "notes": f"{type(exc).__name__}: {exc}",
                    },
                )

    if rows:
        # Read back rather than re-saving: every row is already on disk, and
        # merge_results_table would reject an empty frame for lacking the key
        # columns it is asked to merge on.
        pd.set_option("display.width", 220)
        print("\n" + "=" * 78)
        print(pd.read_csv(path).round(4).to_string(index=False))
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
