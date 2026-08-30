"""RETFound linear probe on the leave-one-domain-out matrix.

Answers the question every reviewer will ask of this project's negative results:
*is this a small-model artefact?* DenseNet121 is 7 M parameters, ConvNeXt-Tiny
28 M; RETFound is a 304 M ViT-L/16 pretrained with a masked autoencoder on ~1.6 M
retinal images. If a retinal foundation model also fails to close the
calibration gap, "the problem is the domain shift, not the network" stops being
a claim about two small CNNs.

⚠ EyePACS is excluded, and not by convention
--------------------------------------------
RETFound's colour-fundus model was pretrained on EyePACS. Using EyePACS as a
held-out target would report leakage as generalization.
``assert_target_not_pretrained`` raises on it. EyePACS stays in the *source*
pool for the other targets, which is correct -- sources are seen by definition.

Why a linear probe, and why that is not a shortcut
--------------------------------------------------
The backbone is frozen and a linear head is trained on cached features. This is
the standard protocol for evaluating a foundation model's representation --
and it is used in recent retinal foundation-model studies -- and on 8 GB it is
the difference between one
hour and thirty. Full fine-tuning of a 304 M ViT needs ~4.9 GB of optimiser
state before a single activation, which does not leave room for a useful batch.

The limitation is real and belongs in the paper: a linear probe measures the
*representation*, not what the model could reach with fine-tuning. What it can
support is "RETFound's features do not separate this problem", which is the
question being asked.

Everything downstream is unchanged
----------------------------------
Features are wrapped in a dataset that yields ``(features, grade, domain_id,
index)`` and the head is a model that consumes them, so the existing Trainer,
``evaluate_experiment``, temperature scaling, predictions CSV, registry row and
every analyse_* script work without modification. The probe is a different
training loop, not a different protocol.

Usage:
    python run_retfound_probe.py --extract          # one frozen pass, ~15 min
    python run_retfound_probe.py                    # probes: ddr, aptos, idrid
    python run_retfound_probe.py --targets ddr --seeds 42

    # the matched ImageNet control, same pipeline, only the features differ
    python run_retfound_probe.py --backbone densenet121 --extract
    python run_retfound_probe.py --backbone densenet121
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

# The backbone is module-level state because run_one() and extract_features()
# both need it and the whole point of this script is that everything except the
# feature extractor is identical between them. Set once by main().
BACKBONE = "retfound_cfp"
IMAGE_SIZE = 224
BATCH_SIZE = 256          # features only; the backbone is not in the graph
EPOCHS = 200
LEARNING_RATE = 5e-3
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]
# EyePACS is absent deliberately -- see the module docstring.
DEFAULT_TARGETS = ["ddr", "aptos", "idrid"]
SEEDS = [42, 1, 2]
FEATURE_FILE = "retfound_cfp_features_224.npz"


def _use_backbone(name: str) -> None:
    """Point the script at a backbone, features file included.

    ``densenet121`` is the matched ImageNet control for the RETFound probe. It
    is the same 1024-dimensional feature width, run through the same
    standardisation, the same head, the same schedule and the same splits, so
    the only thing that differs between the two sets of numbers is what
    produced the features. Without it a RETFound probe can only be compared
    against a fine-tuned network, which confounds the backbone with the
    training protocol and cannot support a claim either way.
    """
    global BACKBONE, FEATURE_FILE
    BACKBONE = name
    FEATURE_FILE = f"{name}_features_{IMAGE_SIZE}.npz"


def _is_retfound() -> bool:
    return BACKBONE.startswith("retfound")


def extract_features(checkpoint: str | None = None) -> None:
    """One frozen forward pass over every cached image."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.data.augmentations import AugmentationConfig, build_eval_transform
    from src.data.unified_dataset import RetinaDataset, load_manifest
    from src.models.backbones import BackboneConfig, build_model
    from src.utils.hardware import assert_cuda_ready
    from src.utils.io import project_root

    assert_cuda_ready()
    root = project_root()
    outputs = root / "outputs"
    destination = outputs / "features" / FEATURE_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(outputs / "reports" / f"manifest_cached_{IMAGE_SIZE}.csv")
    print(f"extracting {BACKBONE} features for {len(manifest):,} images", flush=True)

    config = BackboneConfig(name=BACKBONE, image_size=IMAGE_SIZE, pretrained=True)
    if checkpoint and _is_retfound():
        config.extra["checkpoint"] = checkpoint
    model = build_model(config).to("cuda").eval()
    # Only the RETFound loader writes a report; for a timm backbone the
    # provenance is "timm ImageNet weights" and there is nothing to verify.
    report = config.extra.get("retfound", {"weights": f"timm imagenet ({BACKBONE})"})
    print(f"weights: {report}", flush=True)

    transform = build_eval_transform(AugmentationConfig(image_size=IMAGE_SIZE))
    dataset = RetinaDataset(manifest, transform=transform, preprocess=None,
                            expected_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2,
                        persistent_workers=True)

    features, started = [], time.perf_counter()
    with torch.inference_mode():
        for step, (images, _grades, _domains, _indices) in enumerate(loader):
            images = images.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                batch = model.forward_features(images)
            features.append(batch.float().cpu().numpy())
            if step % 50 == 0:
                done = min((step + 1) * 64, len(dataset))
                rate = done / max(1e-9, time.perf_counter() - started)
                print(f"  {done:,}/{len(dataset):,}  {rate:.0f} img/s", flush=True)

    stacked = np.concatenate(features).astype(np.float32)
    assert len(stacked) == len(manifest), (
        f"extracted {len(stacked)} features for {len(manifest)} manifest rows")
    np.savez_compressed(
        destination, features=stacked,
        image_id=manifest["image_id"].to_numpy().astype(str),
        report=str(report))
    minutes = (time.perf_counter() - started) / 60
    print(f"saved {stacked.shape} -> {destination}  ({minutes:.1f} min)", flush=True)


class _FeatureDataset:
    """Yields ``(features, grade, domain_id, index)``, like RetinaDataset."""

    def __init__(self, features, frame) -> None:
        import torch

        self.features = torch.from_numpy(features)
        self.grades = torch.as_tensor(frame["grade"].astype(int).to_numpy())
        self.domains = torch.as_tensor(frame["domain_id"].astype(int).to_numpy())

    def __len__(self) -> int:
        return len(self.grades)

    def __getitem__(self, index):
        return self.features[index], self.grades[index], self.domains[index], index


def run_one(target: str, seed: int) -> dict | None:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.data.splits import build_experiment_split
    from src.data.unified_dataset import load_manifest
    from src.evaluation.evaluate import evaluate_experiment
    from src.models.retfound import assert_target_not_pretrained
    from src.training.trainer import TrainConfig, Trainer
    from src.utils.io import project_root, write_json
    from src.utils.registry import make_experiment_id, register_experiment
    from src.utils.seed import set_global_seed

    # The pretraining-overlap guard describes RETFound's corpus, not ImageNet's.
    # Applying it to the DenseNet control would refuse EyePACS for a model that
    # never saw it; skipping it for RETFound would report leakage as
    # generalization. So it is asked exactly when it applies.
    if _is_retfound():
        assert_target_not_pretrained(target)
    set_global_seed(seed)
    root = project_root()
    outputs = root / "outputs"

    stored = np.load(outputs / "features" / FEATURE_FILE, allow_pickle=True)
    lookup = {image_id: i for i, image_id in enumerate(stored["image_id"])}
    features = stored["features"]

    sources = [d for d in ALL_DOMAINS if d != target]
    manifest = load_manifest(outputs / "reports" / f"manifest_cached_{IMAGE_SIZE}.csv")
    experiment = build_experiment_split(
        manifest, protocol="lodo", sources=sources, target=target)

    experiment_id = make_experiment_id(
        protocol="lodo", sources=sources, target=target,
        backbone=BACKBONE, method=f"linprobe-b{BATCH_SIZE}", seed=seed)
    print(f"\n{'=' * 78}\n=== {experiment_id}\n    {experiment.sizes()}\n{'=' * 78}",
          flush=True)

    # Per-dimension standardisation, fitted on the SOURCE TRAINING split only.
    #
    # This is not cosmetic. The probe originally fed raw features through an
    # nn.LayerNorm, which normalises each sample across its 1024 dimensions and
    # therefore discards the relative scale between dimensions that a linear
    # head depends on. Measured on the DDR target, that alone cost 0.37 QWK
    # (0.064 with LayerNorm against 0.437 standardised), and the head never
    # predicted grades 1 or 3 at all. The published number would have described
    # the normalisation, not RETFound.
    #
    # The statistics come from the training split and are applied unchanged to
    # validation and target test. Fitting them on anything that includes the
    # target would leak the target's feature distribution into the model.
    train_indices = np.array([lookup[i] for i in experiment.train["image_id"]],
                             dtype=np.int64)
    mean = features[train_indices].mean(axis=0, keepdims=True)
    std = features[train_indices].std(axis=0, keepdims=True)
    # A dimension that never varies in training carries no information, but
    # dividing by ~0 would amplify it a millionfold if it happens to fire on the
    # target -- turning a dead unit into the loudest input the head sees.
    # DenseNet121 has 3 such dimensions of 1024. Pass them through unscaled.
    std = np.where(std < 1e-6, 1.0, std)

    def subset(frame):
        indices = np.array([lookup[i] for i in frame["image_id"]], dtype=np.int64)
        return _FeatureDataset((features[indices] - mean) / std, frame)

    loaders = {
        name: DataLoader(subset(frame), batch_size=BATCH_SIZE,
                         shuffle=(name == "train"), num_workers=0)
        for name, frame in (("train", experiment.train), ("val", experiment.val),
                            ("test", experiment.test))
    }

    class LinearProbe(nn.Module):
        """A head over frozen features, shaped like DRModel for the evaluator."""

        def __init__(self, dim: int, num_classes: int = 5) -> None:
            super().__init__()
            # No LayerNorm: the features arrive already standardised per
            # dimension using training-split statistics (see subset() above).
            # A per-sample normalisation on top of that would undo it.
            self.classifier = nn.Linear(dim, num_classes)

        def forward_features(self, x):
            return x

        def forward(self, x, return_features: bool = False):
            pooled = self.forward_features(x)
            logits = self.classifier(pooled)
            return (logits, pooled) if return_features else logits

    model = LinearProbe(features.shape[1])
    train_config = TrainConfig(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        weight_decay=1e-4, warmup_epochs=1, scheduler="cosine", amp=False,
        # Patience scales with the 200-epoch schedule. At the original 8 the
        # run stopped around epoch 21, well short of the linear optimum: the
        # probe was reporting how far AdamW had got, not what the features
        # support. 200 epochs at lr 5e-3 reaches source-validation QWK 0.588
        # against the exact L-BFGS optimum's 0.596 on the DDR target.
        #
        # That setting was chosen by convergence on SOURCE validation against
        # the L-BFGS reference. No target-test number took part in the choice.
        grad_clip_norm=1.0, early_stopping_patience=50, monitor="qwk", seed=seed,
    )
    trainer = Trainer(
        model, nn.CrossEntropyLoss(), train_config, device="cuda",
        checkpoint_dir=outputs / "checkpoints", experiment_id=experiment_id,
        extra_config={"backbone": BACKBONE, "protocol": "linear_probe",
                      "frozen": True, "sources": sources, "target": target},
    )

    started = time.perf_counter()
    trainer.fit(loaders["train"], loaders["val"])
    seconds = round(time.perf_counter() - started, 1)

    from src.training.checkpointing import load_checkpoint

    evaluation_model = LinearProbe(features.shape[1]).to("cuda")
    load_checkpoint(trainer.checkpoints.best_path, model=evaluation_model,
                    map_location="cuda")
    evaluation = evaluate_experiment(
        evaluation_model, loaders, experiment, experiment_id=experiment_id,
        device="cuda", amp=False, predictions_dir=outputs / "predictions")

    source_result = evaluation["results"]["source_val"]
    target_result = evaluation["results"]["target_test"]
    write_json(outputs / "reports" / f"{experiment_id}_evaluation.json", {
        "experiment_id": experiment_id,
        "method": {"method": "linprobe", "backbone": BACKBONE, "frozen": True},
        "component_statistics": {"features": str(stored["report"])},
        "temperature": evaluation["temperature"],
        "results": {k: {"metrics": v.metrics, "calibration": v.calibration,
                        "calibration_scaled": v.calibration_scaled}
                    for k, v in evaluation["results"].items()},
    })

    register_experiment(outputs / "experiment_registry.csv", {
        "experiment_id": experiment_id, "status": "COMPLETE", "protocol": "lodo",
        "source_domains": sources, "target_domain": target,
        "backbone": BACKBONE, "head": "linear", "method": "linprobe",
        "loss": "cross_entropy", "imbalance_strategy": "none",
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "domain_balanced": False, "accumulation_steps": 1,
        "effective_batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "weight_decay": 1e-4, "epochs_planned": EPOCHS,
        "epochs_run": len(trainer.history), "seed": seed,
        "n_train": len(experiment.train), "n_val": len(experiment.val),
        "n_test": len(experiment.test),
        "test_qwk": target_result.metrics["qwk"],
        "test_f1_macro": target_result.metrics["f1_macro"],
        "test_severe_error_rate": target_result.metrics["severe_error_rate"],
        "test_ece": target_result.calibration["ece"],
        "train_seconds": seconds,
        "predictions_path": target_result.predictions_path,
        "notes": ("RETFound frozen linear probe. EyePACS is in RETFound's "
                  "pretraining corpus and is excluded as a target."
                  if _is_retfound() else
                  f"{BACKBONE} ImageNet frozen linear probe: the matched "
                  "control for the RETFound probe. Identical standardisation, "
                  "head, schedule and splits; only the features differ."),
    })

    return {
        "target": target, "method": "linprobe", "seed": seed,
        "image_size": IMAGE_SIZE, "backbone": BACKBONE, "batch_size": BATCH_SIZE,
        "domain_balanced": False,
        "n_train": len(experiment.train), "n_test": len(experiment.test),
        "source_qwk": source_result.metrics["qwk"],
        "target_qwk": target_result.metrics["qwk"],
        "source_f1": source_result.metrics["f1_macro"],
        "target_f1": target_result.metrics["f1_macro"],
        "source_ece": source_result.calibration["ece"],
        "target_ece": target_result.calibration["ece"],
        "target_ece_scaled": target_result.calibration_scaled.get("ece"),
        "target_severe": target_result.metrics["severe_error_rate"],
        "temperature": (evaluation["temperature"] or {}).get("temperature"),
        "seconds": seconds,
    }


def main() -> None:
    import pandas as pd

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

    # Must come first: it sets the features filename every later line reads.
    _use_backbone(_take("--backbone", BACKBONE))
    checkpoint = _take("--checkpoint", "") or None
    targets = _take("--targets", ",".join(DEFAULT_TARGETS)).split(",")
    seeds = [int(s) for s in _take("--seeds", ",".join(map(str, SEEDS))).split(",")]

    if "--extract" in arguments:
        extract_features(checkpoint)
        return

    outputs = project_root() / "outputs"
    if not (outputs / "features" / FEATURE_FILE).exists():
        raise SystemExit(
            f"no features at outputs/features/{FEATURE_FILE}.\n"
            "Run:  python run_retfound_probe.py --extract")

    # Named for the protocol, not the backbone: the table holds RETFound and
    # its matched ImageNet control, keyed by backbone.
    path = outputs / "tables" / "linear_probe_results.csv"
    rows = []
    for seed in seeds:
        for target in targets:
            try:
                result = run_one(target, seed)
            except Exception as error:  # noqa: BLE001 - one failure must not lose the rest
                import traceback

                print(f"!! {target} seed {seed} FAILED: {type(error).__name__}: {error}",
                      flush=True)
                traceback.print_exc()
                continue
            rows.append(result)
            frame = pd.DataFrame([result])
            if path.exists():
                frame = merge_results_table(
                    pd.read_csv(path), frame,
                    ["target", "method", "seed", "backbone", "image_size",
                     "domain_balanced"])
            frame.to_csv(path, index=False)

    if rows:
        pd.set_option("display.width", 220)
        print("\n" + "=" * 78)
        print(pd.read_csv(path).round(4).to_string(index=False))
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
