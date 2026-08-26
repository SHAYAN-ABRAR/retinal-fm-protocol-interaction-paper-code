"""The training loop: AMP, gradient accumulation, early stopping, resume.

Design notes specific to this project
-------------------------------------
**Mixed precision is on by default.** On 8 GB it is the difference between batch
16 and batch 6 at 224px.  The loss is computed in float32 (``logits.float()``)
even under autocast, because cross-entropy over float16 logits loses precision
exactly where it matters -- in the small probabilities that drive the calibration
metrics.

**Gradient accumulation preserves the effective batch size.** When VRAM forces a
smaller micro-batch, ``accumulation_steps`` keeps the optimisation identical to
the intended batch size rather than silently changing the experiment.  The
effective size is recorded, so a run at 4x4 is distinguishable from one at 16x1.

**OOM is caught and explained.** An OOM mid-epoch otherwise produces a wall of
CUDA text; :meth:`Trainer.fit` converts it into a message that names the batch
size, resolution and the concrete options.

**Validation drives everything.** Early stopping, checkpoint selection and the
scheduler all read source-domain validation metrics. The trainer never receives a
test loader at all -- evaluation on the target domain happens afterwards, in a
separate step, once the model is frozen.

**NaN/Inf is fatal, not silent.** A run that quietly produces NaN losses for
twenty epochs wastes an evening; the loop stops at the first one and says which
batch it was.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..evaluation.metrics import compute_all_metrics
from ..utils.logging import get_logger
from .checkpointing import CheckpointManager, load_checkpoint
from .early_stopping import EarlyStopping

log = get_logger("training.trainer")

__all__ = ["TrainConfig", "EpochRecord", "Trainer", "predict"]


def _is_interactive() -> bool:
    """True when a progress bar would be rendered rather than spooled to a file.

    Notebook / VS Code interactive kernels are treated as interactive (tqdm
    renders a widget there); a plain redirected stream is not.
    """
    import sys

    try:
        get_ipython  # type: ignore[name-defined]  # noqa: B018
        return True
    except NameError:
        pass
    stream = getattr(sys, "stderr", None)
    try:
        return bool(stream is not None and stream.isatty())
    except Exception:  # noqa: BLE001 - a closed or exotic stream
        return False


@dataclass
class TrainConfig:
    """Everything that defines a training run. Saved with the experiment."""

    epochs: int = 30
    batch_size: int = 16
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    head_learning_rate: float | None = None     # None = same as learning_rate
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    scheduler: str = "cosine"                   # 'cosine' | 'plateau' | 'none'
    min_learning_rate: float = 1e-6
    amp: bool = True
    grad_clip_norm: float | None = 1.0
    early_stopping_patience: int = 8
    monitor: str = "qwk"
    monitor_mode: str = "max"
    log_every_n_steps: int = 50
    num_classes: int = 5
    seed: int = 42
    # Per-batch progress bar. Auto-suppressed when output is not a terminal --
    # see _is_interactive() -- so background runs keep one line per epoch.
    show_progress: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps

    def describe(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effective_batch_size"] = self.effective_batch_size
        return payload


@dataclass
class EpochRecord:
    """One row of the training history; drives the training-curve figures."""

    epoch: int
    train_loss: float
    val_loss: float
    val_qwk: float
    val_f1_macro: float
    val_accuracy: float
    learning_rate: float
    grad_norm: float
    seconds: float
    peak_vram_gb: float
    # Mean auxiliary (domain-alignment) loss for the epoch; 0.0 for plain ERM.
    # Recorded so a DG run can be checked for an alignment term that collapsed
    # to zero -- which would make it ERM under a different name.
    auxiliary_loss: float = 0.0


def _build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with no weight decay on norms and biases.

    Decaying bias and normalisation parameters is a well-known small mistake; it
    costs a little accuracy and, more relevantly here, tends to make models
    slightly over-confident.
    """
    decay, no_decay, head = [], [], []
    head_names = ("classifier", "coral_bias")

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(head_names):
            head.append(parameter)
        elif parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    head_lr = config.head_learning_rate or config.learning_rate
    groups = [
        {"params": decay, "weight_decay": config.weight_decay, "lr": config.learning_rate},
        {"params": no_decay, "weight_decay": 0.0, "lr": config.learning_rate},
        {"params": head, "weight_decay": config.weight_decay, "lr": head_lr},
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig, steps_per_epoch: int):
    if config.scheduler == "none":
        return None
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.monitor_mode,
            factor=0.5,
            patience=max(1, config.early_stopping_patience // 3),
            min_lr=config.min_learning_rate,
        )
    if config.scheduler == "cosine":
        # Per-step cosine with linear warmup. Warmup matters when fine-tuning a
        # pretrained backbone: a cold high learning rate erases the features
        # that made pretraining worth using.
        warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)
        total_steps = max(warmup_steps + 1, config.epochs * steps_per_epoch)

        def _factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            floor = config.min_learning_rate / max(config.learning_rate, 1e-12)
            return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _factor)
    raise ValueError(f"unknown scheduler {config.scheduler!r}")


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: Any,
    *,
    device: str | torch.device = "cuda",
    amp: bool = True,
    num_classes: int = 5,
    loss_fn: nn.Module | None = None,
    to_probabilities: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Run inference and return labels, predictions, probabilities and indices.

    Everything is moved to CPU as it is produced -- accumulating 35k x 5 logits on
    an 8 GB card alongside the model is avoidable waste.

    ``to_probabilities`` lets an ordinal head convert its own output; the default
    is softmax.
    """
    model.eval()
    model.to(device)

    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_domains: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        images, targets, domains, indices = batch
        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device != "cpu"):
            logits = model(images)

        logits = logits.float()
        if loss_fn is not None:
            total_loss += float(loss_fn(logits, targets_device).item())
            n_batches += 1

        probabilities = (
            to_probabilities(logits) if to_probabilities is not None
            else torch.softmax(logits, dim=-1)
        )
        all_probabilities.append(probabilities.cpu().numpy())
        all_targets.append(targets.numpy())
        all_domains.append(domains.numpy())
        all_indices.append(indices.numpy())

    probabilities = np.concatenate(all_probabilities) if all_probabilities else np.empty((0, num_classes))
    result = {
        "probabilities": probabilities,
        "y_true": np.concatenate(all_targets) if all_targets else np.empty(0, dtype=int),
        "y_pred": probabilities.argmax(axis=1) if len(probabilities) else np.empty(0, dtype=int),
        "domain_id": np.concatenate(all_domains) if all_domains else np.empty(0, dtype=int),
        "index": np.concatenate(all_indices) if all_indices else np.empty(0, dtype=int),
        "confidence": probabilities.max(axis=1) if len(probabilities) else np.empty(0),
    }
    if loss_fn is not None and n_batches:
        result["loss"] = total_loss / n_batches
    return result


class Trainer:
    """ERM training loop. Phase-4 methods subclass or wrap it."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainConfig,
        *,
        device: str | torch.device = "cuda",
        checkpoint_dir: Path | str = "outputs/checkpoints",
        experiment_id: str = "unnamed",
        extra_config: dict[str, Any] | None = None,
        feature_loss: nn.Module | None = None,
        batch_hook: Callable[[nn.Module, torch.Tensor], Any] | None = None,
        to_probabilities: Callable[[torch.Tensor], torch.Tensor] | None = None,
        objective_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Hooks let the Phase-4 methods reuse this loop instead of forking it.

        feature_loss:
            Called as ``feature_loss(features, domain_ids)`` and ADDED to the task
            loss. This is where Deep CORAL plugs in. When set, the model is asked
            for its pooled features, so it must accept ``return_features=True``.
        objective_fn:
            Called as ``objective_fn(logits, targets, domain_ids)`` and REPLACES
            the task loss entirely. GroupDRO and IRM need this: one reweights the
            per-domain means and the other adds a penalty computed from logits and
            targets, and neither can be expressed as a term added to an already-
            reduced scalar. Validation loss still uses ``loss_fn``, so model
            selection stays on the same criterion for every method.
        batch_hook:
            Called as ``batch_hook(model, domain_ids)`` before each forward pass.
            MixStyle uses it to learn which samples came from which domain.
        to_probabilities:
            Converts raw head output to a class distribution during evaluation.
            The ordinal CORAL head needs this because its output is K-1
            cumulative logits, not K class logits; the default is softmax.
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.experiment_id = experiment_id
        self.extra_config = extra_config or {}
        self.feature_loss = feature_loss.to(device) if feature_loss is not None else None
        self.batch_hook = batch_hook
        self.objective_fn = (objective_fn.to(device)
                             if isinstance(objective_fn, nn.Module) else objective_fn)
        self.to_probabilities = to_probabilities

        self.checkpoints = CheckpointManager(
            directory=checkpoint_dir,
            experiment_id=experiment_id,
            monitor=config.monitor,
            mode=config.monitor_mode,
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=config.monitor_mode,
            monitor=config.monitor,
        )
        self.history: list[EpochRecord] = []
        self.start_epoch = 0
        self.global_step = 0
        self.last_auxiliary_loss = 0.0

        self.optimizer = _build_optimizer(self.model, config)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and str(device) != "cpu")
        self.scheduler = None       # built in fit(), needs steps_per_epoch

    # -- resume -----------------------------------------------------------

    def resume(self, path: Path | str | None = None) -> bool:
        """Resume from a checkpoint. Returns False if there is nothing to resume."""
        path = Path(path) if path else self.checkpoints.last_path
        if not path.exists():
            log.info("no checkpoint at %s; starting from scratch", path)
            return False

        payload = load_checkpoint(
            path, model=self.model, optimizer=self.optimizer, scaler=self.scaler,
            map_location=self.device, restore_rng=True,
        )
        self.start_epoch = int(payload.get("epoch", -1)) + 1
        self.global_step = int(payload.get("global_step", 0))
        # Restore what "best" meant before the interruption, or the first epoch
        # after the resume overwrites the best checkpoint with a worse model and
        # the early-stopping counter starts again from zero.
        if self.checkpoints.restore_best_state():
            log.info("restored best val %s: %.4f at epoch %d",
                     self.checkpoints.monitor, self.checkpoints.best_value,
                     self.checkpoints.best_epoch)
        log.info("resuming %s at epoch %d", self.experiment_id, self.start_epoch)
        return True

    # -- one epoch --------------------------------------------------------

    def _train_epoch(self, loader: Any, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss, n_batches = 0.0, 0
        grad_norms: list[float] = []

        self.optimizer.zero_grad(set_to_none=True)
        accumulation = max(1, self.config.accumulation_steps)

        # tqdm only when attached to a terminal. Redirected to a file (background
        # run, nohup, CI) it emits a line per update -- 719 batches becomes
        # thousands of unreadable lines in the log, which is exactly what the
        # one-line-per-epoch summary is meant to replace.
        progress = None
        if self.config.show_progress and _is_interactive():
            try:
                from tqdm.auto import tqdm

                progress = tqdm(
                    total=len(loader), desc=f"epoch {epoch:>3d}", unit="batch",
                    leave=False, mininterval=0.5,
                )
            except (ImportError, TypeError):
                pass

        running_auxiliary = 0.0

        for step, (images, targets, domains, _indices) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            domains = domains.to(self.device, non_blocking=True)

            # MixStyle needs the batch's domain labels before the forward pass so
            # it mixes style ACROSS domains rather than within one.
            if self.batch_hook is not None:
                self.batch_hook(self.model, domains)

            need_features = self.feature_loss is not None
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.scaler.is_enabled()):
                if need_features:
                    logits, features = self.model(images, return_features=True)
                else:
                    logits, features = self.model(images), None

            # Loss in float32 even under autocast: fp16 log-softmax loses the
            # precision that the calibration metrics depend on.
            if self.objective_fn is not None:
                # GroupDRO / IRM: the method owns the reduction, so it is handed
                # the domain labels and returns the training loss outright.
                loss = self.objective_fn(logits.float(), targets, domains)
            else:
                loss = self.loss_fn(logits.float(), targets)
            task_loss_value = float(loss.item())

            if need_features:
                # Deep CORAL: align source-domain feature covariances. Computed in
                # float32 -- covariance of fp16 features is numerically poor.
                auxiliary = self.feature_loss(features.float(), domains)
                running_auxiliary += float(auxiliary.item())
                loss = loss + auxiliary

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss ({loss.item()}) at epoch {epoch}, step {step} "
                    f"(task {task_loss_value}). Training stopped rather than "
                    "continuing with a corrupt model. Usual causes: learning rate "
                    "too high, an auxiliary loss weight that is far too large, or a "
                    "corrupt image batch."
                )

            self.scaler.scale(loss / accumulation).backward()
            running_loss += float(loss.item())
            n_batches += 1

            if (step + 1) % accumulation == 0 or (step + 1) == len(loader):
                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                    grad_norms.append(float(norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR):
                    self.scheduler.step()

            if progress is not None:
                progress.update(1)
                if step % 10 == 0:
                    progress.set_postfix(loss=f"{running_loss / max(1, n_batches):.4f}")

        if progress is not None:
            progress.close()

        mean_loss = running_loss / max(1, n_batches)
        mean_grad = float(np.mean(grad_norms)) if grad_norms else float("nan")
        self.last_auxiliary_loss = running_auxiliary / max(1, n_batches)
        return mean_loss, mean_grad

    # -- validation -------------------------------------------------------

    def evaluate(self, loader: Any) -> dict[str, Any]:
        """Validation metrics. Source-domain data only -- see module docstring."""
        outputs = predict(
            self.model, loader, device=self.device, amp=self.scaler.is_enabled(),
            num_classes=self.config.num_classes, loss_fn=self.loss_fn,
            to_probabilities=self.to_probabilities,
        )
        metrics = compute_all_metrics(
            outputs["y_true"], outputs["y_pred"], outputs["probabilities"],
            num_classes=self.config.num_classes,
            include_per_class=False, include_referable=False,
        )
        metrics["loss"] = outputs.get("loss", float("nan"))
        return metrics

    # -- fit --------------------------------------------------------------

    def fit(self, train_loader: Any, val_loader: Any) -> list[EpochRecord]:
        """Train with early stopping and checkpointing. Returns the history."""
        steps_per_epoch = max(1, len(train_loader) // max(1, self.config.accumulation_steps))
        self.scheduler = _build_scheduler(self.optimizer, self.config, steps_per_epoch)

        full_config = {**self.config.describe(), **self.extra_config}
        log.info(
            "training %s: %d epochs, batch %d x %d accum = %d effective, lr %.2e, amp=%s",
            self.experiment_id, self.config.epochs, self.config.batch_size,
            self.config.accumulation_steps, self.config.effective_batch_size,
            self.config.learning_rate, self.scaler.is_enabled(),
        )

        for epoch in range(self.start_epoch, self.config.epochs):
            started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            try:
                train_loss, grad_norm = self._train_epoch(train_loader, epoch)
            except torch.cuda.OutOfMemoryError as exc:
                self._explain_oom(exc)
                raise

            val_metrics = self.evaluate(val_loader)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics[self.config.monitor])

            peak = (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            )
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=float(val_metrics.get("loss", float("nan"))),
                val_qwk=float(val_metrics.get("qwk", float("nan"))),
                val_f1_macro=float(val_metrics.get("f1_macro", float("nan"))),
                val_accuracy=float(val_metrics.get("accuracy", float("nan"))),
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                grad_norm=grad_norm,
                seconds=round(time.perf_counter() - started, 1),
                peak_vram_gb=round(peak, 2),
                auxiliary_loss=round(self.last_auxiliary_loss, 6),
            )
            self.history.append(record)

            log.info(
                "epoch %3d | train %.4f%s | val %.4f | QWK %.4f | F1 %.4f | acc %.4f "
                "| lr %.2e | %.0fs | %.2f GB",
                epoch, record.train_loss,
                f" (aux {record.auxiliary_loss:.4f})" if self.feature_loss is not None else "",
                record.val_loss, record.val_qwk, record.val_f1_macro,
                record.val_accuracy, record.learning_rate,
                record.seconds, record.peak_vram_gb,
            )

            self.checkpoints.update(
                epoch=epoch, val_metrics=val_metrics, model=self.model,
                optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler,
                global_step=self.global_step, config=full_config,
            )

            if self.early_stopping.step(val_metrics[self.config.monitor], epoch):
                break

        return self.history

    def _explain_oom(self, exc: Exception) -> None:
        config = self.config
        log.error(
            "CUDA out of memory.\n"
            "  current setting : batch %d x %d accumulation (effective %d)\n"
            "  options, in order of preference:\n"
            "    1. halve batch_size and double accumulation_steps -- the effective\n"
            "       batch size is preserved, so the experiment is unchanged\n"
            "    2. lower the image size (384 -> 224)\n"
            "    3. freeze more of the backbone (BackboneConfig.trainable_blocks)\n"
            "  Record whichever you choose: batch size is part of the experiment.\n"
            "  original error: %s",
            config.batch_size, config.accumulation_steps, config.effective_batch_size,
            str(exc).splitlines()[0],
        )

    def history_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame([asdict(record) for record in self.history])
