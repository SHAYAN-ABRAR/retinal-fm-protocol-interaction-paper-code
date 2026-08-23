"""End-to-end evaluation of a trained model on one or more domains.

Produces, for every evaluated split:

* the full metric suite (discrimination, ordinal error, per-class, AUROC),
* calibration metrics before and after temperature scaling,
* a **sample-level prediction table** with the columns the brief asks for --
  image id, dataset, true and predicted grade, all five class probabilities,
  confidence, correctness and absolute grade error.

That table is the substrate for the error-analysis galleries and the bootstrap
confidence intervals, so it is saved for every evaluation rather than recomputed.

Protocol enforcement
--------------------
:func:`evaluate_experiment` fits temperature on the **source validation** split
and then applies that fixed scalar to the target domain.  It never re-fits on the
target: doing so would tune a parameter on the test set. The split the
temperature was fitted on travels with the result, so the artefact itself shows
the protocol was respected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .calibration import TemperatureScaler, calibration_metrics, fit_temperature
from .metrics import compute_all_metrics

log = get_logger("evaluation.evaluate")

__all__ = ["EvaluationResult", "predict_with_logits", "evaluate_split", "evaluate_experiment"]


@dataclass
class EvaluationResult:
    """Metrics and predictions for one model on one split."""

    split_name: str
    domain: str
    n_samples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    calibration_scaled: dict[str, float] = field(default_factory=dict)
    temperature: dict[str, Any] = field(default_factory=dict)
    predictions_path: str | None = None

    def headline(self) -> dict[str, Any]:
        """The few numbers that belong in a summary table."""
        return {
            "split": self.split_name,
            "domain": self.domain,
            "n": self.n_samples,
            "qwk": self.metrics.get("qwk"),
            "f1_macro": self.metrics.get("f1_macro"),
            "balanced_accuracy": self.metrics.get("balanced_accuracy"),
            "accuracy": self.metrics.get("accuracy"),
            "mae_grade": self.metrics.get("mae_grade"),
            "within_1_grade": self.metrics.get("within_1_grade"),
            "severe_error_rate": self.metrics.get("severe_error_rate"),
            "auroc_macro": self.metrics.get("auroc_macro"),
            "ece": self.calibration.get("ece"),
            "ece_scaled": self.calibration_scaled.get("ece"),
            "nll": self.calibration.get("nll"),
            "nll_scaled": self.calibration_scaled.get("nll"),
            "brier": self.calibration.get("brier"),
        }


def predict_with_logits(
    model: Any,
    loader: Any,
    *,
    device: str = "cuda",
    amp: bool = True,
    num_classes: int = 5,
    to_probabilities: Any | None = None,
    predict_fn: Any | None = None,
) -> dict[str, np.ndarray]:
    """Inference returning **raw logits** as well as probabilities.

    Logits are needed because temperature scaling operates on them; recovering
    them from probabilities would require an inverse softmax and lose the scale.

    ``to_probabilities`` converts the head's raw output into a distribution over
    the K grades. It is **required** for the ordinal CORAL head, whose output is
    K-1 cumulative logits: softmaxing those would produce a K-1 vector and every
    downstream metric would fail (or worse, silently mis-shape). The default is
    softmax, which is correct only for a K-way linear head.

    ``predict_fn`` supplies the head's own decision rule, and matters more than
    it looks. For CORAL the native rule -- count the thresholds whose sigmoid
    exceeds 0.5 -- is NOT the argmax of the differenced distribution: measured on
    synthetic ordinal data the two disagree on ~29% of samples, and the native
    rule scored materially higher (QWK 0.90 vs 0.82).

    Decisively, the native rule is **exactly invariant** to temperature scaling
    (sigmoid(x/T) > 0.5 iff x > 0 for any T > 0), whereas the argmax of the
    differenced distribution changed on 23% of samples under a fitted
    temperature. Using argmax would let a *calibration* step silently change
    accuracy and QWK, destroying the property that makes temperature scaling a
    clean, isolated intervention in this study.

    Confidence is therefore reported as the probability of the **predicted**
    class rather than the maximum probability, so prediction and confidence stay
    consistent when the two rules disagree.
    """
    import torch

    model.eval()
    model.to(device)

    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    domain_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for images, targets, domains, indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device != "cpu"):
                outputs = model(images)
            # Cast to float32 before leaving the GPU: fp16 logits lose precision
            # in exactly the small-probability regime the calibration metrics use.
            logits_chunks.append(outputs.float().cpu().numpy())
            target_chunks.append(targets.numpy())
            domain_chunks.append(domains.numpy())
            index_chunks.append(indices.numpy())

    logits = (
        np.concatenate(logits_chunks) if logits_chunks else np.empty((0, num_classes), np.float32)
    )
    if to_probabilities is not None:
        probabilities = to_probabilities(torch.from_numpy(logits)).numpy()
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

    if probabilities.shape[1] != num_classes:
        raise ValueError(
            f"probabilities have {probabilities.shape[1]} columns but the task has "
            f"{num_classes} classes. An ordinal (CORAL) head emits K-1 cumulative "
            "logits and needs to_probabilities=coral_probabilities; pass the "
            "converter from build_method(...).to_probabilities."
        )

    if len(probabilities):
        if predict_fn is not None:
            predictions = predict_fn(torch.from_numpy(logits)).numpy().astype(int)
        else:
            predictions = probabilities.argmax(axis=1)
        confidence = probabilities[np.arange(len(predictions)), predictions]
    else:
        predictions = np.empty(0, int)
        confidence = np.empty(0, float)

    return {
        "logits": logits,
        "probabilities": probabilities,
        "y_true": np.concatenate(target_chunks) if target_chunks else np.empty(0, int),
        "y_pred": predictions,
        "confidence": confidence,
        "domain_id": np.concatenate(domain_chunks) if domain_chunks else np.empty(0, int),
        "index": np.concatenate(index_chunks) if index_chunks else np.empty(0, int),
    }


def build_prediction_table(
    outputs: dict[str, np.ndarray],
    manifest_slice: Any,
    *,
    probabilities: np.ndarray | None = None,
    num_classes: int = 5,
) -> Any:
    """Sample-level table joining predictions back to the manifest.

    Uses the dataset index carried through the loader, so rows line up with the
    manifest even when the loader shuffled (it does not, for evaluation, but
    relying on order would be fragile).
    """
    import pandas as pd

    probabilities = outputs["probabilities"] if probabilities is None else probabilities
    frame = manifest_slice.reset_index(drop=True).iloc[outputs["index"]].reset_index(drop=True)

    # Use the decision rule the head actually used, carried in outputs["y_pred"],
    # not a fresh argmax -- the two disagree for the ordinal head.
    predictions = np.asarray(outputs["y_pred"]).astype(int)
    table = pd.DataFrame(
        {
            "image_id": frame["image_id"].to_numpy(),
            "dataset": frame["domain"].to_numpy(),
            "path": frame["path"].to_numpy(),
            "true_grade": outputs["y_true"],
            "predicted_grade": predictions,
        }
    )
    for grade in range(num_classes):
        table[f"probability_grade_{grade}"] = probabilities[:, grade]

    table["confidence"] = probabilities[np.arange(len(predictions)), predictions]
    table["correct"] = table["true_grade"] == table["predicted_grade"]
    table["absolute_grade_error"] = (table["true_grade"] - table["predicted_grade"]).abs()
    return table


def evaluate_split(
    model: Any,
    loader: Any,
    manifest_slice: Any,
    *,
    split_name: str,
    device: str = "cuda",
    amp: bool = True,
    num_classes: int = 5,
    temperature: TemperatureScaler | None = None,
    predictions_dir: Path | str | None = None,
    experiment_id: str = "unnamed",
    to_probabilities: Any | None = None,
    predict_fn: Any | None = None,
) -> tuple[EvaluationResult, dict[str, np.ndarray], Any]:
    """Evaluate one split. Returns ``(result, raw outputs, prediction table)``."""
    outputs = predict_with_logits(
        model, loader, device=device, amp=amp, num_classes=num_classes,
        to_probabilities=to_probabilities, predict_fn=predict_fn,
    )
    domains = sorted(set(manifest_slice["domain"].unique()))
    domain_label = domains[0] if len(domains) == 1 else "+".join(domains)

    result = EvaluationResult(
        split_name=split_name,
        domain=domain_label,
        n_samples=int(len(outputs["y_true"])),
    )
    result.metrics = compute_all_metrics(
        outputs["y_true"], outputs["y_pred"], outputs["probabilities"],
        num_classes=num_classes,
    )
    result.calibration = calibration_metrics(
        outputs["probabilities"], outputs["y_true"], num_classes=num_classes
    )

    probabilities_for_table = outputs["probabilities"]
    if temperature is not None:
        scaled = temperature.apply(outputs["logits"], to_probabilities)
        result.calibration_scaled = calibration_metrics(
            scaled, outputs["y_true"], num_classes=num_classes
        )
        result.temperature = temperature.describe()
        outputs["probabilities_scaled"] = scaled

    table = build_prediction_table(
        outputs, manifest_slice, probabilities=probabilities_for_table, num_classes=num_classes
    )
    if temperature is not None:
        table["confidence_scaled"] = outputs["probabilities_scaled"].max(axis=1)

    if predictions_dir is not None:
        directory = ensure_dir(predictions_dir)
        path = directory / f"{experiment_id}__{split_name}_predictions.csv"
        table.to_csv(path, index=False)
        result.predictions_path = str(path)

    log.info(
        "%s [%s] n=%d | QWK %.4f | macroF1 %.4f | MAE %.3f | ECE %.4f%s",
        split_name, domain_label, result.n_samples,
        result.metrics.get("qwk", float("nan")),
        result.metrics.get("f1_macro", float("nan")),
        result.metrics.get("mae_grade", float("nan")),
        result.calibration.get("ece", float("nan")),
        (
            f" -> ECE {result.calibration_scaled['ece']:.4f} after T={temperature.temperature:.3f}"
            if temperature is not None else ""
        ),
    )
    return result, outputs, table


def evaluate_experiment(
    model: Any,
    loaders: dict[str, Any],
    experiment: Any,
    *,
    experiment_id: str,
    device: str = "cuda",
    amp: bool = True,
    num_classes: int = 5,
    predictions_dir: Path | str | None = None,
    fit_temperature_on_val: bool = True,
    to_probabilities: Any | None = None,
    predict_fn: Any | None = None,
) -> dict[str, Any]:
    """Evaluate validation and target-domain test splits under the protocol.

    Order matters and is enforced here:

    1. Evaluate the **source validation** split.
    2. Fit temperature on those validation logits.
    3. Apply that fixed temperature to the **target test** split.

    The target domain never influences the temperature, the metrics chosen, or
    anything else.
    """
    results: dict[str, EvaluationResult] = {}
    tables: dict[str, Any] = {}

    val_result, val_outputs, val_table = evaluate_split(
        model, loaders["val"], experiment.val, split_name="source_val",
        device=device, amp=amp, num_classes=num_classes,
        predictions_dir=predictions_dir, experiment_id=experiment_id,
        to_probabilities=to_probabilities, predict_fn=predict_fn,
    )
    results["source_val"] = val_result
    tables["source_val"] = val_table

    temperature: TemperatureScaler | None = None
    if fit_temperature_on_val and len(val_outputs["y_true"]) > 0:
        sources = "+".join(experiment.source_domains)
        temperature = fit_temperature(
            val_outputs["logits"], val_outputs["y_true"],
            fitted_on=f"source_validation[{sources}]",
            to_probabilities=to_probabilities,
        )
        # Re-score validation with its own temperature, for the before/after figure.
        results["source_val"].calibration_scaled = calibration_metrics(
            temperature.apply(val_outputs["logits"], to_probabilities),
            val_outputs["y_true"], num_classes=num_classes,
        )
        results["source_val"].temperature = temperature.describe()

    test_result, _test_outputs, test_table = evaluate_split(
        model, loaders["test"], experiment.test,
        split_name=f"target_test[{experiment.target_domain}]",
        device=device, amp=amp, num_classes=num_classes, temperature=temperature,
        predictions_dir=predictions_dir, experiment_id=experiment_id,
        to_probabilities=to_probabilities, predict_fn=predict_fn,
    )
    results["target_test"] = test_result
    tables["target_test"] = test_table

    return {
        "experiment_id": experiment_id,
        "protocol": experiment.protocol,
        "source_domains": experiment.source_domains,
        "target_domain": experiment.target_domain,
        "results": results,
        "tables": tables,
        "temperature": temperature.describe() if temperature else None,
    }
