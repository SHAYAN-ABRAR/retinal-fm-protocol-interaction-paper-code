"""Train / validation / test construction, and the experiment protocols.

Two layers:

1. :func:`assign_within_domain_splits` gives every image a ``split`` label inside
   its own domain, using the strongest grouping each domain supports.
2. :func:`build_experiment_split` assembles those per-domain splits into one of
   the three protocols (in-domain, single-source external, leave-one-domain-out).

Splitting strategy per domain, and why
--------------------------------------
======== ====================== ==========================================================
Domain   Strategy               Justification
======== ====================== ==========================================================
eyepacs  **patient-level**      Patient ids exist (``<patient>_<eye>``) and 87.3% of
                                patients share a grade across both eyes. Splitting at
                                image level would leak a near-duplicate label.
ddr      image-level stratified Patient ids are not recoverable (Phase 1 tested and
                                rejected the obvious hypothesis). Documented limitation.
aptos    vendor split kept      Measured disjoint. No patient ids exist at all.
idrid    vendor split kept,     Official train/test is the comparable choice; validation
         val carved from train  is carved from train so test stays untouched.
======== ====================== ==========================================================

The one rule that overrides everything
--------------------------------------
In a leave-one-domain-out experiment the target domain contributes **only** test
data.  It never appears in train or val, so it cannot influence weights, early
stopping, temperature scaling or model selection.  :func:`build_experiment_split`
enforces this with an assertion rather than by convention.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ..utils.logging import get_logger
from .schema import N_GRADES

log = get_logger("data.splits")

__all__ = [
    "SplitConfig",
    "ExperimentSplit",
    "assign_within_domain_splits",
    "build_experiment_split",
    "leave_one_domain_out_plan",
]

Protocol = Literal["in_domain", "single_source", "lodo"]

# Domains whose vendor split is trustworthy and should be preserved.
_VENDOR_SPLIT_DOMAINS = {
    "aptos": {"train": "train", "val": "val", "test": "test"},
    "idrid": {"train": "train", "test": "test"},   # val is carved out of train
}


@dataclass
class SplitConfig:
    """Fractions and seed for constructing splits. Recorded with every run."""

    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.val_fraction < 1 or not 0 <= self.test_fraction < 1:
            raise ValueError("fractions must lie in (0, 1)")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")


@dataclass
class ExperimentSplit:
    """One assembled experiment: which rows are train, val and test."""

    protocol: str
    source_domains: list[str]
    target_domain: str | None
    train: Any = None
    val: Any = None
    test: Any = None
    notes: list[str] = field(default_factory=list)

    def sizes(self) -> dict[str, int]:
        return {
            "train": 0 if self.train is None else len(self.train),
            "val": 0 if self.val is None else len(self.val),
            "test": 0 if self.test is None else len(self.test),
        }

    def name(self) -> str:
        sources = "-".join(sorted(self.source_domains))
        return f"{self.protocol}_{sources}__{self.target_domain or sources}"


# ---------------------------------------------------------------------------
# grouped stratified splitting
# ---------------------------------------------------------------------------

def _group_stratified_split(
    groups: dict[str, list[int]],
    group_label: dict[str, int],
    *,
    fractions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits, stratified by the group's label.

    Groups are the atomic unit: every image in a group lands in the same split.
    For EyePACS a group is a patient; for DDR it is a single image.

    Stratification is done per label so that rare grades (DDR grade 3 has 236
    images) are represented in every split rather than landing entirely in one.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for group, label in group_label.items():
        by_label[label].append(group)

    split_names = list(fractions)
    assignment: dict[str, str] = {}

    for label in sorted(by_label):
        members = sorted(by_label[label])       # sort first => deterministic
        rng.shuffle(members)
        total = len(members)

        # Largest-remainder allocation keeps the split sizes as close to the
        # requested fractions as integer counts allow.
        raw = {name: fractions[name] * total for name in split_names}
        counts = {name: int(raw[name]) for name in split_names}
        remainder = total - sum(counts.values())
        for name in sorted(split_names, key=lambda n: raw[n] - counts[n], reverse=True):
            if remainder <= 0:
                break
            counts[name] += 1
            remainder -= 1

        cursor = 0
        for name in split_names:
            for group in members[cursor : cursor + counts[name]]:
                assignment[group] = name
            cursor += counts[name]

    return assignment


def _patient_label(grades: Sequence[int]) -> int:
    """Severity label for a patient: the worse eye.

    Clinically the referral decision follows the more severe eye, and it keeps
    stratification meaningful when the two eyes disagree (12.7% of patients).
    """
    return int(max(grades))


def assign_within_domain_splits(
    manifest: "Any",
    config: SplitConfig | None = None,
) -> "Any":
    """Add a ``split`` column ('train' | 'val' | 'test') to the manifest."""
    import pandas as pd

    config = config or SplitConfig()
    manifest = manifest.copy()
    manifest["split"] = pd.Series([pd.NA] * len(manifest), dtype="string")

    train_fraction = 1.0 - config.val_fraction - config.test_fraction
    fractions = {
        "train": train_fraction,
        "val": config.val_fraction,
        "test": config.test_fraction,
    }

    for domain, group in manifest.groupby("domain", sort=False):
        vendor = _VENDOR_SPLIT_DOMAINS.get(domain)

        if vendor and set(vendor) == {"train", "val", "test"}:
            # Fully specified vendor split -- keep it verbatim.
            manifest.loc[group.index, "split"] = group["source_split"].values
            log.info("%s: kept vendor train/val/test split", domain)
            continue

        if vendor and set(vendor) == {"train", "test"}:
            # Vendor gives train/test only; carve val out of train, never test.
            manifest.loc[group.index, "split"] = group["source_split"].values
            train_rows = group[group["source_split"] == "train"]
            val_share = config.val_fraction / (1.0 - config.test_fraction)
            assignment = _group_stratified_split(
                {row.image_id: [i] for i, row in zip(train_rows.index, train_rows.itertuples())},
                {row.image_id: int(row.grade) for row in train_rows.itertuples()},
                fractions={"train": 1.0 - val_share, "val": val_share},
                seed=config.seed,
            )
            for index, image_id in zip(train_rows.index, train_rows["image_id"]):
                manifest.at[index, "split"] = assignment[image_id]
            log.info(
                "%s: kept vendor test (%d); carved val from train",
                domain, int((group["source_split"] == "test").sum()),
            )
            continue

        # No usable vendor split: build train/val/test ourselves.
        has_patients = group["patient_id"].notna().all()
        if has_patients:
            groups: dict[str, list[int]] = defaultdict(list)
            grades: dict[str, list[int]] = defaultdict(list)
            for index, patient, grade in zip(
                group.index, group["patient_id"], group["grade"]
            ):
                groups[patient].append(index)
                grades[patient].append(int(grade))
            group_label = {p: _patient_label(g) for p, g in grades.items()}
            unit = "patient"
        else:
            groups = {row.image_id: [index] for index, row in zip(group.index, group.itertuples())}
            group_label = {row.image_id: int(row.grade) for row in group.itertuples()}
            unit = "image"

        assignment = _group_stratified_split(
            groups, group_label, fractions=fractions, seed=config.seed
        )
        for key, indices in groups.items():
            manifest.loc[indices, "split"] = assignment[key]

        log.info(
            "%s: %s-level stratified split over %d %ss -> %s",
            domain, unit, len(groups), unit,
            manifest.loc[group.index, "split"].value_counts().to_dict(),
        )

    if manifest["split"].isna().any():
        unassigned = manifest.loc[manifest["split"].isna(), "domain"].unique().tolist()
        raise RuntimeError(f"rows left unassigned in domains {unassigned}")

    return manifest


# ---------------------------------------------------------------------------
# experiment protocols
# ---------------------------------------------------------------------------

def build_experiment_split(
    manifest: "Any",
    *,
    protocol: Protocol,
    sources: Sequence[str],
    target: str | None = None,
) -> ExperimentSplit:
    """Assemble one experiment from a split-annotated manifest.

    Parameters
    ----------
    protocol:
        ``in_domain``   -- train/val/test all from the single source domain.
        ``single_source`` -- train/val from one source; test = all of ``target``.
        ``lodo``        -- train/val from several sources; test = all of ``target``.
    """
    import pandas as pd

    if "split" not in manifest.columns:
        raise ValueError("manifest has no 'split' column; call assign_within_domain_splits first")

    sources = list(sources)
    known = set(manifest["domain"].unique())
    for domain in [*sources, *([target] if target else [])]:
        if domain not in known:
            raise ValueError(f"unknown domain {domain!r}; manifest has {sorted(known)}")

    if protocol == "in_domain":
        if len(sources) != 1:
            raise ValueError("in_domain requires exactly one source domain")
        if target not in (None, sources[0]):
            raise ValueError("in_domain target must be the source domain")
        target = sources[0]
        subset = manifest[manifest["domain"] == sources[0]]
        result = ExperimentSplit(
            protocol=protocol,
            source_domains=sources,
            target_domain=target,
            train=subset[subset["split"] == "train"].copy(),
            val=subset[subset["split"] == "val"].copy(),
            test=subset[subset["split"] == "test"].copy(),
        )
        result.notes.append("test split comes from the same domain as training")
        return result

    if target is None:
        raise ValueError(f"protocol {protocol!r} requires a target domain")
    if target in sources:
        raise ValueError(
            f"target {target!r} is also listed as a source. The held-out domain must "
            "never contribute training data."
        )
    if protocol == "single_source" and len(sources) != 1:
        raise ValueError("single_source requires exactly one source domain")
    if protocol == "lodo" and len(sources) < 2:
        raise ValueError("lodo requires at least two source domains")

    source_rows = manifest[manifest["domain"].isin(sources)]
    target_rows = manifest[manifest["domain"] == target]

    result = ExperimentSplit(
        protocol=protocol,
        source_domains=sources,
        target_domain=target,
        train=source_rows[source_rows["split"] == "train"].copy(),
        val=source_rows[source_rows["split"] == "val"].copy(),
        # The ENTIRE target domain is test data. Its own train/val labels are
        # irrelevant here -- none of it may be seen before the final evaluation.
        test=target_rows.copy(),
    )
    result.notes.append(
        f"target domain {target!r} contributes test data only ({len(target_rows)} images); "
        "it is excluded from training, validation, early stopping and calibration"
    )
    _assert_target_isolation(result)
    return result


def _assert_target_isolation(split: ExperimentSplit) -> None:
    """The protocol guarantee, checked rather than trusted."""
    if split.protocol == "in_domain":
        return
    for name in ("train", "val"):
        frame = getattr(split, name)
        if frame is None or not len(frame):
            continue
        contaminated = frame[frame["domain"] == split.target_domain]
        if len(contaminated):
            raise AssertionError(
                f"{len(contaminated)} target-domain rows leaked into the {name} split "
                f"of experiment {split.name()!r}"
            )
    overlap = set(split.train["image_id"]) & set(split.test["image_id"])
    if overlap:
        raise AssertionError(f"{len(overlap)} image_ids appear in both train and test")


def leave_one_domain_out_plan(domains: Sequence[str]) -> list[dict[str, Any]]:
    """The four LODO experiments: each domain held out once."""
    domains = list(domains)
    return [
        {
            "protocol": "lodo",
            "sources": [d for d in domains if d != held_out],
            "target": held_out,
        }
        for held_out in domains
    ]
