"""Unified manifest across all four domains, plus the PyTorch Dataset.

The manifest is the single source of truth for every experiment: splits, leakage
audits, figures, dataloaders and error analysis all read from it.  Building it is
cheap (no images are opened) and it is cached to CSV so downstream cells do not
re-scan 90k files.

``RetinaDataset`` imports torch lazily, so the manifest, leakage audit and
dataset figures all work in an environment where torch is not installed yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..utils.io import ensure_dir, project_root
from ..utils.logging import get_logger
from .aptos import build_aptos_manifest
from .ddr import build_ddr_manifest
from .eyepacs import build_eyepacs_manifest
from .idrid import build_idrid_manifest
from .schema import GRADE_NAMES, MANIFEST_COLUMNS, N_GRADES, validate_manifest

log = get_logger("data.unified")

__all__ = [
    "build_unified_manifest",
    "load_manifest",
    "save_manifest",
    "domain_summary",
    "RetinaDataset",
    "GRADE_NAMES",
    "N_GRADES",
]

_BUILDERS: dict[str, Callable[..., Any]] = {
    "ddr": build_ddr_manifest,
    "aptos": build_aptos_manifest,
    "idrid": build_idrid_manifest,
    "eyepacs": build_eyepacs_manifest,
}


def build_unified_manifest(paths_config: dict[str, Any], *, root: Path | None = None) -> "Any":
    """Build the manifest for every domain and concatenate.

    Cross-domain integrity is asserted here, where it can actually be checked --
    an individual loader cannot know what the other domains contain.
    """
    import pandas as pd

    root = root or project_root()
    exclusions = paths_config.get("exclusions", {})

    frames = []
    for key, spec in paths_config["domains"].items():
        builder = _BUILDERS.get(key)
        if builder is None:
            raise KeyError(f"no loader registered for domain {key!r}")
        if key == "eyepacs":
            frames.append(builder(spec, project_root=root, exclusions=exclusions))
        else:
            frames.append(builder(spec))

    manifest = validate_manifest(pd.concat(frames, ignore_index=True), name="unified")
    _assert_cross_domain_integrity(manifest, paths_config)

    log.info(
        "unified manifest: %d images across %d domains",
        len(manifest), manifest["domain"].nunique(),
    )
    return manifest


def _assert_cross_domain_integrity(manifest: "Any", paths_config: dict[str, Any]) -> None:
    """Refuse to return a manifest where one photograph sits in two domains.

    This is the failure mode that would quietly destroy the whole study, so it is
    checked on the *file path*, which is the only identity a filesystem
    guarantees, in addition to the id-level checks the loaders already do.
    """
    duplicated_paths = manifest["path"].duplicated(keep=False)
    if duplicated_paths.any():
        offenders = manifest.loc[duplicated_paths, ["image_id", "domain", "path"]]
        cross = offenders.groupby("path")["domain"].nunique()
        cross = cross[cross > 1]
        if len(cross):
            sample = offenders[offenders["path"].isin(cross.index[:3])]
            raise ValueError(
                f"{len(cross)} image file(s) are claimed by more than one domain:\n"
                f"{sample.to_string(index=False)}"
            )
        raise ValueError(
            f"{int(duplicated_paths.sum())} rows share a file path within a domain"
        )

    # Domain ids must match the protocol exactly; a silent reordering would
    # invalidate every cross-domain table produced afterwards.
    declared = {k: v["domain_id"] for k, v in paths_config["domains"].items()}
    observed = manifest.groupby("domain")["domain_id"].unique().to_dict()
    for domain, ids in observed.items():
        if len(ids) != 1 or int(ids[0]) != declared[domain]:
            raise ValueError(f"domain {domain!r} has domain_id {ids}, expected {declared[domain]}")


def save_manifest(manifest: "Any", path: Path | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    manifest.to_csv(path, index=False)
    log.info("saved manifest (%d rows) -> %s", len(manifest), path)
    return path


def load_manifest(path: Path | str) -> "Any":
    """Load a cached manifest and re-assert its contract."""
    import pandas as pd

    manifest = pd.read_csv(
        path, dtype={"patient_id": "string", "eye": "string", "split": "string"}
    )
    return validate_manifest(manifest, name=str(path))


def domain_summary(manifest: "Any") -> "Any":
    """Per-domain counts, class distribution and patient availability."""
    import pandas as pd

    rows = []
    for domain, group in manifest.groupby("domain", sort=False):
        counts = group["grade"].value_counts().reindex(range(N_GRADES), fill_value=0)
        rows.append(
            {
                "domain": domain,
                "domain_id": int(group["domain_id"].iloc[0]),
                "images": len(group),
                **{f"grade_{g}": int(counts[g]) for g in range(N_GRADES)},
                **{
                    f"grade_{g}_pct": round(100.0 * counts[g] / len(group), 2)
                    for g in range(N_GRADES)
                },
                "patients": (
                    int(group["patient_id"].nunique()) if group["patient_id"].notna().any() else 0
                ),
                "patient_ids_available": bool(group["patient_id"].notna().any()),
                "source_splits": ", ".join(sorted(group["source_split"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values("domain_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class RetinaDataset:
    """Lazy-loading dataset over a manifest slice.

    Images are read from disk on demand and never cached in RAM -- with 16 GB of
    system memory and 51k images, caching decoded tensors is not an option.

    Returns ``(image, grade, domain_id, index)``.  The index lets evaluation code
    join predictions back to ``manifest.iloc[index]`` for error analysis without
    passing strings through the collate function.

    Preprocessing
    -------------
    ``preprocess`` must be given when the manifest points at **raw** images, and
    left ``None`` when it points at the pre-resized cache (whose images are
    already cropped and resized).  Applying it twice would re-run the retina crop
    on an already-cropped image -- harmless in principle but wasteful, and it
    would quietly change the pixels the model sees relative to the declared
    configuration.

    ``expected_size`` turns that from a convention into a check: the first item
    read is verified against it, so a cached/raw mix-up fails immediately with a
    clear message instead of training on silently wrong data.
    """

    def __init__(
        self,
        manifest: "Any",
        transform: Callable[..., Any] | None = None,
        *,
        preprocess: Any | None = None,
        expected_size: int | None = None,
        return_domain: bool = True,
    ) -> None:
        missing = [c for c in ("path", "grade", "domain_id") if c not in manifest.columns]
        if missing:
            raise ValueError(f"manifest is missing columns {missing}")
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.preprocess = preprocess
        self.expected_size = expected_size
        self.return_domain = return_domain
        self._paths = self.manifest["path"].tolist()
        self._grades = self.manifest["grade"].astype(int).tolist()
        self._domains = self.manifest["domain_id"].astype(int).tolist()
        self._size_checked = False

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        import numpy as np
        from PIL import Image

        path = self._paths[index]
        try:
            with Image.open(path) as handle:
                array = np.asarray(handle.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - identify the offending file
            raise RuntimeError(f"failed to read image {path}: {exc}") from exc

        if self.preprocess is not None:
            from .preprocessing import preprocess_array

            array = preprocess_array(array, self.preprocess)

        if self.expected_size is not None and not self._size_checked:
            self._size_checked = True
            if array.shape[:2] != (self.expected_size, self.expected_size):
                raise ValueError(
                    f"expected {self.expected_size}x{self.expected_size} images but read "
                    f"{array.shape[1]}x{array.shape[0]} from {path}.\n"
                    "  Using the raw manifest? pass preprocess=PreprocessConfig(...).\n"
                    "  Using the cached manifest? pass preprocess=None and check the "
                    "cache was built at this size."
                )

        if self.transform is not None:
            array = self.transform(image=array)["image"]

        grade = self._grades[index]
        if self.return_domain:
            return array, grade, self._domains[index], index
        return array, grade, index

    def __repr__(self) -> str:
        domains = ", ".join(sorted(self.manifest["domain"].unique())) if "domain" in self.manifest else "?"
        return f"RetinaDataset(n={len(self)}, domains=[{domains}])"
