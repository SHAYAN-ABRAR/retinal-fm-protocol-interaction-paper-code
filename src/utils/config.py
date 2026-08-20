"""Load YAML experiment configurations into the project's dataclasses.

Why this exists rather than plain ``yaml.safe_load``
----------------------------------------------------
A config file that is only ever read as a dict drifts from the code: a key gets
renamed in Python, the YAML keeps the old name, and the run silently uses a
default instead of the value the file declares. Everything here goes through the
real dataclasses, so an unknown or misspelled key **raises** instead of being
ignored.

That matters more than usual in this project: a config that silently fell back
to a default batch size or a default temperature-fitting split would produce a
result that does not match the file describing it.

Composition
-----------
Configs may declare ``extends: <file>``, merged depth-first with the child
overriding the parent. This lets ``domain_generalization.yaml`` inherit every
training detail from ``baseline.yaml`` and state only what differs -- which is
the point of an ablation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io import load_yaml, project_root
from .logging import get_logger

log = get_logger("utils.config")

__all__ = [
    "ExperimentConfig",
    "load_config",
    "resolve_config",
    "build_components",
]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str, *, root: Path | None = None, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving ``extends`` chains."""
    root = root or project_root()
    path = Path(path)
    if not path.is_absolute():
        path = root / "configs" / path if not (root / path).exists() else root / path
    path = path.resolve()

    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular extends chain at {path}")
    _seen.add(path)

    config = load_yaml(path)
    parent_name = config.pop("extends", None)
    if parent_name:
        parent = load_config(parent_name, root=root, _seen=_seen)
        config = _deep_merge(parent, config)
    return config


def _construct(cls: type, payload: Mapping[str, Any] | None, *, where: str) -> Any:
    """Build a dataclass, rejecting unknown keys instead of ignoring them."""
    import dataclasses

    payload = dict(payload or {})
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - fields)
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown} for {cls.__name__}. "
            f"Valid keys: {sorted(fields)}. A silently ignored key would make the "
            "config file disagree with the run it describes."
        )
    return cls(**payload)


class ExperimentConfig:
    """A fully resolved experiment configuration.

    Attributes are the project's real dataclasses, so anything the YAML declares
    has already been type-checked by construction.
    """

    def __init__(self, raw: Mapping[str, Any], *, name: str = "unnamed") -> None:
        from ..data.augmentations import AugmentationConfig
        from ..data.loaders import LoaderConfig
        from ..data.preprocessing import PreprocessConfig
        from ..data.splits import SplitConfig
        from ..models.backbones import BackboneConfig, resolve_input_size
        from ..training.methods import MethodConfig
        from ..training.trainer import TrainConfig

        self.name = raw.get("name", name)
        self.description = raw.get("description", "")
        self.raw = dict(raw)

        self.seed: int = int(raw.get("seed", 42))
        self.seeds: list[int] = [int(s) for s in raw.get("seeds", [self.seed])]
        self.image_size: int = int(raw.get("image_size", 224))

        self.preprocess = _construct(
            PreprocessConfig, raw.get("preprocess"), where=f"{self.name}.preprocess"
        )
        self.augmentation = _construct(
            AugmentationConfig, raw.get("augmentation"), where=f"{self.name}.augmentation"
        )
        self.split = _construct(SplitConfig, raw.get("split"), where=f"{self.name}.split")
        self.loader = _construct(LoaderConfig, raw.get("loader"), where=f"{self.name}.loader")
        self.method = _construct(MethodConfig, raw.get("method"), where=f"{self.name}.method")
        self.backbone = _construct(
            BackboneConfig, raw.get("backbone"), where=f"{self.name}.backbone"
        )
        self.training = _construct(
            TrainConfig, raw.get("training"), where=f"{self.name}.training"
        )

        # One resolution, declared once at the top, propagated everywhere -- and
        # snapped per backbone, because DINOv2 is patch-14 and cannot take 384.
        self.backbone.image_size = resolve_input_size(self.backbone.name, self.image_size)
        object.__setattr__(self.preprocess, "image_size", self.image_size)
        object.__setattr__(self.augmentation, "image_size", self.image_size)

        # Batch size lives in TrainConfig for the record and in LoaderConfig for
        # the DataLoader. Two sources of truth would eventually disagree.
        if "batch_size" not in (raw.get("loader") or {}):
            self.loader.batch_size = self.training.batch_size
        elif self.loader.batch_size != self.training.batch_size:
            raise ValueError(
                f"{self.name}: loader.batch_size ({self.loader.batch_size}) disagrees with "
                f"training.batch_size ({self.training.batch_size}). Set it once."
            )
        self.loader.seed = self.seed
        self.training.seed = self.seed
        self.split.seed = int(raw.get("split", {}).get("seed", self.split.seed))

        self.protocol: str = raw.get("protocol", "lodo")
        self.sources: list[str] = list(raw.get("sources", []))
        self.target: str | None = raw.get("target")

    def describe(self) -> dict[str, Any]:
        """Flat description for the experiment registry."""
        return {
            "config_name": self.name,
            "protocol": self.protocol,
            "sources": self.sources,
            "target": self.target,
            "seed": self.seed,
            "image_size": self.image_size,
            **self.method.describe(),
            **self.backbone.describe(),
            **self.training.describe(),
            "loader": self.loader.describe(),
            "augmentation": self.augmentation.describe(),
            "preprocess": self.preprocess.describe(),
        }

    def __repr__(self) -> str:
        target = self.target or "-"
        return (
            f"ExperimentConfig({self.name!r}, {self.protocol}, "
            f"sources={self.sources}, target={target!r}, "
            f"method={self.method.name!r}, backbone={self.backbone.name!r}, "
            f"seed={self.seed})"
        )


def resolve_config(path: Path | str, *, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config and return it as an :class:`ExperimentConfig`."""
    raw = load_config(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    config = ExperimentConfig(raw, name=Path(path).stem)
    log.info("resolved config %s", config)
    return config


def build_components(config: ExperimentConfig, *, class_counts: Any = None, device: str = "cuda") -> Any:
    """Instantiate the model, loss and trainer hooks this config describes."""
    from ..training.methods import build_method

    return build_method(
        config.method, config.backbone, class_counts=class_counts, device=device
    )
