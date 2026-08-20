"""Path resolution, config loading and small serialisation helpers.

Everything in this project resolves paths through :func:`project_root` so that
the code works identically when executed from a VS Code interactive cell (whose
working directory is usually the workspace root) and from a shell inside the
package directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "project_root",
    "load_yaml",
    "load_paths_config",
    "resolve",
    "write_json",
    "read_json",
    "ensure_dir",
]

_MARKER_FILES = ("configs/paths.yaml", "requirements.txt")


def project_root(start: Path | str | None = None) -> Path:
    """Return the repository root by walking upwards until marker files appear.

    Falls back to the parent-of-``src`` location of this file, which is correct
    for any normal checkout.
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and all((candidate / m).exists() for m in _MARKER_FILES):
            return candidate
    # src/utils/io.py -> src/utils -> src -> root
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping (got {type(data).__name__})")
    return data


def load_paths_config(root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/paths.yaml`` and verify that every declared path exists.

    Raises ``FileNotFoundError`` listing *all* missing paths rather than dying on
    the first one, so a broken setup can be repaired in a single pass.
    """
    root = root or project_root()
    cfg = load_yaml(root / "configs" / "paths.yaml")

    path_keys = {"source_folder", "image_dir", "image_root", "csv", "label_csv", "data_root"}
    missing: list[str] = []

    def _walk(node: Any, trail: str = "") -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                where = f"{trail}.{key}" if trail else str(key)
                if key in path_keys and isinstance(value, str) and not Path(value).exists():
                    missing.append(f"{where}: {value}")
                _walk(value, where)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                _walk(value, f"{trail}[{i}]")

    _walk(cfg)
    if missing:
        raise FileNotFoundError(
            "configs/paths.yaml references paths that do not exist:\n  "
            + "\n  ".join(missing)
            + "\n\nEdit configs/paths.yaml to match your machine."
        )
    return cfg


def resolve(*parts: Path | str) -> Path:
    """Join ``parts`` onto the project root unless the first part is absolute."""
    first = Path(parts[0])
    base = first if first.is_absolute() else project_root() / first
    return base.joinpath(*(str(p) for p in parts[1:]))


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=_json_default)
        fh.write("\n")
    return p


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
