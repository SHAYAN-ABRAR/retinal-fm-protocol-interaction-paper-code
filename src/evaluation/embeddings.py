"""Extract and project learned features, for the domain-invariance analysis.

The question this answers
------------------------
Phase 2 showed that *hand-crafted image statistics* separate the domains
cleanly. That is unsurprising -- the datasets use different cameras. The
scientifically interesting question is whether the **trained model's** internal
representation still encodes which dataset an image came from, after being
optimised only to predict severity.

If a domain-generalization method works, its features should cluster less by
domain and no worse by grade than ERM's. This module measures that, so the claim
rests on a number rather than on how a scatter plot looks.

Quantifying it
--------------
Two summaries accompany every embedding, because eyeballing a projection is not
evidence:

**Domain separability** -- the accuracy of a simple linear probe trained to
predict the *domain* from the features. If a logistic regression recovers the
domain at 95%, the representation is not domain-invariant, whatever the UMAP
suggests. Chance level (the majority-class rate) is reported alongside, since
domains are not equally sized.

**Silhouette scores** -- how tightly the features cluster by domain versus by
grade. A representation useful for this task should cluster by grade; one that
clusters by domain is encoding the confound.

Both are computed on the *raw* feature vectors, not on the 2-D projection. UMAP
and t-SNE distort global structure by design, so measuring on their output would
measure the projection rather than the representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger("evaluation.embeddings")

__all__ = [
    "EmbeddingResult",
    "extract_features",
    "project_pca",
    "project_umap",
    "project_tsne",
    "domain_separability",
    "cluster_scores",
    "domain_centroid_distances",
    "analyse_embeddings",
]


@dataclass
class EmbeddingResult:
    features: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    y_true: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    domain_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    domain: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    projections: dict[str, np.ndarray] = field(default_factory=dict)
    domain_separability: dict[str, float] = field(default_factory=dict)
    cluster_scores: dict[str, float] = field(default_factory=dict)
    centroid_distances: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": int(len(self.y_true)),
            "feature_dim": int(self.features.shape[1]) if self.features.size else 0,
            "domain_separability": self.domain_separability,
            "cluster_scores": self.cluster_scores,
            "centroid_distances": self.centroid_distances,
            "projections": sorted(self.projections),
        }


def extract_features(
    model: Any,
    loader: Any,
    *,
    device: str = "cuda",
    amp: bool = True,
    max_samples: int | None = 4000,
) -> dict[str, np.ndarray]:
    """Collect pooled backbone features for a loader.

    ``max_samples`` caps the collection: UMAP and t-SNE scale poorly, and a
    seeded subsample of a few thousand points is visually and statistically
    indistinguishable from the full 35k for this purpose.
    """
    import torch

    model.eval()
    model.to(device)

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    collected = 0

    with torch.inference_mode():
        for images, batch_targets, batch_domains, _indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device != "cpu"):
                _logits, batch_features = model(images, return_features=True)
            features.append(batch_features.float().cpu().numpy())
            targets.append(batch_targets.numpy())
            domains.append(batch_domains.numpy())
            collected += len(batch_targets)
            if max_samples is not None and collected >= max_samples:
                break

    result = {
        "features": np.concatenate(features) if features else np.empty((0, 0)),
        "y_true": np.concatenate(targets) if targets else np.empty(0, dtype=int),
        "domain_id": np.concatenate(domains) if domains else np.empty(0, dtype=int),
    }
    log.info(
        "extracted %d feature vectors of dimension %d",
        len(result["y_true"]), result["features"].shape[1] if result["features"].size else 0,
    )
    return result


def _standardise(features: np.ndarray) -> np.ndarray:
    centred = features - features.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return centred / scale


def project_pca(features: np.ndarray, *, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA via SVD. Returns ``(projection, explained variance ratio)``."""
    standardised = _standardise(features)
    _u, singular, components = np.linalg.svd(standardised, full_matrices=False)
    explained = (singular**2 / (singular**2).sum())[:n_components] * 100
    return standardised @ components[:n_components].T, explained


def project_umap(
    features: np.ndarray, *, n_neighbors: int = 30, min_dist: float = 0.1, seed: int = 42
) -> np.ndarray | None:
    """UMAP projection, or None when umap-learn is not installed.

    Returning None rather than raising lets the figure pipeline degrade to PCA
    instead of failing an otherwise-complete analysis.
    """
    try:
        import umap
    except ImportError:
        log.warning("umap-learn is not installed; skipping the UMAP projection")
        return None

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed
    )
    return reducer.fit_transform(_standardise(features))


def project_tsne(features: np.ndarray, *, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """t-SNE projection. Slow above a few thousand points; PCA-reduced first."""
    from sklearn.manifold import TSNE

    reduced = features
    if features.shape[1] > 50:
        reduced, _ = project_pca(features, n_components=50)
    return TSNE(
        n_components=2, perplexity=perplexity, random_state=seed, init="pca"
    ).fit_transform(reduced)


def domain_separability(
    features: np.ndarray,
    domain_id: Sequence[int] | np.ndarray,
    *,
    seed: int = 42,
    max_iter: int = 1000,
) -> dict[str, float]:
    """How well a linear probe recovers the DOMAIN from the features.

    High accuracy means the representation still encodes the dataset of origin.
    ``chance`` is the majority-domain rate, which is the honest baseline when
    domains differ in size -- a probe scoring 0.76 on a 76/24 split has learned
    nothing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    domain_id = np.asarray(domain_id).astype(int)
    if len(np.unique(domain_id)) < 2:
        return {"probe_accuracy": float("nan"), "chance": float("nan"), "lift": float("nan")}

    standardised = _standardise(features)
    # No multi_class argument: it was removed in scikit-learn 1.9, and the
    # behaviour it used to select ("auto") is now the only behaviour.
    probe = LogisticRegression(max_iter=max_iter, random_state=seed)
    scores = cross_val_score(probe, standardised, domain_id, cv=3, scoring="accuracy")

    counts = np.bincount(domain_id)
    chance = float(counts.max() / counts.sum())
    accuracy = float(scores.mean())
    return {
        "probe_accuracy": accuracy,
        "probe_accuracy_std": float(scores.std()),
        "chance": chance,
        # 0 = no domain information beyond the class prior; 1 = perfectly separable.
        "lift": float((accuracy - chance) / (1.0 - chance)) if chance < 1 else float("nan"),
    }


def cluster_scores(
    features: np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    domain_id: Sequence[int] | np.ndarray,
    *,
    sample_size: int = 3000,
    seed: int = 42,
) -> dict[str, float]:
    """Silhouette scores for clustering by domain versus by grade.

    Higher silhouette-by-domain means the representation groups images by
    dataset; higher silhouette-by-grade means it groups them by disease
    severity, which is what the task needs.
    """
    from sklearn.metrics import silhouette_score

    y_true = np.asarray(y_true).astype(int)
    domain_id = np.asarray(domain_id).astype(int)

    rng = np.random.default_rng(seed)
    if len(features) > sample_size:
        index = rng.choice(len(features), sample_size, replace=False)
        features, y_true, domain_id = features[index], y_true[index], domain_id[index]

    standardised = _standardise(features)
    scores: dict[str, float] = {}
    for name, labels in (("domain", domain_id), ("grade", y_true)):
        if len(np.unique(labels)) < 2:
            scores[f"silhouette_by_{name}"] = float("nan")
            continue
        scores[f"silhouette_by_{name}"] = float(silhouette_score(standardised, labels))
    return scores


def domain_centroid_distances(
    features: np.ndarray,
    domain_id: Sequence[int] | np.ndarray,
    *,
    domain_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Pairwise Euclidean distance between per-domain feature centroids.

    Reported relative to the mean within-domain spread, so the number is a ratio
    rather than an arbitrary scale that changes with feature dimension.
    """
    domain_id = np.asarray(domain_id).astype(int)
    standardised = _standardise(features)

    centroids: dict[int, np.ndarray] = {}
    spreads: dict[int, float] = {}
    for domain in np.unique(domain_id):
        subset = standardised[domain_id == domain]
        centroids[int(domain)] = subset.mean(axis=0)
        spreads[int(domain)] = float(np.linalg.norm(subset - subset.mean(axis=0), axis=1).mean())

    names = domain_names or {d: str(d) for d in centroids}
    pairs: dict[str, float] = {}
    ratios: dict[str, float] = {}
    domains = sorted(centroids)
    for i, a in enumerate(domains):
        for b in domains[i + 1:]:
            distance = float(np.linalg.norm(centroids[a] - centroids[b]))
            key = f"{names.get(a, a)}|{names.get(b, b)}"
            pairs[key] = distance
            mean_spread = (spreads[a] + spreads[b]) / 2
            ratios[key] = float(distance / mean_spread) if mean_spread > 0 else float("nan")

    return {
        "centroid_distance": pairs,
        "distance_over_within_domain_spread": ratios,
        "within_domain_spread": {names.get(d, d): s for d, s in spreads.items()},
    }


def analyse_embeddings(
    features: np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    domain_id: Sequence[int] | np.ndarray,
    *,
    domain_names: dict[int, str] | None = None,
    include_umap: bool = True,
    include_tsne: bool = False,
    seed: int = 42,
) -> EmbeddingResult:
    """Project the features and quantify domain versus grade structure."""
    y_true = np.asarray(y_true).astype(int)
    domain_id = np.asarray(domain_id).astype(int)
    names = domain_names or {}

    result = EmbeddingResult(
        features=features,
        y_true=y_true,
        domain_id=domain_id,
        domain=np.array([names.get(int(d), str(d)) for d in domain_id], dtype=object),
    )

    projection, explained = project_pca(features)
    result.projections["pca"] = projection
    result.cluster_scores["pca_explained_variance_pc1_pc2"] = float(explained.sum())

    if include_umap:
        umap_projection = project_umap(features, seed=seed)
        if umap_projection is not None:
            result.projections["umap"] = umap_projection
    if include_tsne:
        result.projections["tsne"] = project_tsne(features, seed=seed)

    result.domain_separability = domain_separability(features, domain_id, seed=seed)
    result.cluster_scores.update(cluster_scores(features, y_true, domain_id, seed=seed))
    result.centroid_distances = domain_centroid_distances(
        features, domain_id, domain_names=names
    )

    log.info(
        "embeddings: domain probe %.3f (chance %.3f, lift %.3f) | "
        "silhouette domain %.3f vs grade %.3f",
        result.domain_separability["probe_accuracy"],
        result.domain_separability["chance"],
        result.domain_separability["lift"],
        result.cluster_scores.get("silhouette_by_domain", float("nan")),
        result.cluster_scores.get("silhouette_by_grade", float("nan")),
    )
    return result
