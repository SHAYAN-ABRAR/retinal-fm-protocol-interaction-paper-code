"""Loading RETFound (Zhou et al., Nature 2023) as a frozen feature extractor.

RETFound is a ViT-Large/16 pretrained with a masked autoencoder on ~1.6 M
retinal images. It is included here to answer the obvious question about every
finding in this project: *is this a small-model artefact?* DenseNet121 is 7 M
parameters and ConvNeXt-Tiny 28 M; RETFound is 304 M and was pretrained on
retina specifically.

⚠ EyePACS is in RETFound's pretraining corpus
---------------------------------------------
RETFound's colour-fundus model was pretrained on MEH-MIDAS **and EyePACS**. The
labels were not used, but the images were, so the representation is fitted to
EyePACS's appearance distribution. **EyePACS therefore cannot serve as a
held-out target domain for this backbone** -- the leave-one-domain-out premise
is simply false there, and reporting such a number would be reporting leakage.

Sources are unaffected: a source domain is seen by construction, so EyePACS
remaining in the *training* pool for a DDR/APTOS/IDRiD target is fine. Only the
target must be unseen. ``assert_target_not_pretrained`` enforces this, and it
raises rather than warns.

DDR, APTOS and IDRiD are *not named* in RETFound's pretraining description,
which is weaker than confirmed absent -- the Nature paper says "MEH-MIDAS and
public datasets" without enumerating the latter in the accessible text. That
uncertainty is recorded in PRETRAINING_OVERLAP and belongs in the paper's
limitations, not in a comment nobody reads.

Obtaining the weights
---------------------
The checkpoint is a gated Hugging Face repo and cannot be downloaded
anonymously. A human must accept the terms once:

    1. sign in at https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
       and request access (auto-approved)
    2. create a token at https://huggingface.co/settings/tokens
    3. set it:  $env:HF_TOKEN = "hf_..."     (or `huggingface-cli login`)

The file is ~4 GB: it is the full MAE checkpoint, encoder plus decoder.

Why the load is checked so aggressively
---------------------------------------
A ViT-Large whose weights failed to load is still a working model -- it just
has random features. It would train, produce plausible-looking numbers, and
support the conclusion "even a foundation model does not help", which is
exactly the result this project would be tempted to believe. A silent load
failure here manufactures a false negative, so :func:`load_retfound_weights`
refuses anything less than a near-complete match and reports what it did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

log = get_logger("models.retfound")

__all__ = [
    "HF_REPO_ID",
    "CHECKPOINT_FILENAME",
    "PRETRAINING_OVERLAP",
    "assert_target_not_pretrained",
    "resolve_checkpoint",
    "load_retfound_weights",
]

HF_REPO_ID = "YukunZhou/RETFound_mae_natureCFP"
CHECKPOINT_FILENAME = "RETFound_mae_natureCFP.pth"

# What is known about the overlap between RETFound's pretraining corpus and this
# project's four domains. "unknown" is deliberately not "clean".
PRETRAINING_OVERLAP: dict[str, str] = {
    "eyepacs": "confirmed",   # named in the pretraining description
    "ddr": "unknown",         # not named; the public-dataset list is not enumerated
    "aptos": "unknown",
    "idrid": "unknown",
}

# At least this fraction of the model's parameters must come from the checkpoint
# before the result is allowed to be called "RETFound".
MIN_LOADED_FRACTION = 0.95


def assert_target_not_pretrained(target: str) -> None:
    """Refuse a target domain RETFound was pretrained on.

    Raises rather than warns. A warning in a log is not a control: the run would
    still finish, write a results row, and be indistinguishable in the tables
    from an honest one.
    """
    status = PRETRAINING_OVERLAP.get(str(target).lower(), "unknown")
    if status == "confirmed":
        raise ValueError(
            f"{target!r} is in RETFound's pretraining corpus, so it cannot be a "
            "held-out target for this backbone -- the model has already seen "
            "these images and the leave-one-domain-out premise does not hold. "
            "It remains valid as a SOURCE domain. Use ddr, aptos or idrid as "
            "the target instead."
        )
    if status == "unknown":
        log.warning(
            "%s is not named in RETFound's pretraining description, but that "
            "description does not enumerate its public datasets. Treat this as "
            "'overlap not ruled out' in the paper's limitations, not as clean.",
            target,
        )


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    """Find the checkpoint locally, or explain exactly how to obtain it."""
    import os

    candidates = []
    if path is not None:
        candidates.append(Path(path))
    environment = os.environ.get("RETFOUND_CHECKPOINT")
    if environment:
        candidates.append(Path(environment))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(HF_REPO_ID, CHECKPOINT_FILENAME))
    except Exception as error:  # noqa: BLE001 - the message matters more than the type
        raise FileNotFoundError(
            f"RETFound checkpoint not available ({type(error).__name__}: {error}).\n"
            f"It is a gated repository, so a person must accept the terms once:\n"
            f"  1. sign in at https://huggingface.co/{HF_REPO_ID} and request access\n"
            f"  2. create a token at https://huggingface.co/settings/tokens\n"
            f'  3. set it:  $env:HF_TOKEN = "hf_..."\n'
            f"Or download {CHECKPOINT_FILENAME} by hand and point "
            f"RETFOUND_CHECKPOINT at it."
        ) from error


def _strip_decoder(state: dict[str, Any]) -> dict[str, Any]:
    """Keep the encoder. The MAE decoder is not part of the feature extractor."""
    drop = ("decoder_", "mask_token")
    return {k: v for k, v in state.items() if not k.startswith(drop)}


def load_retfound_weights(model, checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Load RETFound's encoder into a timm ViT-L/16 and verify that it landed.

    Returns a report describing what matched, which is written into the
    experiment registry so a reader can confirm the weights were real.
    """
    import torch

    path = resolve_checkpoint(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    state = _strip_decoder(state)

    target_state = model.state_dict()

    # MAE fine-tuning uses global average pooling, where the encoder's final
    # norm becomes fc_norm. timm names it that way when global_pool="avg"; the
    # checkpoint still calls it norm.
    if "fc_norm.weight" in target_state and "norm.weight" in state:
        state["fc_norm.weight"] = state.pop("norm.weight")
        state["fc_norm.bias"] = state.pop("norm.bias")

    # Position embeddings are interpolated when the token grid differs, which it
    # does for any input size other than the pretraining 224.
    if "pos_embed" in state and "pos_embed" in target_state:
        if state["pos_embed"].shape != target_state["pos_embed"].shape:
            state["pos_embed"] = _interpolate_pos_embed(
                state["pos_embed"], target_state["pos_embed"])

    usable = {k: v for k, v in state.items()
              if k in target_state and target_state[k].shape == v.shape}
    mismatched = sorted(k for k, v in state.items()
                        if k in target_state and target_state[k].shape != v.shape)
    missing = sorted(set(target_state) - set(usable))

    loaded_parameters = sum(v.numel() for v in usable.values())
    total_parameters = sum(v.numel() for v in target_state.values())
    fraction = loaded_parameters / max(1, total_parameters)

    if fraction < MIN_LOADED_FRACTION:
        raise RuntimeError(
            f"only {fraction:.1%} of the model's parameters came from "
            f"{path.name} (needed {MIN_LOADED_FRACTION:.0%}). The rest would be "
            f"randomly initialised, and a random ViT-L still trains and still "
            f"produces numbers -- it would look like evidence that a foundation "
            f"model does not help.\n"
            f"  missing: {missing[:6]}{' ...' if len(missing) > 6 else ''}\n"
            f"  shape mismatches: {mismatched[:6]}"
        )

    model.load_state_dict(usable, strict=False)
    report = {
        "checkpoint": path.name,
        "keys_loaded": len(usable),
        "parameters_loaded": loaded_parameters,
        "parameters_total": total_parameters,
        "fraction_loaded": round(fraction, 4),
        # The classifier head is expected to be missing -- it is new.
        "not_loaded": missing[:12],
    }
    log.info("RETFound: loaded %d tensors, %.1f%% of parameters, from %s",
             len(usable), 100 * fraction, path.name)
    return report


def _interpolate_pos_embed(source, target):
    """Bicubic-resize the patch position embeddings to the target grid."""
    import math

    import torch
    import torch.nn.functional as F

    n_extra = target.shape[1] - int(math.sqrt(target.shape[1] - 1)) ** 2
    extra, patches = source[:, :n_extra], source[:, n_extra:]
    old = int(math.sqrt(patches.shape[1]))
    new = int(math.sqrt(target.shape[1] - n_extra))
    if old == new:
        return source
    dim = patches.shape[-1]
    patches = patches.reshape(1, old, old, dim).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=(new, new), mode="bicubic", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, new * new, dim)
    return torch.cat([extra, patches], dim=1)
