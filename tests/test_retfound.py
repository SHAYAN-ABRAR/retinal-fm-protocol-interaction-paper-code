"""Tests for the RETFound arm.

Two failure modes here would each manufacture a false result, and both would
look like a successful run.

**Leakage.** RETFound was pretrained on EyePACS. Using EyePACS as a held-out
target would report leakage as generalization, and the number would sit in the
tables looking exactly like an honest one. The guard must raise, not warn.

**A silent load failure.** A ViT-Large whose weights did not load is still a
working model with random features. It trains, produces plausible numbers, and
supports the conclusion "even a foundation model does not help" -- the result
this project would be most tempted to believe. The loader must refuse a partial
match rather than proceed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src.models.retfound import (  # noqa: E402
    MIN_LOADED_FRACTION,
    PRETRAINING_OVERLAP,
    assert_target_not_pretrained,
    load_retfound_weights,
)


def test_eyepacs_is_refused_as_a_target():
    with pytest.raises(ValueError, match="pretraining corpus"):
        assert_target_not_pretrained("eyepacs")
    with pytest.raises(ValueError):
        assert_target_not_pretrained("EyePACS")      # case must not evade it


@pytest.mark.parametrize("target", ["ddr", "aptos", "idrid"])
def test_unnamed_domains_are_allowed_but_flagged(target, caplog):
    """Allowed as targets, with the basis for that recorded.

    RETFound's published CFP corpus is 90.2% MEH-MIDAS and 9.8% EyePACS
    (Zhou et al., Nature 2023) -- an enumerated composition, so these three are
    not in it. The claim rests on the publication, not on the checkpoint, which
    carries no manifest of what it saw; the log says so rather than either
    asserting the datasets are clean or overstating the uncertainty.
    """
    assert PRETRAINING_OVERLAP[target] == "not_in_published_corpus"
    with caplog.at_level("INFO"):
        assert_target_not_pretrained(target)
    assert any("published CFP pretraining corpus" in record.message
               for record in caplog.records)


def test_eyepacs_is_still_refused_outright():
    """The only status that blocks a target is confirmed pretraining overlap."""
    assert PRETRAINING_OVERLAP["eyepacs"] == "confirmed"
    with pytest.raises(ValueError, match="pretraining corpus"):
        assert_target_not_pretrained("eyepacs")


def test_the_probe_runner_excludes_eyepacs_by_default():
    import run_retfound_probe

    assert "eyepacs" not in run_retfound_probe.DEFAULT_TARGETS
    assert set(run_retfound_probe.DEFAULT_TARGETS) == {"ddr", "aptos", "idrid"}
    # It must still appear as a source, or the other targets lose a domain.
    assert "eyepacs" in run_retfound_probe.ALL_DOMAINS


def test_a_partial_weight_load_is_refused(tmp_path):
    """Loading a fraction of a ViT-L must fail loudly, not leave it half random."""
    from torch import nn

    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8))
    # A checkpoint holding only the first layer: far below the threshold.
    partial = {"0.weight": torch.zeros(8, 8), "0.bias": torch.zeros(8)}
    checkpoint = tmp_path / "partial.pth"
    torch.save({"model": partial}, checkpoint)

    with pytest.raises(RuntimeError, match="randomly initialised"):
        load_retfound_weights(model, checkpoint)


def test_a_complete_weight_load_reports_what_landed(tmp_path):
    from torch import nn

    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    complete = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    checkpoint = tmp_path / "complete.pth"
    torch.save({"model": complete}, checkpoint)

    report = load_retfound_weights(model, checkpoint)
    assert report["fraction_loaded"] >= MIN_LOADED_FRACTION
    assert report["keys_loaded"] == len(complete)
    for name, value in model.state_dict().items():
        assert torch.allclose(value, complete[name]), f"{name} was not applied"


def test_the_mae_decoder_is_discarded(tmp_path):
    """The decoder is not part of the feature extractor and must not block the load."""
    from torch import nn

    model = nn.Sequential(nn.Linear(8, 8))
    state = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    state.update({"decoder_embed.weight": torch.randn(4, 4),
                  "decoder_blocks.0.norm1.weight": torch.randn(4),
                  "mask_token": torch.randn(1, 1, 4)})
    checkpoint = tmp_path / "mae.pth"
    torch.save({"model": state}, checkpoint)

    report = load_retfound_weights(model, checkpoint)
    assert report["fraction_loaded"] == 1.0


def test_the_registry_entry_does_not_pull_timm_weights():
    """ImageNet weights would mask a failed RETFound load with a working model."""
    from src.models.backbones import SUPPORTED_BACKBONES

    spec = SUPPORTED_BACKBONES["retfound_cfp"]
    assert spec["timm_pretrained"] is False
    assert spec["weight_loader"] == "retfound"
