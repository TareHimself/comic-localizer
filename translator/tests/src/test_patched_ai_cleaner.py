"""Tests for PatchedAiCleaner mask polarity and non-patched compositing correctness."""

import numpy as np
import pytest
import torch

from manga_translator.cleaning.patched_ai_cleaner import PatchedAiCleaner
from manga_translator.core.constants import DetectionType, SegmentationType
from manga_translator.core.plugin import DetectionResult, SegmentationResult


class _ConstantInpaintModel:
    """Stand-in for a torch.jit inpainting model: ignores input, returns a
    constant-valued tensor of the same shape so output can be asserted on."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, batch: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return torch.full_like(batch, self.value)


def _make_cleaner(monkeypatch, model=None, **kwargs) -> PatchedAiCleaner:
    monkeypatch.setattr(
        PatchedAiCleaner, "load_model", lambda self, path, device: model
    )
    return PatchedAiCleaner(model_path="unused", **kwargs)


def test_extract_patches_mask_background_is_zero_not_one(monkeypatch):
    """The mask background outside the segment polygon must be 0 (keep), not 1,
    since process_masks_batched divides by 255 and 1 would leak a tiny nonzero
    signal into every 'keep this pixel' region fed to the inpainting model."""
    cleaner = _make_cleaner(monkeypatch)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    points = np.array([[5, 5], [15, 5], [15, 15], [5, 15]])
    segment = SegmentationResult(SegmentationType.Text, points, confidence=1.0)

    patches = cleaner.extract_patches([frame], [[segment]])

    assert len(patches) == 1
    mask = patches[0].mask
    assert mask[0, 0] == 0
    assert mask[9, 9] == 255


def test_clean_sync_without_patching_preserves_caller_frames_and_composites_correctly(
    monkeypatch,
):
    """With inpaint_patches=False, clean_sync must not mutate the caller's
    `frames` list, and must composite cleaned content inside detection boxes
    with the original content outside them (not the reverse)."""
    model = _ConstantInpaintModel(0.5)
    cleaner = _make_cleaner(monkeypatch, model=model, inpaint_patches=False)

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    original_frame = frame.copy()
    mask = np.zeros((16, 16), dtype=np.uint8)
    detection = DetectionResult(DetectionType.TextInBubble, (4, 4, 12, 12), 1.0)

    result = cleaner.clean_sync([frame], [mask], [[]], [[detection]])

    # Caller's original frame array must be untouched.
    assert np.array_equal(frame, original_frame)

    out = result[0]
    # Inside the detection box: cleaned (model constant ~127), not the original black.
    assert out[6, 6, 0] == pytest.approx(127, abs=2)
    # Outside the detection box: original content (black), not the cleaned constant.
    assert np.array_equal(out[0, 0], np.array([0, 0, 0], dtype=np.uint8))
