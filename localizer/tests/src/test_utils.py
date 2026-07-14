"""Utility function tests focused on layout, color, image, and torch helper behavior."""

import numpy as np
import pytest
import torch

from comic_localizer import utils


WHITE_PIXEL = np.array([255, 255, 255], dtype=np.uint8)
BLACK_PIXEL = np.array([0, 0, 0], dtype=np.uint8)
FRENCH_LANGUAGE_NAME = "French"
FRENCH_LANGUAGE_TAG = "fr"
CPU_DEVICE_NAME = "cpu"
CPU_DEVICE_LABEL = "CPU"
CUDA_DEVICE_NAME = "cuda"
MPS_DEVICE_NAME = "mps"
DEFAULT_LANGUAGE_TAG = "en-US"


class FakeFont:
    def __init__(self, scale=6):
        self.scale = scale

    def getbbox(self, text: str):
        width = max(1, len(text)) * self.scale
        return (0, 0, width, 10)


def test_find_next_test_returns_midpoint_floor():
    """Binary-search helper should return integer midpoint using floor behavior."""
    assert utils.find_next_test(10, 21) == 15


def test_inverse_luminance_color_returns_complementary_lightness():
    """Inverse luminance should map pure white to pure black."""
    rgb = WHITE_PIXEL
    inv = utils.inverse_luminance_color(rgb)
    assert np.array_equal(inv, BLACK_PIXEL)


def test_inverse_luminance_color_validates_shape_and_dtype():
    """Input validation should reject non-uint8 and non-1D RGB arrays."""
    with pytest.raises(ValueError):
        utils.inverse_luminance_color(np.array([1, 2, 3], dtype=np.int32))

    with pytest.raises(ValueError):
        utils.inverse_luminance_color(np.array([[1, 2, 3]], dtype=np.uint8))


def test_wrap_text_pure_returns_none_when_word_too_wide():
    """Text wrapping should fail fast when any token cannot fit wrap width."""
    result = utils.wrap_text_pure("superlongword", FakeFont(scale=20), wrap_width=30)
    assert result is None


def test_wrap_text_pure_wraps_lines_and_tracks_bounds():
    """Pure wrapping should produce multiple lines and bounded width metrics."""
    result = utils.wrap_text_pure("one two three", FakeFont(scale=5), wrap_width=30)

    assert result is not None
    assert len(result.lines) >= 2
    assert result.bounds[0] <= 30


def test_wrap_text_dispatches_to_hyphenator_variant(monkeypatch):
    """wrap_text should delegate to hyphenator path when hyphenator is provided."""
    sentinel = object()

    def fake_hyphen(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(utils, "wrap_text_with_hyphenator", fake_hyphen)

    out = utils.wrap_text("hello", FakeFont(), wrap_width=100, hyphenator=object())
    assert out is sentinel


def test_wrap_text_dispatches_to_pure_without_hyphenator(monkeypatch):
    """wrap_text should delegate to pure path when hyphenator is not provided."""
    sentinel = object()

    def fake_pure(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(utils, "wrap_text_pure", fake_pure)

    out = utils.wrap_text("hello", FakeFont(), wrap_width=100, hyphenator=None)
    assert out is sentinel


def test_ensure_gray_converts_color_and_copies_grayscale():
    """ensure_gray should convert color input and copy grayscale input."""
    color = np.zeros((4, 4, 3), dtype=np.uint8)
    gray = utils.ensure_gray(color)
    assert gray.shape == (4, 4)

    original = np.zeros((4, 4), dtype=np.uint8)
    copied = utils.ensure_gray(original)
    assert np.array_equal(copied, original)
    assert copied is not original


def test_ensure_gray_uses_rgb_channel_order():
    """ensure_gray must weight the R channel (not B) as the dominant luminance
    contributor, since arrays flowing through the pipeline are RGB, not BGR."""
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[..., 0] = 255  # pure red in RGB order

    gray = utils.ensure_gray(red)

    # cv2.COLOR_RGB2GRAY weights: 0.299*R + 0.587*G + 0.114*B -> ~76 for pure red.
    # The old (buggy) COLOR_BGR2GRAY misreads this as pure blue -> ~29.
    assert gray[0, 0] == pytest.approx(76, abs=1)


def test_compute_draw_bbox_returns_full_when_no_contours():
    """When no bright contour exists, draw bbox should default to full section bounds."""
    section = np.zeros((8, 10, 3), dtype=np.uint8)
    bbox = utils.compute_draw_bbox(section)
    assert np.array_equal(bbox, np.array([0, 0, 10, 8], dtype=np.int32))


def test_natural_sort_key_orders_numeric_suffixes_numerically():
    """Plain string sort would put page10 before page2; natural sort must not."""
    names = ["page10.png", "page1.png", "page2.png"]
    assert sorted(names, key=utils.natural_sort_key) == [
        "page1.png",
        "page2.png",
        "page10.png",
    ]


def test_has_white_detects_white_pixels_presence():
    """White-pixel detector should distinguish threshold-matching and empty images."""
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[1, 1] = WHITE_PIXEL
    assert utils.has_white(image) is True

    no_white = np.zeros((3, 3, 3), dtype=np.uint8)
    assert utils.has_white(no_white) is False


def test_get_available_pytorch_devices_cpu_only(monkeypatch):
    """Device listing should always include CPU when accelerators are unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    devices = utils.get_available_pytorch_devices()
    assert devices == [(CPU_DEVICE_NAME, CPU_DEVICE_LABEL)]


def test_get_default_torch_device_prefers_cuda_then_mps(monkeypatch):
    """Default torch device preference should be CUDA, then MPS, then CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert utils.get_default_torch_device().type == CUDA_DEVICE_NAME

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: True)
    assert utils.get_default_torch_device().type == MPS_DEVICE_NAME

    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    assert utils.get_default_torch_device().type == CPU_DEVICE_NAME


def test_get_autocast_returns_nullcontext_when_disabled_or_unsupported():
    """Autocast should become a nullcontext when disabled or for unsupported device."""
    from contextlib import nullcontext

    cpu = torch.device("cpu")
    assert isinstance(utils.get_autocast(cpu, enabled=False), nullcontext)

    meta = torch.device("meta")
    assert isinstance(utils.get_autocast(meta, enabled=True), nullcontext)


def test_standardize_language_code_uses_find_when_get_raises(monkeypatch):
    """Language standardization should fall back to find() for human language names."""

    class FakeLanguageFacade:
        @staticmethod
        def get(_value):
            raise utils.langcodes.LanguageTagError("bad")

        @staticmethod
        def find(_value):
            class Tag:
                def to_tag(self):
                    return FRENCH_LANGUAGE_TAG

            return Tag()

    monkeypatch.setattr(utils.langcodes, "Language", FakeLanguageFacade)

    assert utils.standardize_language_code(FRENCH_LANGUAGE_NAME) == FRENCH_LANGUAGE_TAG


def test_default_language_and_perf_toggle_helpers():
    """Default language and perf toggles should be callable and deterministic."""
    assert utils.get_default_language() == DEFAULT_LANGUAGE_TAG

    utils.enable_perf()
    utils.disable_perf()
