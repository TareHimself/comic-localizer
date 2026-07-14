"""Tests for OpenAiOCR image encoding and result-ordering behavior."""

import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from comic_localizer.ocr.openai import OpenAiOCR


def _make_ocr() -> OpenAiOCR:
    return OpenAiOCR(api_key="test-key")


def _fake_response(results):
    return SimpleNamespace(output_parsed=SimpleNamespace(results=results))


def _result(index: int, text: str, language: str = "en"):
    return SimpleNamespace(index=index, text=text, language=language)


def test_opencv_image_to_b64_preserves_rgb_color_order():
    """Encoding must convert RGB input to BGR before cv2.imencode so the
    resulting PNG round-trips back to the original RGB colors."""
    ocr = _make_ocr()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:] = (200, 30, 10)  # distinct R, G, B values in RGB order

    data_uri = ocr.opencv_image_to_b64(image)

    encoded_b64 = data_uri.split(",", 1)[1]
    encoded_bytes = base64.b64decode(encoded_b64)
    decoded_bgr = cv2.imdecode(
        np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    assert np.array_equal(decoded_rgb[0, 0], image[0, 0])


def test_do_ocr_places_results_by_index_even_when_returned_out_of_order(monkeypatch):
    """If the model returns results in a different order than requested, they
    must still land on the correct image based on the parsed `index` field."""
    ocr = _make_ocr()
    batch = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]

    shuffled = [_result(2, "third"), _result(0, "first"), _result(1, "second")]
    monkeypatch.setattr(
        ocr.openai.responses, "parse", lambda **kwargs: _fake_response(shuffled)
    )

    results = ocr.do_ocr(batch)

    assert [r.text for r in results] == ["first", "second", "third"]


def test_do_ocr_raises_on_duplicate_index(monkeypatch):
    """Duplicate indices must raise instead of silently overwriting a slot and
    leaving another slot unfilled."""
    ocr = _make_ocr()
    batch = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]

    duplicated = [_result(0, "a"), _result(0, "b")]
    monkeypatch.setattr(
        ocr.openai.responses, "parse", lambda **kwargs: _fake_response(duplicated)
    )

    with pytest.raises(RuntimeError):
        ocr.do_ocr(batch)


def test_do_ocr_raises_on_out_of_range_index(monkeypatch):
    """An index outside 0..K-1 must raise instead of crashing on list assignment
    or silently corrupting results."""
    ocr = _make_ocr()
    batch = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]

    out_of_range = [_result(0, "a"), _result(5, "b")]
    monkeypatch.setattr(
        ocr.openai.responses, "parse", lambda **kwargs: _fake_response(out_of_range)
    )

    with pytest.raises(RuntimeError):
        ocr.do_ocr(batch)
