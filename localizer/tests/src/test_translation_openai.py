"""Tests for OpenAiTranslator length validation on parsed translations."""

from types import SimpleNamespace

import pytest

from comic_localizer.core.plugin import OcrResult
from comic_localizer.translation.openai import OpenAiTranslator


def _make_translator() -> OpenAiTranslator:
    return OpenAiTranslator(api_key="test-key", language="fr")


def _fake_response(translations):
    return SimpleNamespace(output_parsed=SimpleNamespace(translations=translations))


def test_do_translation_raises_when_fewer_translations_returned(monkeypatch):
    """If the model returns fewer translations than requested, the mismatch
    must raise instead of silently leaving the extra slots as empty text."""
    translator = _make_translator()
    batch = [OcrResult("one", "en"), OcrResult("two", "en")]

    monkeypatch.setattr(
        translator.openai.responses,
        "parse",
        lambda **kwargs: _fake_response(["only one"]),
    )

    with pytest.raises(RuntimeError):
        translator.do_translation(batch)


def test_do_translation_assigns_translations_when_counts_match(monkeypatch):
    """Sanity check that the matching-count path still assigns correctly."""
    translator = _make_translator()
    batch = [OcrResult("one", "en"), OcrResult("two", "en")]

    monkeypatch.setattr(
        translator.openai.responses,
        "parse",
        lambda **kwargs: _fake_response(["un", "deux"]),
    )

    result = translator.do_translation(batch)

    assert [r.text for r in result] == ["un", "deux"]
