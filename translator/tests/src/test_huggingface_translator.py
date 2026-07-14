"""Tests for HuggingFaceTranslator src_lang/tgt_lang gating logic.

These tests monkeypatch transformers.pipeline entirely so they don't download
or load a real model, and don't depend on the installed transformers version
supporting the "translation" task.
"""

from types import SimpleNamespace

import torch

from manga_translator.translation import huggingface as huggingface_module
from manga_translator.core.plugin import OcrResult


class _FakeTokenizerWithLangPair:
    def _build_translation_inputs(self, *args, **kwargs):
        raise NotImplementedError


class _FakeTokenizerWithoutLangPair:
    pass


class _FakePipeline:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        return [{"translation_text": f"out:{t}"} for t in texts]


def _make_translator(monkeypatch, tokenizer):
    fake_pipeline = _FakePipeline(tokenizer)
    monkeypatch.setattr(
        huggingface_module, "pipeline", lambda *a, **k: fake_pipeline
    )
    translator = huggingface_module.HuggingFaceTranslator(
        model_url="fake-model", input_language="ja", output_language="en"
    )
    return translator, fake_pipeline


def test_predict_omits_lang_kwargs_for_marian_style_tokenizer(monkeypatch):
    """MarianTokenizer (the default model's tokenizer) has no
    _build_translation_inputs, so src_lang/tgt_lang must not be passed."""
    translator, fake_pipeline = _make_translator(
        monkeypatch, _FakeTokenizerWithoutLangPair()
    )

    with torch.inference_mode():
        translator.predict([OcrResult("hello", "ja")])

    assert fake_pipeline.calls == [{}]


def test_predict_includes_lang_kwargs_for_multilingual_tokenizer(monkeypatch):
    """Multilingual tokenizers (mBART/M2M100/NLLB-style) that define
    _build_translation_inputs should receive src_lang/tgt_lang."""
    translator, fake_pipeline = _make_translator(
        monkeypatch, _FakeTokenizerWithLangPair()
    )

    with torch.inference_mode():
        translator.predict([OcrResult("hello", "ja")])

    assert fake_pipeline.calls == [{"src_lang": "ja", "tgt_lang": "en"}]
