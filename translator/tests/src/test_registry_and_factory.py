"""Tests for plugin registry and config-driven pipeline construction behavior."""

import yaml
import pytest
import inspect

from manga_translator.cleaning.all_white_cleaner import AllWhiteCleaner
from manga_translator.get import (
    construct_image_to_image_pipeline_from_config,
    construct_plugin_by_name,
    get_all,
)
from manga_translator.ocr.debug import DebugOCR
from manga_translator.translation.pipe import PipeTranslator


DEBUG_OCR_TEXT = "detected text"
ENGLISH_NAME = "English"
ENGLISH_TAG = "en"
PIPELINE_DEBUG_TEXT = "hello"


def test_get_all_exposes_expected_plugin_classes():
    """Global registry should contain representative classes from core plugin groups."""
    all_plugins = get_all()

    assert AllWhiteCleaner in all_plugins.values()
    assert DebugOCR in all_plugins.values()
    assert PipeTranslator in all_plugins.values()


def test_plugin_arguments_exist_in_constructor():
    """All plugin arguments should exist in the constructor"""
    all_plugins = get_all()

    for plugin in all_plugins.values():
        signature = inspect.signature(plugin.__init__)
        for argument in plugin.get_arguments():
            assert argument.id in signature.parameters, f"Argument id [{argument.id}] not in {plugin.__name__}.__init__"


def test_construct_plugin_by_name_builds_plugin_with_converted_args():
    """Factory should apply plugin argument converters (human language name to tag)."""
    plugin = construct_plugin_by_name(
        "DebugOCR",
        {
            "text": DEBUG_OCR_TEXT,
            "language": ENGLISH_NAME,
        },
    )

    assert isinstance(plugin, DebugOCR)
    assert plugin.to_write == DEBUG_OCR_TEXT
    assert plugin.language == ENGLISH_TAG


def test_construct_plugin_by_name_raises_for_unknown_class():
    """Factory should fail loudly for unknown class names to avoid silent misconfiguration."""
    with pytest.raises(KeyError):
        construct_plugin_by_name("MissingPlugin", {})


def test_construct_image_to_image_pipeline_from_config_builds_selected_plugins(
    tmp_path,
):
    """Config builder should wire explicit classes and keep defaults when requested."""
    config = {
        "pipeline": {
            "cleaner": {"class": "AllWhiteCleaner"},
            "ocr": {
                "class": "DebugOCR",
                "args": {"text": PIPELINE_DEBUG_TEXT, "language": ENGLISH_NAME},
            },
            "translator": {"class": "PipeTranslator", "args": None},
            "detector": {"class": "Default"},
            "segmenter": {"class": "Default"},
            "drawer": {"class": "Default"},
            "color_detector": {"class": "Default"},
        }
    }

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    pipeline = construct_image_to_image_pipeline_from_config(str(config_path))

    assert isinstance(pipeline.cleaner, AllWhiteCleaner)
    assert isinstance(pipeline.ocr, DebugOCR)
    assert pipeline.ocr.to_write == PIPELINE_DEBUG_TEXT
    assert pipeline.ocr.language == ENGLISH_TAG
    assert isinstance(pipeline.translator, PipeTranslator)
