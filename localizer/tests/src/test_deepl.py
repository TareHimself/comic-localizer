"""Tests for DeepLTranslator target_lang handling."""

from comic_localizer.translation.deepl import DeepLTranslator, _to_deepl_target_lang


def test_to_deepl_target_lang_maps_bare_english_to_region_qualified():
    """DeepL rejects/deprecates bare 'EN' for target_lang; must fall back to a
    region-qualified variant (the client itself uppercases before the request)."""
    assert _to_deepl_target_lang("en").upper() == "EN-US"


def test_to_deepl_target_lang_maps_bare_portuguese_to_region_qualified():
    """DeepL rejects/deprecates bare 'PT' for target_lang; must fall back to a
    region-qualified variant (the client itself uppercases before the request)."""
    assert _to_deepl_target_lang("pt").upper() == "PT-BR"


def test_to_deepl_target_lang_preserves_explicit_region():
    """A code that already specifies a region should only be uppercased, not remapped."""
    assert _to_deepl_target_lang("en-GB") == "EN-GB"


def test_to_deepl_target_lang_uppercases_codes_without_a_fallback():
    """Languages with no known DeepL ambiguity should just be uppercased."""
    assert _to_deepl_target_lang("fr") == "FR"


def test_deepl_translator_stores_fallback_target_lang_separately_from_language():
    """self.language (reported downstream) must stay a plain standardized tag,
    while self.deepl_target_lang carries the DeepL-API-specific variant."""
    translator = DeepLTranslator(auth_key="test-key", language="en")

    assert translator.language == "en"
    assert translator.deepl_target_lang.upper() == "EN-US"
