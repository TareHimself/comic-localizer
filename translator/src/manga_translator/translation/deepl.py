import asyncio
from manga_translator.core.plugin import (
    LanguageStringArgument,
    Translator,
    OcrResult,
    TranslatorResult,
    PluginArgument,
    StringPluginArgument,
)
import deepl

from manga_translator.utils import get_default_language, standardize_language_code


# DeepL's API rejects/deprecates bare region-less codes for some target
# languages (e.g. "EN", "PT") and requires a region-qualified variant.
_DEEPL_TARGET_LANG_FALLBACKS = {
    "en": deepl.Language.ENGLISH_AMERICAN,  # "en-US"
    "pt": deepl.Language.PORTUGUESE_BRAZILIAN,  # "pt-BR"
}


def _to_deepl_target_lang(language_code: str) -> str:
    if "-" in language_code:
        return language_code.upper()
    return _DEEPL_TARGET_LANG_FALLBACKS.get(language_code.lower(), language_code.upper())


class DeepLTranslator(Translator):
    """The Best after GPT but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_key=None, language: str = get_default_language()) -> None:
        super().__init__()
        self.client = deepl.DeepLClient(auth_key)
        self.language = standardize_language_code(language)
        self.deepl_target_lang = _to_deepl_target_lang(self.language)

    def do_api(self, batch: list[OcrResult]):
        results = self.client.translate_text(
            [x.text for x in batch],
            target_lang=self.deepl_target_lang,
        )

        return [TranslatorResult(text=x.text, language=self.language) for x in results] # type: ignore

    async def translate(self, batch: list[OcrResult]):
        return await asyncio.to_thread(self.do_api, batch)

    @staticmethod
    def get_name() -> str:
        return "DeepL"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="auth_key", name="Auth Token", description="DeepL Api Auth Key"
            ),
            LanguageStringArgument(
                id="language",
                name="Target Language",
                description="The language to translate to (confirm support here https://developers.deepl.com/docs/getting-started/supported-languages)",
            ),
        ]
