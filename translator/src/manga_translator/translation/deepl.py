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
import langcodes


class DeepLTranslator(Translator):
    """The Best after GPT but it requires an auth token from here https://www.deepl.com/translator"""

    def __init__(self, auth_key=None, target_language: str = "en-US") -> None:
        super().__init__()
        self.client = deepl.DeepLClient(auth_key)
        self.target_language = langcodes.standardize_tag(target_language)

    def do_api(self, batch: list[OcrResult]):
        results = self.client.translate_text(
            [x.text for x in batch],
            target_lang=self.target_language.upper(),
        )

        return [TranslatorResult(text=x.text) for x in results]

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
                id="target_language",
                name="Target Language",
                description="The language to translate to (confirm support here https://developers.deepl.com/docs/getting-started/supported-languages)",
            ),
        ]
