from comic_localizer.core.plugin import (
    Translator,
    TranslatorResult,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
    LanguageStringArgument,
)
from comic_localizer.utils import get_default_language


class DebugTranslator(Translator):
    """Writes the specified text"""

    def __init__(self, text="", language=get_default_language()) -> None:
        super().__init__()
        self.to_write = text
        self.language = language

    async def translate(self, batch: list[OcrResult]):
        return [TranslatorResult(self.to_write, language=self.language) for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "Custom Text"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="text", name="Debug Text", description="What to write"
            ),
            LanguageStringArgument(
                id="language",
                name="Output Language",
                description="The language the output text is",
            ),
        ]
