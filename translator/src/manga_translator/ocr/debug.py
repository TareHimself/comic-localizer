import numpy as np
from manga_translator.core.plugin import (
    LanguageStringArgument,
    OCR,
    OcrResult,
    StringPluginArgument,
    PluginArgument,
)
from manga_translator.utils import get_default_language


class DebugOCR(OCR):
    """Outputs the specified text"""

    def __init__(self, text="", language=get_default_language()) -> None:
        super().__init__()
        self.to_write = text
        self.language = language

    async def extract(self, batch: list[np.ndarray]):
        return [OcrResult(text=self.to_write, language=self.language) for _ in batch]

    @staticmethod
    def get_name() -> str:
        return "Debug"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="text", name="Debug Text", description="What to output"
            ),
            LanguageStringArgument(
                id="language",
                name="Output Language",
                description="The language the output text is",
            ),
        ]
