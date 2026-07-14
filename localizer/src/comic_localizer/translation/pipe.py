from comic_localizer.core.plugin import Translator, TranslatorResult, OcrResult


class PipeTranslator(Translator):
    """Outputs OCR result"""

    def __init__(self) -> None:
        super().__init__()

    async def translate(self, batch: list[OcrResult]):
        return [TranslatorResult(a.text, language=a.language) for a in batch]

    @staticmethod
    def get_name() -> str:
        return "Pipe"
