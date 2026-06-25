from typing import Sequence
from manga_translator.core.plugin import OCR
from manga_translator.ocr.debug import DebugOCR
from manga_translator.ocr.huggingface import HuggingFaceOCR
from manga_translator.ocr.manga_ocr import MangaOCR
from manga_translator.ocr.openai import OpenAiOCR
from manga_translator.ocr.minstral import MinstralOCR
from manga_translator.ocr.google_cloud import GoogleCloudOCR

_ocr_data = list(
    filter(
        lambda a: a.is_valid(),
        [DebugOCR, HuggingFaceOCR, MangaOCR, OpenAiOCR, MinstralOCR, GoogleCloudOCR],
    )
)


def get_ocrs() -> Sequence[type[OCR]]:
    return _ocr_data
