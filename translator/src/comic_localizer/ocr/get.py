from typing import Sequence
from comic_localizer.core.plugin import OCR
from comic_localizer.ocr.debug import DebugOCR
from comic_localizer.ocr.huggingface import HuggingFaceOCR
from comic_localizer.ocr.manga_ocr import MangaOCR
from comic_localizer.ocr.openai import OpenAiOCR
from comic_localizer.ocr.minstral import MinstralOCR
from comic_localizer.ocr.google_cloud import GoogleCloudOCR

_ocr_data = list(
    filter(
        lambda a: a.is_valid(),
        [DebugOCR, HuggingFaceOCR, MangaOCR, OpenAiOCR, MinstralOCR, GoogleCloudOCR],
    )
)


def get_ocrs() -> Sequence[type[OCR]]:
    return _ocr_data
