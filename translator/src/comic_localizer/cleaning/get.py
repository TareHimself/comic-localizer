from typing import Sequence
from comic_localizer.core.plugin import Cleaner
from comic_localizer.cleaning.all_white_cleaner import AllWhiteCleaner
from comic_localizer.cleaning.opencv import OpenCvCleaner
from comic_localizer.cleaning.deepfillv2 import DeepFillV2Cleaner
from comic_localizer.cleaning.lama import LamaCleaner

_data = list(
    filter(
        lambda a: a.is_valid(),
        [AllWhiteCleaner, OpenCvCleaner, DeepFillV2Cleaner, LamaCleaner],
    )
)


def get_cleaners() -> Sequence[type[Cleaner]]:
    return _data
