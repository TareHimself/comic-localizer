# Adapted from https://github.com/nipponjo/deepfillv2-pytorch
from comic_localizer.cleaning.patched_ai_cleaner import PatchedAiCleaner


class DeepFillV2Cleaner(PatchedAiCleaner):
    """Cleans using Free-Form Image Inpainting with Gated Convolution https://arxiv.org/abs/1806.03589"""

    @staticmethod
    def get_name() -> str:
        return "DeepFillV2"

    @staticmethod
    def is_valid() -> bool:
        return True
