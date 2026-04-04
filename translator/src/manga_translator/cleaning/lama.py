# Adapted from https://github.com/enesmsahin/simple-lama-inpainting , maybe setup our own repo in the future since I can't make sense of the original lama repo
from manga_translator.cleaning.patched_ai_cleaner import PatchedAiCleaner


class LamaCleaner(PatchedAiCleaner):
    """
    Cleans using LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions https://arxiv.org/abs/2109.07161
    """

    @staticmethod
    def get_name() -> str:
        return "Lama"

    @staticmethod
    def is_valid() -> bool:
        return True
