import asyncio

import cv2
import numpy as np

from comic_localizer.core.plugin import ColorDetectionResult, ColorDetector
from comic_localizer.utils import ensure_gray


class BasicColorDetector(ColorDetector):
    """Black text if the area is whitish else white text"""

    def __init__(self) -> None:
        super().__init__()

    def do_color_detection(
        self, cleaned: list[np.ndarray]
    ) -> list[ColorDetectionResult]:
        results = []
        for x in cleaned:
            ret, thresh = cv2.threshold(ensure_gray(x), 127, 255, 0)
            mostlyWhite = np.mean(thresh > 127) > 0.5
            if mostlyWhite:
                results.append(ColorDetectionResult(np.zeros((3), dtype=np.uint8), 1))
            else:
                results.append(
                    ColorDetectionResult(np.ones((3), dtype=np.uint8) * 255, 1)
                )

        return results

    async def detect_color(
        self,
        text: list[np.ndarray],
        cleaned: list[np.ndarray],
        original: list[np.ndarray],
    ) -> list[ColorDetectionResult]:
        results = await asyncio.to_thread(self.do_color_detection, cleaned)
        return results

    @staticmethod
    def get_name() -> str:
        return "Basic"
