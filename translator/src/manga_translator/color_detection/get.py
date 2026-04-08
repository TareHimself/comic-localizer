from manga_translator.color_detection.basic import BasicColorDetector
from manga_translator.core.plugin import ColorDetector
from manga_translator.color_detection.openai import OpenAiColorDetector

_color_detection_data = list(
    filter(
        lambda a: a.is_valid(),
        [BasicColorDetector, OpenAiColorDetector],
    )
)


def get_color_detectors() -> list[ColorDetector]:
    return _color_detection_data
