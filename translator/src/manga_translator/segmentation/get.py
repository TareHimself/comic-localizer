from typing import Sequence
from manga_translator.core.plugin import Segmenter
from manga_translator.segmentation.yolo import YoloSegmenter

_segmentation_data = list(
    filter(
        lambda a: a.is_valid(),
        [YoloSegmenter],
    )
)


def get_segmenters() -> Sequence[type[Segmenter]]:
    return _segmentation_data
