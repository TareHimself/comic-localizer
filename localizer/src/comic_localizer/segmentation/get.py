from typing import Sequence
from comic_localizer.core.plugin import Segmenter
from comic_localizer.segmentation.yolo import YoloSegmenter

_segmentation_data = list(
    filter(
        lambda a: a.is_valid(),
        [YoloSegmenter],
    )
)


def get_segmenters() -> Sequence[type[Segmenter]]:
    return _segmentation_data
