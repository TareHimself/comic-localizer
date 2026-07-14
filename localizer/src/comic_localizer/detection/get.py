from typing import Sequence
from comic_localizer.core.plugin import Detector
from comic_localizer.detection.yolo import YoloDetector

_detection_data = list(
    filter(
        lambda a: a.is_valid(),
        [YoloDetector],
    )
)


def get_detectors() -> Sequence[type[Detector]]:
    return _detection_data
