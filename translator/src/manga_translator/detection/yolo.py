import numpy as np

from manga_translator.core.constants import DetectionType
from manga_translator.core.plugin import (
    Detector,
    DetectionResult,
    PluginArgument,
    StringPluginArgument,
    PytorchDevicePluginArgument,
)
from ultralytics import YOLO
import torch
import asyncio
from manga_translator.utils import get_default_torch_device


class YoloDetector(Detector):
    def __init__(
        self,
        model_path: str,
        device: torch.device = get_default_torch_device(),
        confidence=0.3,
        iou=0.3,
        skip_free_text=True,
    ):

        super().__init__()
        self.model = YOLO(model=model_path, verbose=False)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.skip_free_text = skip_free_text

    @staticmethod
    def conv_cls(cls_id: int):
        try:
            return DetectionType(cls_id)
        except ValueError:
            return DetectionType.TextOnPage

    def predict(self, batch: list[np.ndarray]):
        with torch.inference_mode():
            results = []
            for prediction in self.model.predict(
                [
                    x[..., ::-1] for x in batch
                ],  # flip RGB to BGR since ultralytics expects BGR
                device=self.device,
                verbose=False,
                conf=self.confidence,
                iou=self.iou,
            ):
                result = []
                if prediction.boxes is None:
                    results.append(result)
                    continue
                boxes = prediction.boxes.xyxy.cpu().int().numpy()  # type: ignore[union-attr]
                classes = prediction.boxes.cls.cpu().int()  # type: ignore[union-attr]
                confidence = prediction.boxes.conf.cpu()  # type: ignore[union-attr]

                for bbox, cls, conf in zip(boxes, classes, confidence):
                    # depending on model accuracy I have noticed some text bubbles are detected as free text, will need to do custom nms to catch this since we trust text_bubble more than text_free
                    if cls > 0 and self.skip_free_text:
                        continue
                    result.append(
                        DetectionResult(
                            YoloDetector.conv_cls(int(cls.item())), bbox, conf.item()
                        )
                    )

                results.append(result)

            return results

    async def detect(self, frames: list[np.ndarray]) -> list[list[DetectionResult]]:
        return await asyncio.to_thread(self.predict, frames)

    @staticmethod
    def get_name() -> str:
        return "Yolo"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument("model_path", "Model Path", "Path to the yolo model"),
            PytorchDevicePluginArgument("device", "Device"),
        ]
