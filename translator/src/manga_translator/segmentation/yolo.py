from manga_translator.core.constants import SegmentationType
from manga_translator.core.plugin import (
    PluginArgument,
    PytorchDevicePluginArgument,
    StringPluginArgument,
    Segmenter,
    SegmentationResult,
)
from ultralytics import YOLO
import torch
import asyncio
from manga_translator.utils import get_default_torch_device


class YoloSegmenter(Segmenter):
    def __init__(
        self, model_path: str, device: torch.device = get_default_torch_device()
    ):

        super().__init__()
        self.model = YOLO(model=model_path, verbose=False)
        self.device = device

    @staticmethod
    def conv_cls(cls: int):
        try:
            return SegmentationType(cls)
        except ValueError:
            return SegmentationType.Text

    def predict(self, batch):
        with torch.inference_mode():
            results = []
            for prediction in self.model.predict(
                # TODO  RGB to BGR since model.predic expects BGR
                batch,  # [x[..., ::-1] for x in batch],
                conf=0.1,
                device=self.device,
                verbose=False,
            ):
                result = []

                if prediction.masks is not None:
                    classes = prediction.boxes.cls.cpu().int()
                    confidence = prediction.boxes.conf.cpu()
                    masks = prediction.masks.xy

                    for mask, cls, conf in zip(masks, classes, confidence):
                        result.append(
                            SegmentationResult(
                                YoloSegmenter.conv_cls(cls.item()),
                                mask.astype(int),
                                conf.item(),
                            )
                        )

                results.append(result)

            return results

    async def segment(self, batch):
        return await asyncio.to_thread(self.predict, batch)

    @staticmethod
    def get_name() -> str:
        return "Yolo"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument("model_path", "Model Path", "Path to the yolo model"),
            PytorchDevicePluginArgument("device", "Device"),
        ]
