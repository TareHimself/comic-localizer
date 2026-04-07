import asyncio
import base64
from more_itertools import chunked
import numpy as np
from manga_translator.core.plugin import (
    OCR,
    OcrResult,
    PluginArgument,
    StringPluginArgument,
)
import langcodes
import cv2
import httpx


class GoogleCloudOCR(OCR):
    """Uses google cloud to perform ocr , requires an API key"""

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            base_url="https://vision.googleapis.com/v1",
        )

    @staticmethod
    def opencv_image_to_b64(image: np.ndarray):
        success, encoded_bytes = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image")

        # google does not want the data.../... part
        return base64.b64encode(encoded_bytes).decode("utf-8")

    @staticmethod
    def opencv_image_to_b64_batch(batch: list[np.ndarray]):
        return [GoogleCloudOCR.opencv_image_to_b64(x) for x in batch]

    async def extract(self, batch: list[np.ndarray]):
        ocr_results: list[OcrResult] = []
        encoded_images = await asyncio.to_thread(
            GoogleCloudOCR.opencv_image_to_b64_batch, batch
        )
        # 16 is the cap set by google https://docs.cloud.google.com/vision/quotas
        for images in chunked(encoded_images, 16):
            response = await self.client.post(
                "/images:annotate",
                json={
                    "requests": [
                        (
                            {
                                "image": {"content": x},
                                "features": [{"type": "TEXT_DETECTION"}],
                            }
                        )
                        for x in images
                    ]
                },
            )

            response.raise_for_status()

            result = response.json()

            for response in result["responses"]:
                annotations = response.get("textAnnotations", None)
                if annotations is not None and len(annotations) > 0:
                    annotation = annotations[0]
                    text = annotation["description"]
                    ocr_results.append(OcrResult(text=text, language=annotation["locale"]))
                    continue

                ocr_results.append(OcrResult())

        return ocr_results

    @staticmethod
    def get_name() -> str:
        return "Google Cloud OCR"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="api_key", name="API Key", description="Google cloud Vision API key"
            )
        ]
