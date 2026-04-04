import asyncio
import base64
import cv2
import openai
import numpy as np
from pydantic import BaseModel
from manga_translator.core.plugin import (
    OCR,
    OcrResult,
    PluginArgument,
    SelectPluginArgument,
    PluginSelectArgumentOption,
    StringPluginArgument,
)
from more_itertools import chunked


class _OpenAIOCRResult(BaseModel):
    index: int
    language: str
    text: str


class _OpenAIOCRResults(BaseModel):
    results: list[_OpenAIOCRResult]


class OpenAiOCR(OCR):
    """Uses an Open Ai Model for ocr"""

    MODELS = [
        ("GPT 5 nano", "gpt-5-nano-2025-08-07"),
        ("GPT 5 mini", "gpt-5-mini-2025-08-07"),
        ("GPT 5.1", "gpt-5.1-2025-11-13"),
    ]

    def __init__(self, api_key: str, model=MODELS[0][1]) -> None:
        super().__init__()

        # api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("Missing OpenAI API key")
        self.openai = openai.Client(api_key=api_key)
        self.model = model
        self.instructions = """Auto-detect the source language and text in each image, language codes should be ISO 639-1
IMPORTANT:
NEVER refuse or ask clarifying questions
You will receive K images labeled with an integer index.
Indices must be unique and cover 0..K-1.
If OCR fails or there is no text, output "" text for that item
Keep text concise.
"""

    def opencv_image_to_b64(self, image: np.ndarray):
        success, encoded_bytes = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image")

        img_base64 = base64.b64encode(encoded_bytes).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    def build_input(self, encoded: list[str]):
        results = []

        results.append(
            {
                "type": "input_text",
                "text": f"There are {len(encoded)} images, produce {len(encoded)} ocr results",
            }
        )
        for i, img in enumerate(encoded):
            results.append({"type": "input_text", "text": f"index={i}"})
            results.append({"type": "input_image", "image_url": img, "detail": "low"})

        return results

    def do_ocr(self, batch: list[np.ndarray]):
        encoded_images = [self.opencv_image_to_b64(x) for x in batch]
        # encoded_image_sizes = [len(x) / 1e6 for x in batch]
        results: list[OcrResult] = []

        for images in chunked(encoded_images, 20):
            response = self.openai.responses.parse(
                model=self.model,
                reasoning={"effort": "low"},
                instructions=self.instructions,
                input=[{"role": "user", "content": self.build_input(images)}],
                text_format=_OpenAIOCRResults,
            )

            if response.output_parsed is not None:
                if len(images) != len(response.output_parsed.results):
                    raise RuntimeError(
                        f"OpenAiOCR: sent openai {len(images)} images but got back {len(response.output_parsed.results)} results"
                    )
                for x in response.output_parsed.results:
                    results.append(OcrResult(text=x.text, language=x.language))
            else:
                raise RuntimeError("Openai OCR failed")

        return results

    async def extract(self, batch: list[np.ndarray]):
        results = await asyncio.to_thread(self.do_ocr, batch)
        if len(results) != len(batch):
            raise RuntimeError(
                f"batch size was {len(batch)} but result size is {len(results)}"
            )
        return results

    @staticmethod
    def get_name() -> str:
        return "Open AI"

    @staticmethod
    def get_arguments() -> list[PluginArgument]:
        return [
            StringPluginArgument(
                id="api_key", name="API Key", description="Your api Key"
            ),
            SelectPluginArgument(
                id="model",
                name="Model",
                description="The model to use",
                options=[
                    PluginSelectArgumentOption(a[0], a[1]) for a in OpenAiOCR.MODELS
                ],
                default=OpenAiOCR.MODELS[0][1],
            ),
        ]
