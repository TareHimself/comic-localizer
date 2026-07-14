from comic_localizer.pipelines.image_to_image import ImageToImagePipeline
from comic_localizer.core.plugin import ColorDetector
from comic_localizer.cleaning.opencv import OpenCvCleaner
from comic_localizer.detection.yolo import YoloDetector
from comic_localizer.segmentation.yolo import YoloSegmenter
from comic_localizer.translation.deepl import DeepLTranslator
from comic_localizer.ocr.manga_ocr import MangaOCR
from comic_localizer.drawing.horizontal import HorizontalDrawer
import asyncio
import cv2
import torch


async def main():
    device = torch.device("cpu")
    pipeline = ImageToImagePipeline(
        translator=DeepLTranslator(auth_key="Some deepl api key"),
        detector=YoloDetector(r"Some yolo detector path", device=device),
        segmenter=YoloSegmenter(r"Some yolo segmenter path", device=device),
        cleaner=OpenCvCleaner(),  # AllWhiteCleaner(),
        drawer=HorizontalDrawer(font_file=r"Some Font File"),
        ocr=MangaOCR(device=device),  # MangaOcr(device=torch.device('cuda:0')),
        color_detector=ColorDetector(),
    )
    image = cv2.imread(r"./file.png", cv2.IMREAD_COLOR_RGB)
    result = (await pipeline([image]))[0]
    cv2.imshow("Translated", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    asyncio.run(main())
