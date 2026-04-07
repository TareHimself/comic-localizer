
from dotenv import load_dotenv

import numpy as np
import os
import io
import urllib.parse
import requests
import cv2
import asyncio
import re
import webbrowser
import traceback
from dataclasses import dataclass, field
from blacksheep import (
    Application,
    Content,
    FormPart,
    Response,
    file,
    get,
    post,
    ContentDispositionType,
    json,
    FromForm,
)
from manga_translator.utils import pil_to_cv2
from manga_translator.pipelines.image_to_image import ImageToImagePipeline
from manga_translator.get import construct_plugin_by_name
from manga_translator.translation.get import get_translators
from manga_translator.ocr.get import get_ocrs
from manga_translator.drawing.get import get_drawers
from manga_translator.detection.get import get_detectors
from manga_translator.segmentation.get import get_segmenters
from manga_translator.cleaning.get import get_cleaners
from PIL import Image
from dataclass_wizard import JSONWizard
from typing import Any

load_dotenv()

app = Application()
build_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
app.serve_files(source_folder=build_path, discovery=True)

default_component = {"id": 0, "args": {}}


def cv2_image_from_url(url: str):
    if url.startswith("http"):
        return pil_to_cv2(Image.open(io.BytesIO(requests.get(url).content)))
    else:
        sanitized = urllib.parse.unquote(url.split("?")[0])
        data = cv2.imread(sanitized)

        if data is None:
            raise BaseException(f"Failed to load image from path {url}")
        return data


REQUEST_SECTION_REGEX = r"id=([0-9]+)(.*)"
REQUEST_SECTION_PARAMS_REGEX = r"\$([a-z0-9_]+)=([^\/$]+)"


def extract_params(data: str) -> tuple[int, dict[str, str]]:
    selected_id, params_to_parse = re.findall(REQUEST_SECTION_REGEX, data)[0]
    params = {}

    if len(params_to_parse.strip()) > 0:
        for param_name, param_value in re.findall(
            REQUEST_SECTION_PARAMS_REGEX, params_to_parse.strip()
        ):
            if len(param_value.strip()) > 0:
                params[param_name] = param_value

    return int(selected_id), params


def bytes_to_mat(data: bytes):
    array = np.asarray(bytearray(data), dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR_BGR)


def mat_to_bytes(data: np.ndarray):
    success, encoded_bytes = cv2.imencode(".png", data)
    return encoded_bytes.tobytes()


@dataclass
class ServerComponentDataclass(JSONWizard):
    id: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerPayloadDataclass(JSONWizard):
    detector: ServerComponentDataclass
    segmenter: ServerComponentDataclass
    translator: ServerComponentDataclass
    ocr: ServerComponentDataclass
    drawer: ServerComponentDataclass
    cleaner: ServerComponentDataclass


@dataclass
class RequestFormDataclass:
    data: str
    file: list[FormPart]


@post("/clean")
async def clean(data: FromForm[RequestFormDataclass]):
    try:
        image = data.value.file[0].data

        data_object = ServerPayloadDataclass.from_json(data.value.data)

        image_mat = await asyncio.to_thread(bytes_to_mat, image)

        converter = ImageToImagePipeline(
            cleaner=construct_plugin_by_name(
                data_object.cleaner.id, data_object.cleaner.args
            ),
            detector=construct_plugin_by_name(
                data_object.detector.id, data_object.detector.args
            ),
            segmenter=construct_plugin_by_name(
                data_object.segmenter.id, data_object.segmenter.args
            ),
        )

        results = await converter([image_mat])

        result_bytes = await asyncio.to_thread(mat_to_bytes, results[0])

        return Response(200, None, Content(b"image/png", result_bytes))
    except:  # noqa: E722
        traceback.print_exc()
        return Response(
            500, None, Content(b"text/html", traceback.format_exc().encode())
        )


@post("/translate")
async def translate(data: FromForm[RequestFormDataclass]):
    try:
        image = data.value.file[0].data

        data_object = ServerPayloadDataclass.from_json(data.value.data)

        image_mat = await asyncio.to_thread(bytes_to_mat, image)

        converter = ImageToImagePipeline(
            translator=construct_plugin_by_name(
                data_object.translator.id, data_object.translator.args
            ),
            ocr=construct_plugin_by_name(data_object.ocr.id, data_object.ocr.args),
            drawer=construct_plugin_by_name(
                data_object.drawer.id, data_object.drawer.args
            ),
            cleaner=construct_plugin_by_name(
                data_object.cleaner.id, data_object.cleaner.args
            ),
            detector=construct_plugin_by_name(
                data_object.detector.id, data_object.detector.args
            ),
            segmenter=construct_plugin_by_name(
                data_object.segmenter.id, data_object.segmenter.args
            ),
            # color_detect_model=None,
        )

        results = await converter([image_mat])

        result_bytes = await asyncio.to_thread(mat_to_bytes, results[0])

        return Response(200, None, Content(b"image/png", result_bytes))
    except:  # noqa: E722
        traceback.print_exc()
        return Response(
            500, None, Content(b"text/html", traceback.format_exc().encode())
        )


@get("/info")
async def get_info():
    try:
        data = {
            "detectors": [],
            "segmenters": [],
            "translators": [],
            "ocrs": [],
            "drawers": [],
            "cleaners": [],
        }

        detectors = get_detectors()

        for x in range(len(detectors)):
            data["detectors"].append(
                {
                    "id": detectors[x].__name__,
                    "name": detectors[x].get_name(),
                    "description": detectors[x].__doc__,
                    "args": [x.get() for x in detectors[x].get_arguments()],
                }
            )

        segmenters = get_segmenters()

        for x in range(len(segmenters)):
            data["segmenters"].append(
                {
                    "id": segmenters[x].__name__,
                    "name": segmenters[x].get_name(),
                    "description": segmenters[x].__doc__,
                    "args": [x.get() for x in segmenters[x].get_arguments()],
                }
            )

        translators = get_translators()

        for x in range(len(translators)):
            data["translators"].append(
                {
                    "id": translators[x].__name__,
                    "name": translators[x].get_name(),
                    "description": translators[x].__doc__,
                    "args": [x.get() for x in translators[x].get_arguments()],
                }
            )

        ocr = get_ocrs()

        for x in range(len(ocr)):
            data["ocrs"].append(
                {
                    "id": ocr[x].__name__,
                    "name": ocr[x].get_name(),
                    "description": ocr[x].__doc__,
                    "args": [x.get() for x in ocr[x].get_arguments()],
                }
            )

        drawers = get_drawers()

        for x in range(len(drawers)):
            data["drawers"].append(
                {
                    "id": drawers[x].__name__,
                    "name": drawers[x].get_name(),
                    "description": drawers[x].__doc__,
                    "args": [x.get() for x in drawers[x].get_arguments()],
                }
            )

        cleaners = get_cleaners()

        for x in range(len(cleaners)):
            data["cleaners"].append(
                {
                    "id": cleaners[x].__name__,
                    "name": cleaners[x].get_name(),
                    "description": cleaners[x].__doc__,
                    "args": [x.get() for x in cleaners[x].get_arguments()],
                }
            )

        return json(data, 200)
    except:  # noqa: E722
        traceback.print_exc()
        return Response(
            500, None, Content(b"text/html", traceback.format_exc().encode())
        )


@get("/")
async def get_page():
    return file(
        os.path.join(build_path, "index.html"),
        content_type="text/html",
        content_disposition=ContentDispositionType.INLINE,
    )


@app.on_start
async def on_startup(app: Application):
    webbrowser.open("http://localhost:5000")
