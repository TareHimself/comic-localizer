import os
import cv2
import asyncio
from manga_translator.utils import get_default_torch_device, enable_perf
from manga_translator.get import construct_image_to_image_pipeline_from_config
import traceback
import hashlib
from blacksheep.server.authentication.apikey import APIKey, APIKeyAuthentication
from blacksheep import (
    Application,
    Content,
    FromFiles,
    Request,
    Response,
    auth,
    file,
    get,
    post,
    ContentDispositionType,
    json,
    bad_request,
)
from blacksheep.client import ClientSession
from essentials.secrets import Secret
import numpy as np
import yaml
import base64

enable_perf()
print("Using pytorch device", get_default_torch_device())
default_component = {"id": 0, "args": {}}
APP_PORT = 9000
TRANSLATED_IMAGES_PATH = os.path.abspath(os.path.join(".", ".temp", "translated"))
CONFIG_PATH = os.path.abspath(os.path.join(".", "config.yaml"))
PUBLIC_SERVER_ADDRESS = f"http://127.0.0.1:{APP_PORT}"
pipeline = construct_image_to_image_pipeline_from_config(config_path=CONFIG_PATH)
os.makedirs(TRANSLATED_IMAGES_PATH, exist_ok=True)

PENDING_TRANSLATION_JOBS: dict[str, asyncio.Future] = {}

app = Application()
lock = asyncio.Lock()

app.use_cors(
    allow_methods="*",
    allow_origins="*",
    allow_headers="* Authorization",
    max_age=300,
)


CACHE_ENABLED = True
BASE_64_ENABLED = False

with open(CONFIG_PATH, "rb") as _config_file:
    API_KEYS_KEY = "keys"
    SERVER_ADDRESS_KEY = "address"
    CACHE_KEY = "cache"
    BASE_64_KEY = "b64"
    data = yaml.safe_load(_config_file)
    if API_KEYS_KEY in data and data[API_KEYS_KEY] is not None:
        keys = data[API_KEYS_KEY]

        auth_strategy = app.use_authentication()

        for key in keys:
            auth_strategy.add(
                APIKeyAuthentication(
                    APIKey(secret=Secret.from_plain_text(key)),
                    param_name="X-API-Key",
                )
            )

        app.use_authorization()

    if SERVER_ADDRESS_KEY in data and isinstance(data[SERVER_ADDRESS_KEY], str):
        PUBLIC_SERVER_ADDRESS = data[SERVER_ADDRESS_KEY]

    if CACHE_KEY in data and isinstance(data[CACHE_KEY], bool):
        CACHE_ENABLED = data[CACHE_KEY]

    if BASE_64_KEY in data and isinstance(data[BASE_64_KEY], bool):
        BASE_64_ENABLED = data[BASE_64_KEY]

if not CACHE_ENABLED:
    BASE_64_ENABLED = True


# I cant figure out how to send these requests from the extension
@post("/api/v1/get-image")
async def fetch_image(request: Request):
    data = await request.json()
    async with ClientSession() as client:
        proxy_response = await client.get(data["url"], headers=data["headers"])

        assert proxy_response is not None

        resp = Response(
            proxy_response.status,
            [],
            Content(proxy_response.content_type(), await proxy_response.read()),
        )

        for header in proxy_response.headers.keys():
            if header.lower() in (b"content-length", b"transfer-encoding"):
                continue
            for h in proxy_response.headers.get(header):
                resp.headers.add(header, h)

        return resp


def compute_hash(data):
    return hashlib.sha256(data).hexdigest()


def bytes_to_mat(data: bytes):
    array = np.asarray(bytearray(data), dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR_RGB)


def post_translation(key: str, data: np.ndarray) -> str:
    if CACHE_ENABLED:
        save_path = os.path.join(TRANSLATED_IMAGES_PATH, f"{key}.png")
        cv2.imwrite(save_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))

    if BASE_64_ENABLED:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("Failed to encode translated image")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode(
            "ascii"
        )
    else:
        return make_translated_url(key)


def get_cached(key: str) -> str:
    cached_path = os.path.join(TRANSLATED_IMAGES_PATH, f"{key}.png")
    if BASE_64_ENABLED:
        with open(cached_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    else:
        return make_translated_url(key)


def save_translated(file_path: str, data: np.ndarray):
    cv2.imwrite(file_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))


def make_translated_url(key):
    return f"{PUBLIC_SERVER_ADDRESS}/api/v1/translated/{key}.png"


@auth()
@post("/api/v1/translate")
async def translate_images(files: FromFiles):
    try:
        images = files.value

        if len(images) == 0:
            return bad_request("No Images Sent")

        keys = await asyncio.gather(
            *[asyncio.to_thread(compute_hash, image.data) for image in images]
        )

        if CACHE_ENABLED:
            file_names = [f"{key}.png" for key in keys]
            file_paths = [
                os.path.join(TRANSLATED_IMAGES_PATH, file_name)
                for file_name in file_names
            ]
            files_exist = await asyncio.gather(
                *[
                    asyncio.to_thread(os.path.exists, file_path)
                    for file_path in file_paths
                ]
            )

            to_translate_indices = [
                i for i in range(len(files_exist)) if not files_exist[i]
            ]

            loop = asyncio.get_running_loop()

            results: list[asyncio.Future] = [
                asyncio.to_thread(get_cached, key)
                if files_exist[i]
                else loop.create_future()
                for i, key in enumerate(keys)
            ]
        else:
            loop = asyncio.get_running_loop()

            to_translate_indices: list[int] = []
            results: list[asyncio.Future] = []

            for i, key in enumerate(keys):
                to_translate_indices.append(i)
                results.append(loop.create_future())

        if to_translate_indices:
            translation_jobs: list[tuple[int, str, bytes, asyncio.Future]] = []
            async with lock:
                for i in to_translate_indices:
                    key = keys[i]
                    job = PENDING_TRANSLATION_JOBS.get(key, None)
                    if job is None:
                        job = results[i]
                        PENDING_TRANSLATION_JOBS[key] = job
                        translation_jobs.append((i, key, images[i].data, job))
                    else:
                        results[i] = job

            if len(translation_jobs) > 0:
                try:
                    batch = await asyncio.gather(
                        *[
                            asyncio.to_thread(bytes_to_mat, data)
                            for _, _, data, _ in translation_jobs
                        ]
                    )

                    translated_batch = await pipeline(batch)

                    translations = await asyncio.gather(
                        *[
                            asyncio.to_thread(post_translation, info[1], result)
                            for info, result in zip(translation_jobs, translated_batch)
                        ]
                    )

                    async with lock:
                        for i, job in enumerate(translation_jobs):
                            _, key, _, pending = job
                            pending.set_result(translations[i])
                            PENDING_TRANSLATION_JOBS.pop(key)
                except Exception as e:
                    async with lock:
                        for _, key, _, pending in translation_jobs:
                            pending.set_exception(e)
                            PENDING_TRANSLATION_JOBS.pop(key)
                    raise

        urls = await asyncio.gather(*results)

        return json({"urls": urls})
    except Exception:
        traceback.print_exc()
        return Response(
            500, content=Content(b"text/plain", traceback.format_exc().encode())
        )


@get("/api/v1/translated/{key}")
async def get_translated_image(key: str):
    # Reject keys containing path separators or traversal sequences
    if os.sep in key or (os.altsep and os.altsep in key) or ".." in key:
        return Response(400)

    file_path = os.path.realpath(os.path.join(TRANSLATED_IMAGES_PATH, key))
    base_path = os.path.realpath(TRANSLATED_IMAGES_PATH)

    # Ensure the resolved path stays within TRANSLATED_IMAGES_PATH
    if not file_path.startswith(base_path + os.sep):
        return Response(400)

    if await asyncio.to_thread(os.path.exists, file_path):
        return file(
            value=file_path,
            content_type="image/png",
            file_name=key,
            content_disposition=ContentDispositionType.INLINE,
        )
    else:
        return Response(404)


@app.on_start
async def on_startup(app: Application):
    print(f"Running at {PUBLIC_SERVER_ADDRESS}/api/v1")
