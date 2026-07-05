"""Tests for GoogleCloudOCR image encoding color-space handling."""

import cv2
import numpy as np

from manga_translator.ocr.google_cloud import GoogleCloudOCR


def test_opencv_image_to_b64_preserves_rgb_color_order():
    """Encoding must convert RGB input to BGR before cv2.imencode so the
    resulting PNG round-trips back to the original RGB colors."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:] = (200, 30, 10)  # distinct R, G, B values in RGB order

    encoded_b64 = GoogleCloudOCR.opencv_image_to_b64(image)

    import base64

    encoded_bytes = base64.b64decode(encoded_b64)
    decoded_bgr = cv2.imdecode(
        np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    assert np.array_equal(decoded_rgb[0, 0], image[0, 0])
