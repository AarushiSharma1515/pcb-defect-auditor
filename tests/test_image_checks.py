"""Unit tests for the header-only checks in app/ml/image_checks.py."""
import cv2
import numpy as np
import pytest

from app.ml.image_checks import detect_image_format, read_image_size


def encode(extension, width, height, params=()):
    return cv2.imencode(extension, np.zeros((height, width, 3), np.uint8), list(params))[1].tobytes()


@pytest.mark.parametrize("extension, params, expected_format", [
    (".png", (), "PNG"),
    (".jpg", (), "JPEG"),                                        # baseline JPEG
    (".jpg", (cv2.IMWRITE_JPEG_PROGRESSIVE, 1), "JPEG"),         # progressive JPEG (different frame marker)
])
def test_size_is_read_from_header(extension, params, expected_format):
    data = encode(extension, 321, 123, params)                   # width != height, to catch a swap
    assert detect_image_format(data) == expected_format
    assert read_image_size(data, expected_format) == (321, 123)


def test_not_an_image():
    assert detect_image_format(b"%PDF-1.4") is None
    assert detect_image_format(b"") is None


def test_truncated_headers_return_none():
    assert read_image_size(encode(".jpg", 50, 50)[:20], "JPEG") is None
    assert read_image_size(encode(".png", 50, 50)[:20], "PNG") is None
