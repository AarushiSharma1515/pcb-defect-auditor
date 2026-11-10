"""
Cheap checks on an uploaded file's raw bytes, done BEFORE the image is decoded.

We look at the bytes themselves instead of trusting the Content-Type the client
sent, because clients often label every file "application/octet-stream" (or send
no label at all), and a wrong label doesn't make a real image invalid.
"""

# Every file of these formats starts with these exact bytes ("magic bytes").
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def detect_image_format(data: bytes):
    """Return "JPEG" or "PNG" based on the file's first bytes, or None if it is neither."""
    if data.startswith(JPEG_MAGIC):
        return "JPEG"
    if data.startswith(PNG_MAGIC):
        return "PNG"
    return None
