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


def read_image_size(data: bytes, image_format: str):
    """Return (width, height) read from the file's header, WITHOUT decoding any pixels.

    This lets us refuse a "pixel bomb" (e.g. a 0.5 MB PNG that unpacks to 432 MB) before
    OpenCV allocates memory for it. Returns None if the header can't be read (corrupt file).
    """
    if image_format == "PNG":
        # PNG layout: 8-byte signature, then the IHDR chunk:
        # 4-byte length, the letters "IHDR", 4-byte width, 4-byte height (big-endian).
        if len(data) >= 24 and data[12:16] == b"IHDR":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        return None

    # JPEG layout: a chain of segments, each "FF <marker> <2-byte length> <data...>".
    # The size is in the "start of frame" segment (markers C0-CF, except C4, C8 and CC,
    # which mean other things): <1-byte precision> <2-byte height> <2-byte width>.
    i = 2  # skip the FF D8 "start of image" marker
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            return None                       # lost track of the segments: corrupt file
        marker = data[i + 1]
        if marker == 0xFF:                    # padding byte, skip it
            i += 1
            continue
        if marker == 0xDA:                    # image data starts, but no frame header seen
            return None
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        i += 2 + length                       # jump to the next segment
    return None
