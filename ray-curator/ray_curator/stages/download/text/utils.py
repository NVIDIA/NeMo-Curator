import unicodedata

import pycld2 as cld2
from charset_normalizer import detect


def lang_detect(decoded_html: str) -> str:
    try:
        details = cld2.detect(decoded_html)[2]
    except Exception:  # noqa: BLE001
        # Remove control characters
        cleaned_html = "".join(i for i in decoded_html if unicodedata.category(i)[0] != "C")
        details = cld2.detect(cleaned_html)[2]

    return details[0][0].upper()


def decode_html(html_bytes: bytes) -> str | None:
    # Convert from bytes to text using utf-8 encoding
    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # If utf-8 fails, try to find a different encoding
        return try_decode_with_detected_encoding(html_bytes)


def try_decode_with_detected_encoding(html_bytes: bytes) -> str | None:
    detected_encoding = detect(html_bytes)["encoding"]
    bad_detection = not detected_encoding or detected_encoding == "utf-8"
    if bad_detection:
        return None
    try:
        return html_bytes.decode(detected_encoding)
    except:  # noqa: E722
        return None
