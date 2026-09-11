from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# Make language detection deterministic.
DetectorFactory.seed = 0


MAX_DETECTION_CHARS = 5000
MIN_DETECTION_CHARS = 20


def detect_language(text):
    """
    Detect the language of document text.

    Returns:
        ISO 639-1 language code such as:
        'en', 'te', 'hi', 'ta', 'kn', 'ml', etc.

        Returns 'unknown' when reliable detection is not possible.
    """

    if text is None:
        return "unknown"

    if not isinstance(text, str):
        text = str(text)

    # Remove excessive whitespace.
    text = " ".join(
        text.split()
    )

    if len(text) < MIN_DETECTION_CHARS:
        return "unknown"

    # Limit input size for faster detection.
    sample = text[:MAX_DETECTION_CHARS]

    try:
        return detect(sample)

    except (
        LangDetectException,
        ValueError,
        TypeError
    ):
        return "unknown"