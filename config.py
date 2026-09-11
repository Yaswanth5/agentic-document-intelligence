import os

# ============================================================
# STORAGE
# ============================================================

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    os.path.join(
        os.path.dirname(__file__),
        "storage",
        "chroma"
    )
)


# ============================================================
# LLM
# ============================================================

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:4b"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_TEMPERATURE = float(
    os.getenv(
        "OLLAMA_TEMPERATURE",
        "0"
    )
)

OLLAMA_NUM_PREDICT = int(
    os.getenv(
        "OLLAMA_NUM_PREDICT",
        "256"
    )
)

OLLAMA_JSON_NUM_PREDICT = int(
    os.getenv(
        "OLLAMA_JSON_NUM_PREDICT",
        "256"
    )
)

OLLAMA_KEEP_ALIVE = os.getenv(
    "OLLAMA_KEEP_ALIVE",
    "-1"
)


# ============================================================
# OCR
# ============================================================

OCR_ENGINE_VERSION = "v5"

OCR_SCALE = float(
    os.getenv(
        "OCR_SCALE",
        "1.5"
    )
)

OCR_TIMEOUT = int(
    os.getenv(
        "OCR_TIMEOUT",
        "10"
    )
)

OCR_PROBE_TIMEOUT = int(
    os.getenv(
        "OCR_PROBE_TIMEOUT",
        "3"
    )
)

OCR_MIN_GOOD_CHARS = int(
    os.getenv(
        "OCR_MIN_GOOD_CHARS",
        "40"
    )
)

OCR_GOOD_CONFIDENCE = float(
    os.getenv(
        "OCR_GOOD_CONFIDENCE",
        "50"
    )
)

OCR_MIN_CONFIDENCE = float(
    os.getenv(
        "OCR_MIN_CONFIDENCE",
        "20"
    )
)

OCR_MAX_FINAL_PASSES = int(
    os.getenv(
        "OCR_MAX_FINAL_PASSES",
        "2"
    )
)

OCR_CACHE_ENABLED = (
    os.getenv(
        "OCR_CACHE_ENABLED",
        "true"
    ).lower()
    == "true"
)


# ============================================================
# PDF
# ============================================================

PDF_RENDER_SCALE = float(
    os.getenv(
        "PDF_RENDER_SCALE",
        "1.25"
    )
)

PDF_OCR_RENDER_SCALE = float(
    os.getenv(
        "PDF_OCR_RENDER_SCALE",
        "2.0"
    )
)

PDF_OCR_IF_TEXT_CHARS_BELOW = int(
    os.getenv(
        "PDF_OCR_IF_TEXT_CHARS_BELOW",
        "40"
    )
)

PDF_MAX_EMBEDDED_IMAGES = int(
    os.getenv(
        "PDF_MAX_EMBEDDED_IMAGES",
        "8"
    )
)

PDF_USE_EMBEDDED_IMAGES = (
    os.getenv(
        "PDF_USE_EMBEDDED_IMAGES",
        "true"
    ).lower()
    == "true"
)

PDF_RENDER_VISUALS = (
    os.getenv(
        "PDF_RENDER_VISUALS",
        "true"
    ).lower()
    == "true"
)


# ============================================================
# STRUCTURE
# ============================================================

TABLE_EXTRACTION_ENABLED = (
    os.getenv(
        "TABLE_EXTRACTION_ENABLED",
        "true"
    ).lower()
    == "true"
)

LAYOUT_EXTRACTION_ENABLED = (
    os.getenv(
        "LAYOUT_EXTRACTION_ENABLED",
        "true"
    ).lower()
    == "true"
)

VECTOR_EXTRACTION_ENABLED = (
    os.getenv(
        "VECTOR_EXTRACTION_ENABLED",
        "true"
    ).lower()
    == "true"
)


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "2500"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "250"
    )
)


# ============================================================
# RETRIEVAL
# ============================================================

RETRIEVAL_TOP_K = int(
    os.getenv(
        "RETRIEVAL_TOP_K",
        "8"
    )
)

RETRIEVAL_MAX_K = int(
    os.getenv(
        "RETRIEVAL_MAX_K",
        "15"
    )
)