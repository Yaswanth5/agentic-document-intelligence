import os
import cv2
import json
import subprocess
from pathlib import Path


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCRProcessor:
    """
    Fast generic OCR processor for heterogeneous documents.

    Strategy:
    - Avoid expensive preprocessing unless required.
    - Run one fast OCR pass first.
    - Run enhanced OCR only when confidence is low.
    - Cache available Tesseract languages.
    - Avoid repeatedly preprocessing images.
    - Save debug artifacts only when explicitly enabled.
    """

    def __init__(self, debug=False):

        if not os.path.exists(TESSERACT_PATH):
            raise FileNotFoundError(
                f"Tesseract not found at: {TESSERACT_PATH}"
            )

        self.debug = debug

        self.output_dir = Path(
            "data/processed/ocr_debug"
        )

        if self.debug:
            self.output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        # ---------------------------------------------------------
        # PERFORMANCE SETTINGS
        # ---------------------------------------------------------

        self.good_confidence = 65.0
        self.minimum_text_length = 40

        # Number of OCR passes:
        # 1 = fastest
        # 2 = fallback when first pass is poor
        self.max_passes = 2

        # Cache Tesseract languages once.
        self._available_languages = None

    # =========================================================
    # CHECK AVAILABLE TESSERACT LANGUAGES
    # =========================================================

    def get_available_languages(self):

        if self._available_languages is not None:
            return self._available_languages

        try:

            result = subprocess.run(
                [
                    TESSERACT_PATH,
                    "--list-langs"
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                )
            )

            languages = []

            for line in result.stdout.splitlines():

                line = line.strip()

                if (
                    line
                    and line != "List of available languages in"
                ):
                    languages.append(line)

            self._available_languages = languages

            return languages

        except Exception:

            self._available_languages = ["eng"]

            return self._available_languages

    # =========================================================
    # RESOLVE OCR LANGUAGE
    # =========================================================

    def resolve_language(self, language=None):

        available = self.get_available_languages()

        if language:

            requested = language.split("+")

            valid = [
                lang
                for lang in requested
                if lang in available
            ]

            if valid:
                return "+".join(valid)

        if "eng" in available:
            return "eng"

        if available:
            return available[0]

        raise RuntimeError(
            "No Tesseract language models found."
        )

    # =========================================================
    # IMAGE PREPROCESSING
    # =========================================================

    def preprocess_variants(self, image):

        if image is None:
            raise ValueError(
                "Input image is None"
            )

        variants = {}

        # ---------------------------------------------------------
        # ORIGINAL
        # ---------------------------------------------------------

        variants["original"] = image

        # ---------------------------------------------------------
        # GRAYSCALE + CLAHE
        # ---------------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        variants["enhanced"] = enhanced

        # ---------------------------------------------------------
        # OTSU
        # ---------------------------------------------------------

        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        variants["otsu"] = otsu

        return variants

    # =========================================================
    # FAST TESSERACT OCR
    # =========================================================

    def tesseract(
        self,
        image,
        language="eng",
        psm=6
    ):

        command = [
            TESSERACT_PATH,
            "stdin",
            "stdout",
            "-l",
            language,
            "--psm",
            str(psm),
            "--oem",
            "1",
            "tsv"
        ]

        try:

            success, encoded = cv2.imencode(
                ".png",
                image
            )

            if not success:
                return {
                    "text": "",
                    "confidence": 0.0
                }

            result = subprocess.run(
                command,
                input=encoded.tobytes(),
                capture_output=True,
                text=False,
                timeout=60,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                )
            )

            if result.returncode != 0:
                return {
                    "text": "",
                    "confidence": 0.0
                }

            stdout = result.stdout.decode(
                "utf-8",
                errors="ignore"
            )

            lines = stdout.splitlines()

            if len(lines) <= 1:
                return {
                    "text": "",
                    "confidence": 0.0
                }

            text_parts = []
            confidences = []

            for line in lines[1:]:

                parts = line.split("\t")

                if len(parts) < 12:
                    continue

                text = parts[11].strip()

                if not text:
                    continue

                try:

                    confidence = float(
                        parts[10]
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    continue

                text_parts.append(text)

                if confidence >= 0:
                    confidences.append(
                        confidence
                    )

            text = " ".join(
                text_parts
            ).strip()

            if confidences:

                average_confidence = (
                    sum(confidences)
                    / len(confidences)
                )

            else:

                average_confidence = 0.0

            return {
                "text": text,
                "confidence": round(
                    average_confidence,
                    2
                )
            }

        except Exception as exc:

            print(
                f"[OCR] Tesseract error: "
                f"{type(exc).__name__}"
            )

            return {
                "text": "",
                "confidence": 0.0
            }

    # =========================================================
    # OCR QUALITY CHECK
    # =========================================================

    def _is_good_result(self, result):

        text = result.get(
            "text",
            ""
        ).strip()

        confidence = result.get(
            "confidence",
            0.0
        )

        if len(text) < self.minimum_text_length:
            return False

        if confidence < self.good_confidence:
            return False

        return True

    # =========================================================
    # RUN OCR
    # =========================================================

    def run_ocr(
        self,
        image,
        language=None
    ):

        language = self.resolve_language(
            language
        )

        # ---------------------------------------------------------
        # PASS 1: ORIGINAL IMAGE
        # ---------------------------------------------------------

        first_result = self.tesseract(
            image,
            language=language,
            psm=6
        )

        results = {
            "original": {
                "text": first_result["text"],
                "confidence": first_result["confidence"],
                "variant": "original"
            }
        }

        # ---------------------------------------------------------
        # FAST EXIT
        # ---------------------------------------------------------

        if self._is_good_result(
            first_result
        ):

            return {
                "language": language,
                "best_variant": "original",
                "text": first_result["text"],
                "confidence": first_result["confidence"],
                "variants": results
            }

        # ---------------------------------------------------------
        # PASS 2: ENHANCED IMAGE
        # ---------------------------------------------------------

        variants = self.preprocess_variants(
            image
        )

        enhanced_image = variants[
            "enhanced"
        ]

        enhanced_result = self.tesseract(
            enhanced_image,
            language=language,
            psm=6
        )

        results["enhanced"] = {
            "text": enhanced_result["text"],
            "confidence": enhanced_result["confidence"],
            "variant": "enhanced"
        }

        # ---------------------------------------------------------
        # SELECT BEST RESULT
        # ---------------------------------------------------------

        best_variant = max(
            results,
            key=lambda x: (
                results[x]["confidence"],
                len(results[x]["text"])
            )
        )

        best_result = results[
            best_variant
        ]

        return {
            "language": language,
            "best_variant": best_variant,
            "text": best_result["text"],
            "confidence": best_result["confidence"],
            "variants": results
        }

    # =========================================================
    # PROCESS DOCUMENT
    # =========================================================

    def process_document(
        self,
        image_path,
        language=None
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Could not find: {image_path}"
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                f"Could not read image: "
                f"{image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        print(
            f"[OCR] Document size: "
            f"{width} x {height}"
        )

        # ---------------------------------------------------------
        # OCR
        # ---------------------------------------------------------

        result = self.run_ocr(
            image,
            language=language
        )

        print(
            f"[OCR] Best variant: "
            f"{result['best_variant']} | "
            f"Confidence: "
            f"{result['confidence']:.2f}"
        )

        # ---------------------------------------------------------
        # DEBUG OUTPUT
        # ---------------------------------------------------------

        if self.debug:

            variants = self.preprocess_variants(
                image
            )

            for name, variant in variants.items():

                output_file = (
                    self.output_dir
                    / f"document_{name}.png"
                )

                cv2.imwrite(
                    str(output_file),
                    variant
                )

            output_file = (
                self.output_dir
                / "ocr_results.json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        return result