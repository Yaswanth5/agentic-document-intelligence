# processing/document_loader.py

import os
import re
import cv2
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image


# ============================================================
# OPTIONAL CONFIG IMPORT
# ============================================================

try:
    from config import (
        OCR_ENGINE_VERSION,
        OCR_SCALE,
        OCR_TIMEOUT,
        OCR_PROBE_TIMEOUT,
        OCR_MIN_GOOD_CHARS,
        OCR_GOOD_CONFIDENCE,
        OCR_MIN_CONFIDENCE,
        OCR_MAX_FINAL_PASSES,
        OCR_CACHE_ENABLED,
        PDF_RENDER_SCALE,
        PDF_OCR_RENDER_SCALE,
        PDF_OCR_IF_TEXT_CHARS_BELOW,
        PDF_MAX_EMBEDDED_IMAGES,
        PDF_USE_EMBEDDED_IMAGES,
        PDF_RENDER_VISUALS,
        TABLE_EXTRACTION_ENABLED,
        LAYOUT_EXTRACTION_ENABLED,
        VECTOR_EXTRACTION_ENABLED,
    )
except ImportError:

    OCR_ENGINE_VERSION = "v5"

    OCR_SCALE = 1.5

    OCR_TIMEOUT = 10
    OCR_PROBE_TIMEOUT = 3

    OCR_MIN_GOOD_CHARS = 40
    OCR_GOOD_CONFIDENCE = 50
    OCR_MIN_CONFIDENCE = 20
    OCR_MAX_FINAL_PASSES = 2

    OCR_CACHE_ENABLED = True

    PDF_RENDER_SCALE = 1.25
    PDF_OCR_RENDER_SCALE = 2.0
    PDF_OCR_IF_TEXT_CHARS_BELOW = 40

    PDF_MAX_EMBEDDED_IMAGES = 8
    PDF_USE_EMBEDDED_IMAGES = True
    PDF_RENDER_VISUALS = True

    TABLE_EXTRACTION_ENABLED = True
    LAYOUT_EXTRACTION_ENABLED = True
    VECTOR_EXTRACTION_ENABLED = True


# ============================================================
# DOCUMENT LOADER
# ============================================================

class DocumentLoader:

    """
    Generic multimodal document loader.

    Supported:
        PDF
        PNG
        JPG / JPEG
        WEBP
        BMP
        TIFF
        DOCX
        XLSX / XLS
        PPTX
        TXT
        MD
        CSV

    Design principles:
        1. Native extraction first.
        2. OCR only when necessary.
        3. Embedded images are extracted independently.
        4. Preserve original visual information.
        5. Preserve page-level structure.
        6. No document-specific field logic.
        7. Multilingual OCR uses installed Tesseract languages.
    """

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self):

        self.tesseract_cmd = (
            self._find_tesseract()
        )

        if self.tesseract_cmd:

            pytesseract.pytesseract.tesseract_cmd = (
                self.tesseract_cmd
            )

            print(
                f"[OCR] Tesseract: "
                f"{self.tesseract_cmd}"
            )

        self.available_languages = (
            self._get_available_languages()
        )

        if self.available_languages:

            print(
                "[OCR] Languages available: "
                + ", ".join(
                    self.available_languages
                )
            )

        self._ocr_cache = {}

    # ========================================================
    # TESSERACT DISCOVERY
    # ========================================================

    def _find_tesseract(self) -> Optional[str]:

        candidates = [

            os.getenv(
                "TESSERACT_CMD",
                ""
            ),

            r"C:\Program Files\Tesseract-OCR\tesseract.exe",

            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",

            "tesseract",
        ]

        for candidate in candidates:

            if not candidate:
                continue

            if candidate == "tesseract":

                try:

                    result = subprocess.run(
                        [
                            "where",
                            "tesseract"
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3
                    )

                    if result.returncode == 0:

                        path = (
                            result.stdout
                            .strip()
                            .splitlines()
                        )

                        if path:

                            return path[0]

                except Exception:
                    pass

                continue

            if Path(candidate).exists():

                return candidate

        return None

    # ========================================================
    # AVAILABLE OCR LANGUAGES
    # ========================================================

    def _get_available_languages(self) -> List[str]:

        if not self.tesseract_cmd:
            return []

        try:

            langs = (
                pytesseract.get_languages(
                    config=""
                )
            )

            langs = [
                str(x).strip()
                for x in langs
                if str(x).strip()
            ]

            return sorted(
                set(langs)
            )

        except Exception as exc:

            print(
                "[OCR] Could not determine "
                f"installed languages: {exc}"
            )

            return []

    # ========================================================
    # HASH
    # ========================================================

    def _image_hash(
        self,
        image: np.ndarray
    ) -> str:

        try:

            if image is None:
                return ""

            data = image.tobytes()

            return hashlib.md5(
                data
            ).hexdigest()

        except Exception:

            return ""

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    def _clean_text(
        self,
        text: Any
    ) -> str:

        if text is None:
            return ""

        text = str(text)

        text = text.replace(
            "\x00",
            " "
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # ========================================================
    # OCR QUALITY
    # ========================================================

    def _quality_score(
        self,
        text: str,
        confidence: float
    ) -> float:

        if not text:

            return 0.0

        cleaned = self._clean_text(
            text
        )

        if not cleaned:

            return 0.0

        chars = len(cleaned)

        alnum = sum(
            c.isalnum()
            for c in cleaned
        )

        alpha = sum(
            c.isalpha()
            for c in cleaned
        )

        printable = sum(
            c.isprintable()
            for c in cleaned
        )

        alnum_ratio = (
            alnum / max(chars, 1)
        )

        alpha_ratio = (
            alpha / max(chars, 1)
        )

        printable_ratio = (
            printable / max(chars, 1)
        )

        length_score = min(
            chars / 300.0,
            1.0
        )

        score = (

            0.35 * min(
                max(confidence, 0.0) / 100.0,
                1.0
            )

            +

            0.25 * alnum_ratio

            +

            0.20 * alpha_ratio

            +

            0.10 * printable_ratio

            +

            0.10 * length_score
        )

        return round(
            score * 100,
            2
        )

    # ========================================================
    # OCR GOOD CHECK
    # ========================================================

    def _ocr_is_good(
        self,
        text: str,
        confidence: float
    ) -> bool:

        cleaned = self._clean_text(
            text
        )

        if len(cleaned) < OCR_MIN_GOOD_CHARS:
            return False

        if confidence >= OCR_GOOD_CONFIDENCE:
            return True

        score = self._quality_score(
            cleaned,
            confidence
        )

        return score >= 55

    # ========================================================
    # SCRIPT DETECTION
    # ========================================================

    def _detect_scripts(
        self,
        text: str
    ) -> List[str]:

        if not text:
            return []

        counts = {

            "tel": 0,
            "tam": 0,
            "kan": 0,
            "mal": 0,
            "hin": 0,
            "ben": 0,
            "guj": 0,
            "pan": 0,
            "ori": 0,
            "urd": 0,
            "eng": 0,
        }

        for char in text:

            code = ord(char)

            # Telugu
            if 0x0C00 <= code <= 0x0C7F:
                counts["tel"] += 1

            # Tamil
            elif 0x0B80 <= code <= 0x0BFF:
                counts["tam"] += 1

            # Kannada
            elif 0x0C80 <= code <= 0x0CFF:
                counts["kan"] += 1

            # Malayalam
            elif 0x0D00 <= code <= 0x0D7F:
                counts["mal"] += 1

            # Devanagari
            elif 0x0900 <= code <= 0x097F:
                counts["hin"] += 1

            # Bengali
            elif 0x0980 <= code <= 0x09FF:
                counts["ben"] += 1

            # Gujarati
            elif 0x0A80 <= code <= 0x0AFF:
                counts["guj"] += 1

            # Gurmukhi
            elif 0x0A00 <= code <= 0x0A7F:
                counts["pan"] += 1

            # Oriya
            elif 0x0B00 <= code <= 0x0B7F:
                counts["ori"] += 1

            # Arabic
            elif (
                0x0600 <= code <= 0x06FF
            ):
                counts["urd"] += 1

            # Latin
            elif (
                ("A" <= char <= "Z")
                or
                ("a" <= char <= "z")
            ):
                counts["eng"] += 1

        total = sum(
            counts.values()
        )

        if total == 0:
            return []

        ordered = sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        scripts = []

        for lang, count in ordered:

            if count <= 0:
                continue

            ratio = (
                count / total
            )

            if ratio >= 0.03:

                scripts.append(
                    lang
                )

        return scripts

    # ========================================================
    # OCR LANGUAGE CANDIDATES
    # ========================================================

    def _language_candidates(
        self,
        script_text: str
    ) -> List[str]:

        detected = self._detect_scripts(
            script_text
        )

        candidates = []

        for lang in detected:

            if lang in self.available_languages:

                candidates.append(
                    lang
                )

        # English is useful as a fallback
        # for mixed-language documents.

        if (
            "eng" in self.available_languages
            and "eng" not in candidates
        ):

            candidates.append(
                "eng"
            )

        # Avoid expensive multilingual
        # combinations.

        return candidates[:2]

    # ========================================================
    # ORIENTATION DETECTION
    # ========================================================

    def _detect_orientation(
        self,
        image: np.ndarray
    ) -> int:

        if not self.tesseract_cmd:
            return 0

        try:

            pil_image = Image.fromarray(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )
            )

            result = pytesseract.image_to_osd(
                pil_image,
                config="--psm 0",
                timeout=OCR_PROBE_TIMEOUT
            )

            match = re.search(
                r"Rotate:\s+(\d+)",
                result
            )

            if match:

                angle = int(
                    match.group(1)
                )

                return angle

        except Exception as exc:

            print(
                "[OCR OSD] "
                f"{type(exc).__name__}: {exc}"
            )

        return 0

    # ========================================================
    # ROTATE IMAGE
    # ========================================================

    def _rotate_image(
        self,
        image: np.ndarray,
        angle: int
    ) -> np.ndarray:

        if angle == 90:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE
            )

        if angle == 180:

            return cv2.rotate(
                image,
                cv2.ROTATE_180
            )

        if angle == 270:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )

        return image

    # ========================================================
    # OCR SINGLE PASS
    # ========================================================

    def _run_ocr(
        self,
        image: np.ndarray,
        lang: str = "eng",
        psm: int = 6,
        timeout: Optional[int] = None
    ) -> Tuple[str, float, Dict[str, Any]]:

        if (
            image is None
            or image.size == 0
        ):

            return "", 0.0, {}

        if not self.tesseract_cmd:

            return "", 0.0, {}

        timeout = (
            timeout
            if timeout is not None
            else OCR_TIMEOUT
        )

        start = time.time()

        try:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                rgb
            )

            config = (
                f"--psm {psm}"
            )

            data = pytesseract.image_to_data(
                pil_image,
                lang=lang,
                config=config,
                output_type=(
                    pytesseract.Output.DICT
                ),
                timeout=timeout
            )

            texts = []

            confidences = []

            count = len(
                data.get(
                    "text",
                    []
                )
            )

            for index in range(count):

                text = str(
                    data["text"][index]
                ).strip()

                if not text:
                    continue

                texts.append(
                    text
                )

                try:

                    conf = float(
                        data["conf"][index]
                    )

                    if conf >= 0:

                        confidences.append(
                            conf
                        )

                except Exception:
                    pass

            text = self._clean_text(
                " ".join(texts)
            )

            confidence = (
                sum(confidences)
                /
                len(confidences)
                if confidences
                else 0.0
            )

            elapsed = (
                time.time() - start
            )

            score = self._quality_score(
                text,
                confidence
            )

            print(
                f"[OCR PASS] "
                f"lang={lang} "
                f"psm={psm} "
                f"chars={len(text)} "
                f"confidence={confidence:.1f} "
                f"quality={score:.1f} "
                f"time={elapsed:.2f}s"
            )

            return (

                text,

                round(
                    confidence,
                    2
                ),

                {
                    "quality_score": score,
                    "elapsed": elapsed,
                    "language": lang,
                    "psm": psm,
                }
            )

        except Exception as exc:

            elapsed = (
                time.time() - start
            )

            print(
                "[OCR PASS] "
                f"timeout/failure: "
                f"{type(exc).__name__}: {exc}"
            )

            return (
                "",
                0.0,
                {
                    "quality_score": 0,
                    "elapsed": elapsed,
                    "language": lang,
                    "psm": psm,
                    "error": str(exc),
                }
            )

    # ========================================================
    # OCR MAIN PIPELINE
    # ========================================================

    def _perform_ocr(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:

        if (
            image is None
            or image.size == 0
        ):

            return {
                "text": "",
                "confidence": 0.0,
                "language": [],
                "orientation": 0,
                "passes": [],
            }

        if not self.tesseract_cmd:

            return {
                "text": "",
                "confidence": 0.0,
                "language": [],
                "orientation": 0,
                "passes": [],
            }

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        image_hash = self._image_hash(
            image
        )

        cache_key = (
            OCR_ENGINE_VERSION,
            image_hash
        )

        if (
            OCR_CACHE_ENABLED
            and cache_key in self._ocr_cache
        ):

            cached = self._ocr_cache[
                cache_key
            ]

            print(
                "[OCR CACHE] Hit"
            )

            return cached

        passes = []

        # ----------------------------------------------------
        # PASS 1
        # Fast English baseline
        # ----------------------------------------------------

        baseline_text, baseline_conf, baseline_meta = (
            self._run_ocr(
                image,
                lang=(
                    "eng"
                    if "eng"
                    in self.available_languages
                    else (
                        self.available_languages[0]
                        if self.available_languages
                        else "eng"
                    )
                ),
                psm=6,
                timeout=OCR_TIMEOUT
            )
        )

        passes.append(
            baseline_meta
        )

        if self._ocr_is_good(
            baseline_text,
            baseline_conf
        ):

            result = {

                "text": baseline_text,

                "confidence": baseline_conf,

                "language": [
                    baseline_meta.get(
                        "language"
                    )
                ],

                "orientation": 0,

                "passes": passes,
            }

            if OCR_CACHE_ENABLED:

                self._ocr_cache[
                    cache_key
                ] = result

            return result

        # ----------------------------------------------------
        # ORIENTATION
        # Only expensive fallback when
        # baseline OCR was poor.
        # ----------------------------------------------------

        orientation = (
            self._detect_orientation(
                image
            )
        )

        if orientation:

            print(
                f"[OCR] Corrected orientation "
                f"by {orientation} degrees"
            )

            image = self._rotate_image(
                image,
                orientation
            )

            # Re-run baseline on corrected image.

            corrected_text, corrected_conf, corrected_meta = (
                self._run_ocr(
                    image,
                    lang=(
                        "eng"
                        if "eng"
                        in self.available_languages
                        else (
                            self.available_languages[0]
                            if self.available_languages
                            else "eng"
                        )
                    ),
                    psm=6,
                    timeout=OCR_TIMEOUT
                )
            )

            passes.append(
                corrected_meta
            )

            if self._ocr_is_good(
                corrected_text,
                corrected_conf
            ):

                result = {

                    "text": corrected_text,

                    "confidence": corrected_conf,

                    "language": [
                        corrected_meta.get(
                            "language"
                        )
                    ],

                    "orientation": orientation,

                    "passes": passes,
                }

                if OCR_CACHE_ENABLED:

                    self._ocr_cache[
                        cache_key
                    ] = result

                return result

            baseline_text = corrected_text
            baseline_conf = corrected_conf

        # ----------------------------------------------------
        # SCRIPT HINT
        # ----------------------------------------------------

        script_hint = self._clean_text(
            baseline_text
        )

        candidates = (
            self._language_candidates(
                script_hint
            )
        )

        print(
            "[OCR] Dynamic language candidates: "
            + (
                ", ".join(candidates)
                if candidates
                else "none"
            )
        )

        # ----------------------------------------------------
        # MULTILINGUAL FALLBACK
        # ----------------------------------------------------

        best_text = baseline_text
        best_conf = baseline_conf
        best_score = self._quality_score(
            baseline_text,
            baseline_conf
        )

        best_language = "eng"

        # Do not probe every installed language.
        # Only use script-supported candidates.

        for language in candidates:

            if len(passes) >= (
                OCR_MAX_FINAL_PASSES + 1
            ):
                break

            if language == "eng":
                continue

            text, confidence, meta = (
                self._run_ocr(
                    image,
                    lang=language,
                    psm=6,
                    timeout=OCR_TIMEOUT
                )
            )

            passes.append(
                meta
            )

            score = self._quality_score(
                text,
                confidence
            )

            if score > best_score:

                best_text = text
                best_conf = confidence
                best_score = score
                best_language = language

            if self._ocr_is_good(
                text,
                confidence
            ):

                break

        # ----------------------------------------------------
        # SECOND PSM FALLBACK
        # ----------------------------------------------------

        if (
            not self._ocr_is_good(
                best_text,
                best_conf
            )
            and len(passes)
            < OCR_MAX_FINAL_PASSES + 1
        ):

            text, confidence, meta = (
                self._run_ocr(
                    image,
                    lang=best_language,
                    psm=11,
                    timeout=OCR_TIMEOUT
                )
            )

            passes.append(
                meta
            )

            score = self._quality_score(
                text,
                confidence
            )

            if score > best_score:

                best_text = text
                best_conf = confidence
                best_score = score

        result = {

            "text": self._clean_text(
                best_text
            ),

            "confidence": round(
                best_conf,
                2
            ),

            "language": [
                best_language
            ]
            if best_language
            else [],

            "orientation": orientation,

            "passes": passes,
        }

        if OCR_CACHE_ENABLED:

            self._ocr_cache[
                cache_key
            ] = result

        return result

    # ========================================================
    # IMAGE RESIZE
    # ========================================================

    def _resize_for_ocr(
        self,
        image: np.ndarray,
        scale: float
    ) -> np.ndarray:

        if (
            image is None
            or image.size == 0
        ):

            return image

        if scale == 1.0:

            return image

        height, width = (
            image.shape[:2]
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        return cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=(
                cv2.INTER_CUBIC
                if scale > 1
                else cv2.INTER_AREA
            )
        )

    # ========================================================
    # IMAGE PROCESSING
    # ========================================================

    def _process_image(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        start = time.time()

        print(
            f"[IMAGE] Processing: "
            f"{Path(file_path).name}"
        )

        image = cv2.imread(
            file_path,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise ValueError(
                f"Could not read image: "
                f"{file_path}"
            )

        original_height, original_width = (
            image.shape[:2]
        )

        ocr_image = (
            self._resize_for_ocr(
                image,
                OCR_SCALE
            )
        )

        ocr_result = (
            self._perform_ocr(
                ocr_image
            )
        )

        text = self._clean_text(
            ocr_result.get(
                "text",
                ""
            )
        )

        elapsed = (
            time.time() - start
        )

        print(
            f"[OCR] Completed "
            f"{Path(file_path).name} "
            f"in {elapsed:.2f}s "
            f"chars={len(text)} "
            f"languages="
            f"{ocr_result.get('language', [])}"
        )

        return {

            "file_name": Path(
                file_path
            ).name,

            "file_path": str(
                file_path
            ),

            "page_count": 1,

            "text": text,

            "images": [
                {
                    "image_id": "image_1",
                    "path": str(file_path),
                    "source": "original",
                    "width": original_width,
                    "height": original_height,
                    "page": 1,
                }
            ],

            "tables": [],

            "layout": [],

            "pages": [
                {

                    "page_number": 1,

                    "native_text": "",

                    "ocr_text": text,

                    "text": text,

                    "ocr": ocr_result,

                    "images": [
                        {
                            "image_id": "image_1",
                            "path": str(file_path),
                            "source": "original",
                            "width": original_width,
                            "height": original_height,
                        }
                    ],

                    "tables": [],

                    "layout": [],
                }
            ],

            "metadata": {

                "file_name": Path(
                    file_path
                ).name,

                "file_type": (
                    Path(file_path)
                    .suffix
                    .lower()
                ),

                "document_modality": "image",

                "ocr_engine": "tesseract",

                "ocr_languages": (
                    ocr_result.get(
                        "language",
                        []
                    )
                ),

                "ocr_confidence": (
                    ocr_result.get(
                        "confidence",
                        0
                    )
                ),

                "processing_time": elapsed,

                "width": original_width,

                "height": original_height,
            },
        }

    # ========================================================
    # PDF PAGE RENDER
    # ========================================================

    def _render_page(
        self,
        page,
        output_dir: Path,
        page_number: int,
        scale: float
    ) -> Optional[str]:

        try:

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            matrix = fitz.Matrix(
                scale,
                scale
            )

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            output_path = (
                output_dir
                /
                f"page_{page_number}.png"
            )

            pix.save(
                str(output_path)
            )

            return str(
                output_path
            )

        except Exception as exc:

            print(
                "[PDF RENDER] Failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return None

    # ========================================================
    # EMBEDDED IMAGE EXTRACTION
    # ========================================================

    def _extract_embedded_images(
        self,
        doc,
        page,
        output_dir: Path,
        page_number: int
    ) -> List[Dict[str, Any]]:

        """
        Extract actual embedded PDF images.

        IMPORTANT:
        This is independent of OCR.

        A PDF page may have:
            native text + photographs
            native text + scanned image
            drawings + embedded images

        We therefore extract embedded images regardless
        of whether OCR is required.
        """

        if not PDF_USE_EMBEDDED_IMAGES:

            return []

        images = []

        try:

            page_images = (
                page.get_images(
                    full=True
                )
            )

            if not page_images:

                return []

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            seen_xrefs = set()

            count = 0

            for image_info in page_images:

                if count >= PDF_MAX_EMBEDDED_IMAGES:
                    break

                if not image_info:
                    continue

                xref = image_info[0]

                if xref in seen_xrefs:
                    continue

                seen_xrefs.add(
                    xref
                )

                try:

                    image_data = (
                        doc.extract_image(
                            xref
                        )
                    )

                    if not image_data:
                        continue

                    image_bytes = (
                        image_data.get(
                            "image"
                        )
                    )

                    extension = (
                        image_data.get(
                            "ext",
                            "png"
                        )
                    )

                    width = (
                        image_data.get(
                            "width",
                            0
                        )
                    )

                    height = (
                        image_data.get(
                            "height",
                            0
                        )
                    )

                    if not image_bytes:
                        continue

                    output_path = (
                        output_dir
                        /
                        (
                            f"page_{page_number}"
                            f"_image_{count + 1}"
                            f".{extension}"
                        )
                    )

                    with open(
                        output_path,
                        "wb"
                    ) as f:

                        f.write(
                            image_bytes
                        )

                    images.append(
                        {

                            "image_id": (
                                f"page_{page_number}"
                                f"_image_{count + 1}"
                            ),

                            "path": str(
                                output_path
                            ),

                            "source": "embedded",

                            "xref": xref,

                            "page": page_number,

                            "width": width,

                            "height": height,

                            "extension": extension,
                        }
                    )

                    count += 1

                except Exception as exc:

                    print(
                        "[PDF IMAGE] "
                        f"xref={xref} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

        except Exception as exc:

            print(
                "[PDF IMAGE] "
                f"Extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

        return images

    # ========================================================
    # PAGE BLOCK EXTRACTION
    # ========================================================

    def _extract_page_blocks(
        self,
        page
    ) -> List[Dict[str, Any]]:

        if not LAYOUT_EXTRACTION_ENABLED:

            return []

        blocks = []

        try:

            data = page.get_text(
                "dict"
            )

            for block in data.get(
                "blocks",
                []
            ):

                block_type = block.get(
                    "type"
                )

                bbox = block.get(
                    "bbox"
                )

                if block_type == 0:

                    lines = []

                    for line in block.get(
                        "lines",
                        []
                    ):

                        line_text = ""

                        for span in line.get(
                            "spans",
                            []
                        ):

                            line_text += (
                                str(
                                    span.get(
                                        "text",
                                        ""
                                    )
                                )
                            )

                        line_text = (
                            self._clean_text(
                                line_text
                            )
                        )

                        if line_text:

                            lines.append(
                                {
                                    "text": line_text,
                                    "bbox": line.get(
                                        "bbox"
                                    ),
                                }
                            )

                    text = self._clean_text(
                        " ".join(
                            line["text"]
                            for line in lines
                        )
                    )

                    if text:

                        blocks.append(
                            {
                                "type": "text",
                                "text": text,
                                "bbox": bbox,
                                "lines": lines,
                            }
                        )

                else:

                    blocks.append(
                        {
                            "type": "non_text",
                            "bbox": bbox,
                        }
                    )

        except Exception as exc:

            print(
                "[LAYOUT] "
                f"Block extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

        return blocks

    # ========================================================
    # VECTOR / DRAWING EXTRACTION
    # ========================================================

    def _extract_vector_layout(
        self,
        page
    ) -> List[Dict[str, Any]]:

        if not VECTOR_EXTRACTION_ENABLED:

            return []

        result = []

        try:

            drawings = page.get_drawings()

            for index, drawing in enumerate(
                drawings
            ):

                items = []

                for item in drawing.get(
                    "items",
                    []
                ):

                    try:

                        item_type = (
                            item[0]
                            if item
                            else None
                        )

                        if item_type == "l":

                            p1 = item[1]
                            p2 = item[2]

                            items.append(
                                {
                                    "type": "line",
                                    "start": [
                                        p1.x,
                                        p1.y
                                    ],
                                    "end": [
                                        p2.x,
                                        p2.y
                                    ],
                                }
                            )

                        elif item_type == "re":

                            rect = item[1]

                            items.append(
                                {
                                    "type": "rectangle",
                                    "bbox": [
                                        rect.x0,
                                        rect.y0,
                                        rect.x1,
                                        rect.y1
                                    ],
                                }
                            )

                        elif item_type == "c":

                            points = []

                            for point in item[1:]:

                                if hasattr(
                                    point,
                                    "x"
                                ):

                                    points.append(
                                        [
                                            point.x,
                                            point.y
                                        ]
                                    )

                            items.append(
                                {
                                    "type": "curve",
                                    "points": points,
                                }
                            )

                        elif item_type == "qu":

                            quad = item[1]

                            points = []

                            for point in [
                                quad.ul,
                                quad.ur,
                                quad.ll,
                                quad.lr
                            ]:

                                points.append(
                                    [
                                        point.x,
                                        point.y
                                    ]
                                )

                            items.append(
                                {
                                    "type": "quad",
                                    "points": points,
                                }
                            )

                    except Exception:
                        continue

                rect = drawing.get(
                    "rect"
                )

                result.append(
                    {

                        "object_id": (
                            f"vector_{index + 1}"
                        ),

                        "type": "drawing",

                        "bbox": (
                            [
                                rect.x0,
                                rect.y0,
                                rect.x1,
                                rect.y1
                            ]
                            if rect
                            else None
                        ),

                        "items": items,

                        "fill": drawing.get(
                            "fill"
                        ),

                        "color": drawing.get(
                            "color"
                        ),

                        "width": drawing.get(
                            "width"
                        ),
                    }
                )

        except Exception as exc:

            print(
                "[VECTOR] "
                f"Extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

        return result

    # ========================================================
    # SIMPLE TABLE EXTRACTION
    # ========================================================

    def _extract_tables(
        self,
        page
    ) -> List[Dict[str, Any]]:

        if not TABLE_EXTRACTION_ENABLED:

            return []

        tables = []

        try:

            # PyMuPDF table extraction is
            # available only in supported versions.

            if not hasattr(
                page,
                "find_tables"
            ):

                return []

            finder = page.find_tables()

            table_objects = getattr(
                finder,
                "tables",
                []
            )

            for index, table in enumerate(
                table_objects
            ):

                try:

                    extracted = table.extract()

                    if not extracted:
                        continue

                    rows = []

                    for row in extracted:

                        if row is None:
                            row = []

                        cleaned_row = []

                        for cell in row:

                            cleaned_row.append(
                                self._clean_text(
                                    cell
                                )
                            )

                        rows.append(
                            cleaned_row
                        )

                    if not rows:
                        continue

                    headers = []

                    if rows:

                        headers = rows[0]

                    columns = []

                    max_width = max(
                        (
                            len(row)
                            for row in rows
                        ),
                        default=0
                    )

                    for col_index in range(
                        max_width
                    ):

                        columns.append(
                            f"column_{col_index + 1}"
                        )

                    tables.append(
                        {

                            "table_id": (
                                f"table_{index + 1}"
                            ),

                            "page": None,

                            "rows": rows,

                            "columns": columns,

                            "headers": headers,

                            "cells": rows,

                            "source": "pymupdf",
                        }
                    )

                except Exception as exc:

                    print(
                        "[TABLE] "
                        f"Table {index + 1} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

        except Exception as exc:

            print(
                "[TABLE] "
                f"Extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

        return tables

    # ========================================================
    # DOCLING
    # ========================================================

    def _process_with_docling(
        self,
        file_path: str
    ) -> Optional[Dict[str, Any]]:

        """
        Optional native Docling processing for non-PDF
        document formats.

        PDF is intentionally handled by PyMuPDF first
        because this loader needs explicit control over:
            native text
            OCR decisions
            embedded images
            page rendering
            vector information.
        """

        try:

            from docling.document_converter import (
                DocumentConverter
            )

            print(
                "[DOCLING] Loading: "
                f"{Path(file_path).name}"
            )

            converter = (
                DocumentConverter()
            )

            result = converter.convert(
                file_path
            )

            document = (
                result.document
            )

            markdown = (
                document.export_to_markdown()
            )

            markdown = self._clean_text(
                markdown
            )

            return {

                "file_name": Path(
                    file_path
                ).name,

                "file_path": str(
                    file_path
                ),

                "page_count": 1,

                "text": markdown,

                "images": [],

                "tables": [],

                "layout": [],

                "pages": [
                    {

                        "page_number": 1,

                        "native_text": markdown,

                        "ocr_text": "",

                        "text": markdown,

                        "ocr": {},

                        "images": [],

                        "tables": [],

                        "layout": [],
                    }
                ],

                "metadata": {

                    "file_name": Path(
                        file_path
                    ).name,

                    "file_type": (
                        Path(file_path)
                        .suffix
                        .lower()
                    ),

                    "document_modality": (
                        "document"
                    ),

                    "loader": "docling",
                },
            }

        except Exception as exc:

            print(
                "[DOCLING] Failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return None

    # ========================================================
    # PDF PROCESSING
    # ========================================================

    def _process_pdf(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        start = time.time()

        pdf_path = Path(
            file_path
        )

        visual_dir = (
            pdf_path.parent
            /
            "processed_visuals"
            /
            pdf_path.stem
        )

        image_dir = (
            pdf_path.parent
            /
            "extracted_images"
            /
            pdf_path.stem
        )

        doc = fitz.open(
            file_path
        )

        pages = []

        all_text = []

        all_images = []

        all_tables = []

        all_layout = []

        native_text_pages = 0

        ocr_pages = 0

        vector_object_count = 0

        embedded_image_count = 0

        try:

            page_count = len(
                doc
            )

            print(
                f"[PDF] Pages: "
                f"{page_count}"
            )

            # =================================================
            # PAGE LOOP
            # =================================================

            for page_index in range(
                page_count
            ):

                page_number = (
                    page_index + 1
                )

                page = doc[
                    page_index
                ]

                page_start = time.time()

                # -------------------------------------------------
                # NATIVE TEXT FIRST
                # -------------------------------------------------

                try:

                    native_text = (
                        page.get_text(
                            "text"
                        )
                    )

                except Exception:

                    native_text = ""

                native_text = (
                    self._clean_text(
                        native_text
                    )
                )

                has_native_text = (
                    len(native_text)
                    >= PDF_OCR_IF_TEXT_CHARS_BELOW
                )

                if has_native_text:

                    native_text_pages += 1

                # -------------------------------------------------
                # BLOCKS
                # -------------------------------------------------

                blocks = (
                    self._extract_page_blocks(
                        page
                    )
                )

                # -------------------------------------------------
                # VECTOR LAYOUT
                # -------------------------------------------------

                vector_layout = (
                    self._extract_vector_layout(
                        page
                    )
                )

                vector_object_count += len(
                    vector_layout
                )

                # -------------------------------------------------
                # TABLES
                # -------------------------------------------------

                page_tables = (
                    self._extract_tables(
                        page
                    )
                )

                for table in page_tables:

                    table["page"] = (
                        page_number
                    )

                    all_tables.append(
                        table
                    )

                # -------------------------------------------------
                # EMBEDDED IMAGES
                #
                # IMPORTANT:
                # ALWAYS EXTRACT.
                # NOT DEPENDENT ON OCR.
                # -------------------------------------------------

                embedded_images = (
                    self._extract_embedded_images(
                        doc,
                        page,
                        image_dir,
                        page_number
                    )
                )

                embedded_image_count += (
                    len(embedded_images)
                )

                all_images.extend(
                    embedded_images
                )

                # -------------------------------------------------
                # OCR DECISION
                # -------------------------------------------------

                needs_ocr = (
                    not has_native_text
                )

                ocr_text = ""

                ocr_result = {}

                rendered_path = None

                # -------------------------------------------------
                # RENDER PAGE
                #
                # Low-resolution render for UI.
                # Only OCR render at high resolution
                # when OCR is actually required.
                # -------------------------------------------------

                if PDF_RENDER_VISUALS:

                    rendered_path = (
                        self._render_page(
                            page,
                            visual_dir,
                            page_number,
                            PDF_RENDER_SCALE
                        )
                    )

                # -------------------------------------------------
                # OCR
                # -------------------------------------------------

                if needs_ocr:

                    ocr_pages += 1

                    ocr_path = (
                        self._render_page(
                            page,
                            visual_dir,
                            page_number,
                            PDF_OCR_RENDER_SCALE
                        )
                    )

                    if ocr_path:

                        try:

                            ocr_image = (
                                cv2.imread(
                                    ocr_path,
                                    cv2.IMREAD_COLOR
                                )
                            )

                            if ocr_image is not None:

                                ocr_result = (
                                    self._perform_ocr(
                                        ocr_image
                                    )
                                )

                                ocr_text = (
                                    self._clean_text(
                                        ocr_result.get(
                                            "text",
                                            ""
                                        )
                                    )
                                )

                        except Exception as exc:

                            print(
                                "[PDF OCR] "
                                f"Page {page_number} failed: "
                                f"{type(exc).__name__}: {exc}"
                            )

                # -------------------------------------------------
                # FINAL PAGE TEXT
                # -------------------------------------------------

                page_text_parts = []

                if native_text:

                    page_text_parts.append(
                        native_text
                    )

                if ocr_text:

                    # Avoid duplicating native text
                    # when OCR is used as supplemental evidence.

                    if (
                        not native_text
                        or
                        ocr_text
                        not in native_text
                    ):

                        page_text_parts.append(
                            ocr_text
                        )

                page_text = self._clean_text(
                    "\n".join(
                        page_text_parts
                    )
                )

                if page_text:

                    all_text.append(
                        page_text
                    )

                # -------------------------------------------------
                # PAGE RECORD
                # -------------------------------------------------

                page_record = {

                    "page_number": page_number,

                    "native_text": native_text,

                    "ocr_text": ocr_text,

                    "text": page_text,

                    "ocr": ocr_result,

                    "rendered_path": (
                        rendered_path
                    ),

                    "blocks": blocks,

                    "layout": vector_layout,

                    "images": embedded_images,

                    "tables": page_tables,

                    "metadata": {

                        "has_native_text": (
                            bool(native_text)
                        ),

                        "native_text_characters": (
                            len(native_text)
                        ),

                        "has_ocr": (
                            bool(ocr_text)
                        ),

                        "ocr_characters": (
                            len(ocr_text)
                        ),

                        "needs_ocr": needs_ocr,

                        "embedded_image_count": (
                            len(embedded_images)
                        ),

                        "vector_object_count": (
                            len(vector_layout)
                        ),
                    },
                }

                pages.append(
                    page_record
                )

                page_elapsed = (
                    time.time()
                    - page_start
                )

                print(
                    f"[PDF PAGE] "
                    f"{page_number}/{page_count} "
                    f"native_chars={len(native_text)} "
                    f"ocr={'yes' if ocr_text else 'no'} "
                    f"images={len(embedded_images)} "
                    f"vectors={len(vector_layout)} "
                    f"time={page_elapsed:.2f}s"
                )

        finally:

            doc.close()

        elapsed = (
            time.time()
            - start
        )

        print(
            f"[PDF] Completed "
            f"{pdf_path.name} "
            f"in {elapsed:.2f}s"
        )

        return {

            "file_name": pdf_path.name,

            "file_path": str(
                pdf_path
            ),

            "page_count": len(
                pages
            ),

            "text": self._clean_text(
                "\n\n".join(
                    all_text
                )
            ),

            "images": all_images,

            "tables": all_tables,

            "layout": all_layout,

            "pages": pages,

            "metadata": {

                "file_name": pdf_path.name,

                "file_type": ".pdf",

                "document_modality": (
                    "pdf"
                ),

                "page_count": len(
                    pages
                ),

                "native_text_pages": (
                    native_text_pages
                ),

                "ocr_pages": (
                    ocr_pages
                ),

                "embedded_image_count": (
                    embedded_image_count
                ),

                "vector_object_count": (
                    vector_object_count
                ),

                "processing_time": elapsed,

                "loader": "pymupdf",
            },
        }

    # ========================================================
    # PLAIN TEXT
    # ========================================================

    def _process_plain_text(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        path = Path(
            file_path
        )

        try:

            text = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            text = path.read_text(
                encoding="latin-1"
            )

        text = self._clean_text(
            text
        )

        return {

            "file_name": path.name,

            "file_path": str(path),

            "page_count": 1,

            "text": text,

            "images": [],

            "tables": [],

            "layout": [],

            "pages": [
                {

                    "page_number": 1,

                    "native_text": text,

                    "ocr_text": "",

                    "text": text,

                    "ocr": {},

                    "images": [],

                    "tables": [],

                    "layout": [],
                }
            ],

            "metadata": {

                "file_name": path.name,

                "file_type": (
                    path.suffix.lower()
                ),

                "document_modality": (
                    "text"
                ),

                "loader": "native",
            },
        }

    # ========================================================
    # MAIN LOAD
    # ========================================================

    def load(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        path = Path(
            file_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: "
                f"{file_path}"
            )

        extension = (
            path.suffix
            .lower()
        )

        print()
        print(
            f"[LOADER] Loading: "
            f"{path.name}"
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if extension == ".pdf":

            return self._process_pdf(
                str(path)
            )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if extension in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
        }:

            return self._process_image(
                str(path)
            )

        # ----------------------------------------------------
        # PLAIN TEXT
        # ----------------------------------------------------

        if extension in {
            ".txt",
            ".md",
            ".csv",
        }:

            return self._process_plain_text(
                str(path)
            )

        # ----------------------------------------------------
        # DOCX / XLSX / PPTX
        # ----------------------------------------------------

        if extension in {
            ".docx",
            ".xlsx",
            ".xls",
            ".pptx",
        }:

            result = (
                self._process_with_docling(
                    str(path)
                )
            )

            if result is not None:

                return result

            # Fallback to plain text only
            # if Docling fails.

            try:

                return self._process_plain_text(
                    str(path)
                )

            except Exception:

                raise RuntimeError(
                    "Unable to process "
                    f"{path.name}"
                )

        # ----------------------------------------------------
        # GENERIC DOCLING FALLBACK
        # ----------------------------------------------------

        result = (
            self._process_with_docling(
                str(path)
            )
        )

        if result is not None:

            return result

        raise ValueError(
            "Unsupported or unreadable "
            f"document format: {extension}"
        )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

GenericDocumentLoader = DocumentLoader
DocumentLoaderAgent = DocumentLoader