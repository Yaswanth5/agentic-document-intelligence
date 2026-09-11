import re
from collections import Counter


class DocumentClassifier:
    """
    Generic, document-agnostic document classifier.

    Responsibilities:
    - Detect likely language/script
    - Detect broad document type
    - Detect presence of tables
    - Detect broad business/domain signals
    - Detect document modality

    This classifier does NOT contain document-specific rules.
    """

    SCRIPT_RANGES = {
        "Devanagari": [
            (0x0900, 0x097F),
        ],
        "Bengali": [
            (0x0980, 0x09FF),
        ],
        "Gujarati": [
            (0x0A80, 0x0AFF),
        ],
        "Gurmukhi": [
            (0x0A00, 0x0A7F),
        ],
        "Oriya": [
            (0x0B00, 0x0B7F),
        ],
        "Tamil": [
            (0x0B80, 0x0BFF),
        ],
        "Telugu": [
            (0x0C00, 0x0C7F),
        ],
        "Kannada": [
            (0x0C80, 0x0CFF),
        ],
        "Malayalam": [
            (0x0D00, 0x0D7F),
        ],
        "Arabic": [
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
        ],
        "Latin": [
            (0x0041, 0x005A),
            (0x0061, 0x007A),
            (0x00C0, 0x00FF),
            (0x0100, 0x024F),
        ],
    }

    LANGUAGE_BY_SCRIPT = {
        "Devanagari": "Hindi",
        "Bengali": "Bengali",
        "Gujarati": "Gujarati",
        "Gurmukhi": "Punjabi",
        "Oriya": "Odia",
        "Tamil": "Tamil",
        "Telugu": "Telugu",
        "Kannada": "Kannada",
        "Malayalam": "Malayalam",
        "Arabic": "Arabic",
    }

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # SCRIPT DETECTION
    # ---------------------------------------------------------

    def _char_script(self, char):
        code = ord(char)

        for script, ranges in self.SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code <= end:
                    return script

        return None

    def _detect_scripts(self, text):
        if not text:
            return Counter()

        counts = Counter()

        for char in text:
            script = self._char_script(char)

            if script:
                counts[script] += 1

        return counts

    # ---------------------------------------------------------
    # LANGUAGE DETECTION
    # ---------------------------------------------------------

    def _detect_language(self, text):
        if not text or not text.strip():
            return {
                "language": "unknown",
                "languages": [],
                "scripts": [],
            }

        script_counts = self._detect_scripts(text)

        total_script_chars = sum(script_counts.values())

        if total_script_chars == 0:
            return {
                "language": "unknown",
                "languages": [],
                "scripts": [],
            }

        # Ignore extremely small script fragments.
        meaningful_scripts = [
            script
            for script, count in script_counts.items()
            if count >= 5
        ]

        meaningful_scripts.sort(
            key=lambda x: script_counts[x],
            reverse=True
        )

        # -----------------------------------------------------
        # Non-Latin scripts
        # -----------------------------------------------------

        detected_languages = []

        for script in meaningful_scripts:
            language = self.LANGUAGE_BY_SCRIPT.get(script)

            if language:
                detected_languages.append(language)

        # -----------------------------------------------------
        # Multiple scripts = multilingual
        # -----------------------------------------------------

        if len(detected_languages) >= 2:
            return {
                "language": "multilingual",
                "languages": detected_languages,
                "scripts": meaningful_scripts,
            }

        if len(detected_languages) == 1:
            return {
                "language": detected_languages[0],
                "languages": detected_languages,
                "scripts": meaningful_scripts,
            }

        # -----------------------------------------------------
        # Latin script
        #
        # Do NOT automatically call Latin = English.
        # Latin can represent many languages.
        # -----------------------------------------------------

        if "Latin" in meaningful_scripts:

            english_words = {
                "the",
                "and",
                "of",
                "to",
                "in",
                "for",
                "is",
                "are",
                "this",
                "that",
                "document",
                "name",
                "date",
                "address",
                "number",
                "total",
                "amount",
                "page",
                "certificate",
                "account",
                "customer",
                "company",
                "application",
                "details",
            }

            words = re.findall(
                r"\b[a-zA-Z]{2,}\b",
                text.lower()
            )

            english_hits = sum(
                1 for word in words
                if word in english_words
            )

            if english_hits >= 2:
                return {
                    "language": "English",
                    "languages": ["English"],
                    "scripts": meaningful_scripts,
                }

            return {
                "language": "unknown",
                "languages": [],
                "scripts": meaningful_scripts,
            }

        return {
            "language": "unknown",
            "languages": [],
            "scripts": meaningful_scripts,
        }

    # ---------------------------------------------------------
    # TABLE DETECTION
    # ---------------------------------------------------------

    def _detect_tables(self, text, document=None):
        if not text:
            return False

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return False

        # Markdown table
        markdown_table = any(
            "|" in line and "---" in line
            for line in lines
        )

        if markdown_table:
            return True

        # Repeated column-like separators
        separator_count = sum(
            1
            for line in lines
            if line.count("|") >= 2
            or line.count("\t") >= 2
        )

        if separator_count >= 2:
            return True

        # Strong whitespace-column pattern
        column_like_lines = 0

        for line in lines:
            parts = re.split(r"\s{3,}", line)

            if len(parts) >= 3:
                column_like_lines += 1

        if column_like_lines >= 3:
            return True

        # Structured document metadata may already tell us.
        if isinstance(document, dict):

            tables = document.get("tables")

            if isinstance(tables, list) and len(tables) > 0:
                return True

        return False

    # ---------------------------------------------------------
    # DOCUMENT TYPE
    # ---------------------------------------------------------

    def _detect_document_type(self, text, has_tables=False):
        if not text:
            return "unknown"

        lowered = text.lower()

        if has_tables:
            return "structured_table_document"

        if any(
            keyword in lowered
            for keyword in [
                "invoice",
                "bill to",
                "tax invoice",
                "amount due",
                "invoice number",
            ]
        ):
            return "invoice"

        if any(
            keyword in lowered
            for keyword in [
                "application",
                "applicant",
                "form number",
                "application number",
            ]
        ):
            return "application_form"

        if any(
            keyword in lowered
            for keyword in [
                "certificate",
                "certify that",
                "certificate number",
            ]
        ):
            return "certificate"

        if any(
            keyword in lowered
            for keyword in [
                "statement",
                "account statement",
                "transaction date",
                "opening balance",
                "closing balance",
            ]
        ):
            return "financial_statement"

        if any(
            keyword in lowered
            for keyword in [
                "agreement",
                "terms and conditions",
                "party of the first part",
                "party of the second part",
            ]
        ):
            return "agreement"

        if any(
            keyword in lowered
            for keyword in [
                "report",
                "executive summary",
                "conclusion",
                "findings",
            ]
        ):
            return "report"

        if any(
            keyword in lowered
            for keyword in [
                "drawing",
                "floor plan",
                "section",
                "elevation",
                "column",
                "beam",
                "foundation",
            ]
        ):
            return "technical_drawing"

        return "general_document"

    # ---------------------------------------------------------
    # DOMAIN DETECTION
    # ---------------------------------------------------------

    def _detect_domain(self, text):
        if not text:
            return "general"

        lowered = text.lower()

        scores = {
            "financial": 0,
            "legal": 0,
            "medical": 0,
            "education": 0,
            "technical": 0,
            "government": 0,
            "business": 0,
        }

        domain_keywords = {
            "financial": [
                "loan",
                "credit",
                "debit",
                "account",
                "balance",
                "interest",
                "amount",
                "transaction",
                "bank",
                "emi",
                "payment",
            ],
            "legal": [
                "agreement",
                "contract",
                "clause",
                "court",
                "legal",
                "party",
                "law",
                "jurisdiction",
            ],
            "medical": [
                "patient",
                "diagnosis",
                "hospital",
                "doctor",
                "clinical",
                "medicine",
                "prescription",
                "treatment",
            ],
            "education": [
                "student",
                "school",
                "college",
                "university",
                "marks",
                "grade",
                "semester",
                "examination",
            ],
            "technical": [
                "drawing",
                "engineering",
                "dimension",
                "diameter",
                "beam",
                "column",
                "structure",
                "specification",
                "material",
            ],
            "government": [
                "government",
                "department",
                "district",
                "mandal",
                "municipality",
                "authority",
                "official",
            ],
            "business": [
                "company",
                "business",
                "customer",
                "vendor",
                "employee",
                "revenue",
                "sales",
            ],
        }

        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in lowered:
                    scores[domain] += 1

        best_domain = max(
            scores,
            key=scores.get
        )

        if scores[best_domain] == 0:
            return "general"

        return best_domain

    # ---------------------------------------------------------
    # MODALITY
    # ---------------------------------------------------------

    def _detect_modality(self, document):
        if not isinstance(document, dict):
            return "text"

        has_images = bool(
            document.get("images")
        )

        has_tables = bool(
            document.get("tables")
        )

        has_layout = bool(
            document.get("layout")
        )

        has_vectors = bool(
            document.get("vector_objects")
        )

        modalities = []

        if document.get("text"):
            modalities.append("text")

        if has_tables:
            modalities.append("table")

        if has_images:
            modalities.append("image")

        if has_layout:
            modalities.append("layout")

        if has_vectors:
            modalities.append("vector")

        if not modalities:
            return "unknown"

        if len(modalities) == 1:
            return modalities[0]

        return "multimodal"

    # ---------------------------------------------------------
    # MAIN CLASSIFICATION
    # ---------------------------------------------------------

    def classify(self, document):
        """
        Classify a normalized document.

        Supports:
        - dict normalized documents
        - strings
        - lists
        - Docling-like objects
        """

        text = ""

        if isinstance(document, str):
            text = document

        elif isinstance(document, dict):
            text = str(
                document.get("text", "")
            )

        elif isinstance(document, list):
            text = "\n".join(
                str(item)
                for item in document
            )

        else:
            # Best-effort extraction from document objects.
            try:
                text = str(
                    getattr(
                        document,
                        "text",
                        ""
                    )
                )
            except Exception:
                text = ""

        language_info = self._detect_language(text)

        has_tables = self._detect_tables(
            text,
            document=document
        )

        document_type = self._detect_document_type(
            text,
            has_tables=has_tables
        )

        business_domain = self._detect_domain(text)

        modality = self._detect_modality(document)

        result = {
            "document_type": document_type,
            "business_domain": business_domain,
            "language": language_info["language"],
            "languages": language_info["languages"],
            "scripts": language_info["scripts"],
            "contains_tables": has_tables,
            "contains_financial_information": (
                business_domain == "financial"
            ),
            "document_modality": modality,
            "confidence": 0.0,
        }

        # Conservative confidence estimate.
        signals = 0

        if language_info["language"] != "unknown":
            signals += 1

        if document_type != "unknown":
            signals += 1

        if business_domain != "general":
            signals += 1

        if modality != "unknown":
            signals += 1

        result["confidence"] = round(
            signals / 4.0,
            2
        )

        return result

    # ---------------------------------------------------------
    # BACKWARD-COMPATIBILITY HELPERS
    # ---------------------------------------------------------

    def predict(self, document):
        return self.classify(document)

    def run(self, document):
        return self.classify(document)


# Existing code may use any of these names.
ClassifierAgent = DocumentClassifier
DocumentClassifierAgent = DocumentClassifier