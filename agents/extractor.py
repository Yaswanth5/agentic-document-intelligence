import json
import time
from typing import Any, Dict, List

from agents.llm import LocalLLM


class GenericExtractor:

    """
    Generic document extraction engine.

    Important architecture:

        deterministic extraction
                +
        compact LLM semantic extraction

    The LLM is NOT used to count things that can be
    counted programmatically.

    The LLM is used for:
        - summary
        - semantic entities
        - relationships
        - important information
        - interpretation
        - uncertainty

    Deterministic extraction handles:
        - pages
        - images
        - tables
        - rows
        - columns
        - layout objects
        - text statistics
    """

    def __init__(self):

        self.llm = LocalLLM()

    # ============================================================
    # EMPTY RESULT
    # ============================================================

    def _empty_result(self):

        return {

            "document_summary": "",

            "document_type": "",

            "languages": [],

            "entities": [],

            "fields": [],

            "measurements": [],

            "relationships": [],

            "tables": [],

            "images": [],

            "drawings": [],

            "spatial_information": [],

            "layout_information": [],

            "key_information": [],

            "uncertainties": [],

            "evidence": [],

        }

    # ============================================================
    # TABLE STRUCTURE
    # ============================================================

    def _deterministic_table_summary(
        self,
        tables: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        result = []

        for index, table in enumerate(
            tables or []
        ):

            if not isinstance(
                table,
                dict
            ):
                continue

            rows = table.get(
                "rows",
                []
            )

            columns = table.get(
                "columns",
                []
            )

            headers = table.get(
                "headers",
                []
            )

            # ----------------------------------------------------
            # Column count
            # ----------------------------------------------------

            column_count = len(
                columns
            )

            if column_count == 0:

                max_row_width = 0

                for row in rows:

                    if isinstance(
                        row,
                        dict
                    ):

                        cells = row.get(
                            "cells",
                            []
                        )

                    else:

                        cells = row

                    if isinstance(
                        cells,
                        list
                    ):

                        max_row_width = max(
                            max_row_width,
                            len(cells)
                        )

                column_count = max_row_width

            # ----------------------------------------------------
            # Row count
            # ----------------------------------------------------

            row_count = len(
                rows
            )

            result.append(
                {

                    "table_id": table.get(
                        "table_id",
                        f"table_{index + 1}"
                    ),

                    "page": table.get(
                        "page"
                    ),

                    "row_count": row_count,

                    "column_count": column_count,

                    "headers": headers,

                    "columns": columns,

                    "rows": rows,

                    "cells": table.get(
                        "cells",
                        []
                    ),

                    "source": table.get(
                        "source"
                    ),

                }
            )

        return result

    # ============================================================
    # IMAGE SUMMARY
    # ============================================================

    def _deterministic_image_summary(
        self,
        document: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        result = []

        images = document.get(
            "images",
            []
        )

        for index, image in enumerate(
            images
        ):

            if not isinstance(
                image,
                dict
            ):
                continue

            result.append(
                {

                    "image_id": image.get(
                        "image_id",
                        f"image_{index + 1}"
                    ),

                    "page": image.get(
                        "page"
                    ),

                    "path": image.get(
                        "path"
                    ),

                    "source": image.get(
                        "source"
                    ),

                    "width": image.get(
                        "width"
                    ),

                    "height": image.get(
                        "height"
                    ),

                    "extension": image.get(
                        "extension"
                    ),

                    "bbox": image.get(
                        "bbox"
                    ),

                }
            )

        return result

    # ============================================================
    # STRUCTURAL SUMMARY
    # ============================================================

    def _build_structure(
        self,
        document: Dict[str, Any],
        deterministic_tables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        pages = document.get(
            "pages",
            []
        )

        images = document.get(
            "images",
            []
        )

        layout = document.get(
            "layout",
            []
        )

        return {

            "page_count": document.get(
                "page_count",
                len(pages)
            ),

            "text_character_count": len(
                document.get(
                    "text",
                    ""
                )
            ),

            "image_count": len(
                images
            ),

            "table_count": len(
                deterministic_tables
            ),

            "layout_object_count": len(
                layout
            ),

            "pages_with_native_text": sum(
                bool(
                    page.get(
                        "native_text",
                        ""
                    )
                )
                for page in pages
                if isinstance(
                    page,
                    dict
                )
            ),

            "pages_with_ocr": sum(
                bool(
                    page.get(
                        "ocr_text",
                        ""
                    )
                )
                for page in pages
                if isinstance(
                    page,
                    dict
                )
            ),

        }

    # ============================================================
    # COMPACT LLM INPUT
    # ============================================================

    def _build_compact_context(
        self,
        document: Dict[str, Any]
    ) -> str:

        """
        Build a compact representation for Gemma.

        DO NOT send every chunk's duplicated metadata.

        This dramatically reduces prompt size and therefore
        CPU inference work.
        """

        parts = []

        # --------------------------------------------------------
        # Page text
        # --------------------------------------------------------

        pages = document.get(
            "pages",
            []
        )

        for page in pages:

            if not isinstance(
                page,
                dict
            ):
                continue

            page_number = page.get(
                "page_number"
            )

            text = page.get(
                "text",
                ""
            )

            if text:

                parts.append(
                    f"PAGE {page_number}\n"
                    f"{text}"
                )

        # --------------------------------------------------------
        # If pages don't contain text,
        # use document text.
        # --------------------------------------------------------

        if not parts:

            text = document.get(
                "text",
                ""
            )

            if text:

                parts.append(
                    f"DOCUMENT TEXT\n{text}"
                )

        # --------------------------------------------------------
        # Hard safety limit.
        #
        # The full document is not always appropriate for a
        # local 4B model.
        # --------------------------------------------------------

        context = "\n\n".join(
            parts
        )

        max_chars = 12000

        if len(context) > max_chars:

            context = (
                context[:max_chars]
                + "\n\n[TEXT TRUNCATED FOR "
                  "LOCAL SEMANTIC EXTRACTION]"
            )

        return context

    # ============================================================
    # BUILD PROMPT
    # ============================================================

    def _build_prompt(
        self,
        document: Dict[str, Any],
        deterministic_tables: List[Dict[str, Any]]
    ) -> str:

        context = self._build_compact_context(
            document
        )

        structure = self._build_structure(
            document,
            deterministic_tables
        )

        # Only send a compact structural summary.
        image_summary = (
            self._deterministic_image_summary(
                document
            )
        )

        # Don't send huge raw image objects.
        if len(image_summary) > 20:

            image_summary = image_summary[:20]

        return f"""
You are a generic document intelligence engine.

Analyze ONLY the supplied document evidence.

Do not invent information.

The document may be any type:
business, financial, legal, medical, technical,
engineering, architectural, scanned, multilingual,
spreadsheet, presentation, form, report, invoice,
blueprint or other document.

Your job is semantic interpretation.

Deterministic structural facts have already been
calculated separately. Do not override them.

RULES:

1. Do not hallucinate.
2. Use only supplied evidence.
3. Preserve exact values where possible.
4. Preserve page numbers.
5. Do not guess missing values.
6. Do not create document-specific fields unless evidence supports them.
7. Do not count table rows or columns yourself.
8. Do not claim an image contains something unless the supplied evidence supports it.
9. If information is unavailable, return null or an empty list.
10. Keep the response concise.

Return JSON only.

JSON structure:

{{
  "document_summary": "",
  "document_type": "",
  "languages": [],
  "entities": [],
  "fields": [],
  "measurements": [],
  "relationships": [],
  "key_information": [],
  "uncertainties": [],
  "evidence": []
}}

For fields:

{{
  "name": "",
  "value": "",
  "page": 0,
  "confidence": 0.0,
  "source": ""
}}

For measurements:

{{
  "value": "",
  "unit": "",
  "description": "",
  "page": 0,
  "source": ""
}}

For relationships:

{{
  "subject": "",
  "relation": "",
  "object": "",
  "page": 0,
  "source": ""
}}

DOCUMENT STRUCTURE:

{json.dumps(
    structure,
    ensure_ascii=False
)}

AVAILABLE IMAGES:

{json.dumps(
    image_summary,
    ensure_ascii=False
)}

DOCUMENT TEXT:

{context}
""".strip()

    # ============================================================
    # EXTRACT
    # ============================================================

    def extract(
        self,
        chunks: List[Dict[str, Any]],
        document: Dict[str, Any]
    ) -> Dict[str, Any]:

        total_start = time.time()

        if not isinstance(
            document,
            dict
        ):
            document = {}

        # --------------------------------------------------------
        # Deterministic extraction
        # --------------------------------------------------------

        deterministic_tables = (
            self._deterministic_table_summary(
                document.get(
                    "tables",
                    []
                )
            )
        )

        # Page-level tables.
        for page in document.get(
            "pages",
            []
        ):

            if not isinstance(
                page,
                dict
            ):
                continue

            page_tables = page.get(
                "tables",
                []
            )

            deterministic_tables.extend(
                self._deterministic_table_summary(
                    page_tables
                )
            )

        # --------------------------------------------------------
        # Remove duplicate table IDs.
        # --------------------------------------------------------

        unique_tables = []

        seen_table_ids = set()

        for table in deterministic_tables:

            table_id = table.get(
                "table_id"
            )

            if table_id in seen_table_ids:
                continue

            seen_table_ids.add(
                table_id
            )

            unique_tables.append(
                table
            )

        deterministic_tables = (
            unique_tables
        )

        # --------------------------------------------------------
        # Build compact LLM prompt
        # --------------------------------------------------------

        prompt = self._build_prompt(
            document,
            deterministic_tables
        )

        print(
            f"[EXTRACTOR] "
            f"Compact prompt chars="
            f"{len(prompt)}"
        )

        # --------------------------------------------------------
        # LLM semantic extraction
        # --------------------------------------------------------

        llm_result = self.llm.generate_json(
            prompt,
            num_predict=256
        )

        if not isinstance(
            llm_result,
            dict
        ):
            llm_result = {}

        # --------------------------------------------------------
        # Ensure standard keys.
        # --------------------------------------------------------

        result = self._empty_result()

        result.update(
            llm_result
        )

        # --------------------------------------------------------
        # DETERMINISTIC TABLES ALWAYS WIN
        # --------------------------------------------------------

        result["tables"] = (
            deterministic_tables
        )

        # --------------------------------------------------------
        # Deterministic images always preserved.
        # --------------------------------------------------------

        result["images"] = (
            self._deterministic_image_summary(
                document
            )
        )

        # --------------------------------------------------------
        # Structural facts
        # --------------------------------------------------------

        result["document_structure"] = (
            self._build_structure(
                document,
                deterministic_tables
            )
        )

        # --------------------------------------------------------
        # Explicit structural evidence
        # --------------------------------------------------------

        result["structural_evidence"] = {

            "page_count": document.get(
                "page_count"
            ),

            "image_count": len(
                document.get(
                    "images",
                    []
                )
            ),

            "table_count": len(
                deterministic_tables
            ),

            "layout_object_count": len(
                document.get(
                    "layout",
                    []
                )
            ),

        }

        elapsed = (
            time.time()
            - total_start
        )

        print(
            f"[EXTRACTOR] Completed "
            f"in {elapsed:.2f}s"
        )

        return result


# ================================================================
# BACKWARD COMPATIBILITY
# ================================================================

ExtractionAgent = GenericExtractor
DocumentExtractor = GenericExtractor