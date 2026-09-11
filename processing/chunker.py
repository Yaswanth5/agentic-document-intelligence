from typing import Any, Dict, List
import re


class DocumentChunker:
    """
    Generic document chunker.

    Designed for:
        - normal text
        - page-aware documents
        - tables
        - OCR output
        - scanned documents
        - multilingual documents
        - forms
        - reports
        - financial statements
        - resumes
        - contracts
        - arbitrary document structures

    Important:
        Tables are kept as dedicated chunks whenever they can
        be detected from extracted text.
    """

    def __init__(
        self,
        chunk_size=2500,
        overlap=250,
        table_chunk_size=6000,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.table_chunk_size = table_chunk_size

    # =========================================================
    # SAFE TEXT
    # =========================================================

    @staticmethod
    def _text(value):
        if value is None:
            return ""

        return str(value)

    # =========================================================
    # TABLE SERIALIZATION
    # =========================================================

    def _serialize_table(
        self,
        table: Dict[str, Any],
    ) -> str:

        if not isinstance(table, dict):
            return ""

        lines = []

        table_id = table.get(
            "table_id",
            "",
        )

        rows = table.get(
            "rows",
            [],
        )

        columns = table.get(
            "columns",
            [],
        )

        headers = table.get(
            "headers",
            [],
        )

        lines.append(
            f"TABLE {table_id}"
        )

        lines.append(
            f"ROW_COUNT: {len(rows)}"
        )

        lines.append(
            f"COLUMN_COUNT: {len(columns)}"
        )

        if headers:
            lines.append(
                "HEADERS: "
                + " | ".join(
                    self._text(v)
                    for v in headers
                )
            )

        for row_index, row in enumerate(rows):

            if isinstance(row, dict):

                values = row.get(
                    "cells",
                    [],
                )

            else:
                values = row

            if not isinstance(values, list):
                values = [values]

            lines.append(
                f"ROW {row_index + 1}: "
                + " | ".join(
                    self._text(v)
                    for v in values
                )
            )

        return "\n".join(lines)

    # =========================================================
    # DETECT PIPE / MARKDOWN TABLES
    # =========================================================

    @staticmethod
    def _looks_like_table_line(line: str) -> bool:

        if "|" not in line:
            return False

        stripped = line.strip()

        if not stripped:
            return False

        parts = [
            part.strip()
            for part in stripped.strip("|").split("|")
        ]

        return len(parts) >= 2

    @staticmethod
    def _looks_like_separator(line: str) -> bool:

        stripped = line.strip()

        if "|" not in stripped:
            return False

        parts = [
            part.strip()
            for part in stripped.strip("|").split("|")
        ]

        if len(parts) < 2:
            return False

        return all(
            bool(
                re.fullmatch(
                    r":?-{2,}:?",
                    part,
                )
            )
            for part in parts
        )

    def _extract_text_tables(
        self,
        text: str,
    ):
        """
        Detect contiguous Markdown / pipe-style tables.

        Returns:

            normal_text
            detected_tables

        This is intentionally generic and does not depend
        on any particular document type.
        """

        if not text:
            return "", []

        lines = text.splitlines()

        normal_lines = []
        tables = []

        current_table = []

        def flush_table():
            nonlocal current_table

            if current_table:
                table_text = "\n".join(
                    current_table
                ).strip()

                if table_text:
                    tables.append(
                        table_text
                    )

            current_table = []

        index = 0

        while index < len(lines):

            line = lines[index]

            if self._looks_like_table_line(line):

                current_table.append(line)

                index += 1

                # Continue while subsequent lines look
                # like table rows / separators.
                while index < len(lines):

                    next_line = lines[index]

                    if (
                        self._looks_like_table_line(
                            next_line
                        )
                    ):
                        current_table.append(
                            next_line
                        )
                        index += 1
                    else:
                        break

                # Only regard it as a table if there are
                # enough structural lines.
                if len(current_table) >= 2:
                    flush_table()
                else:
                    normal_lines.extend(
                        current_table
                    )
                    current_table = []

                continue

            normal_lines.append(line)

            index += 1

        flush_table()

        return (
            "\n".join(normal_lines).strip(),
            tables,
        )

    # =========================================================
    # SPLIT TEXT
    # =========================================================

    def _split_text(
        self,
        text: str,
        chunk_size=None,
    ) -> List[str]:

        if not text:
            return []

        size = (
            chunk_size
            if chunk_size is not None
            else self.chunk_size
        )

        if len(text) <= size:
            return [text.strip()]

        chunks = []

        start = 0

        while start < len(text):

            end = min(
                start + size,
                len(text),
            )

            candidate = text[
                start:end
            ]

            # Prefer ending on a paragraph/newline
            # instead of cutting arbitrary text.
            if end < len(text):

                newline_pos = candidate.rfind(
                    "\n"
                )

                if newline_pos > size * 0.60:
                    end = (
                        start
                        + newline_pos
                    )

            chunk = text[
                start:end
            ].strip()

            if chunk:
                chunks.append(
                    chunk
                )

            if end >= len(text):
                break

            start = max(
                end - self.overlap,
                start + 1,
            )

        return chunks

    # =========================================================
    # CREATE CHUNK
    # =========================================================

    @staticmethod
    def _make_chunk(
        *,
        chunk_id,
        text,
        page,
        source,
        blocks=None,
        lines=None,
        images=None,
        tables=None,
        layout=None,
        metadata=None,
    ):

        return {
            "chunk_id": str(chunk_id),

            "text": str(text).strip(),

            "page": page,

            "source": source,

            "blocks": blocks or [],

            "lines": lines or [],

            "images": images or [],

            "tables": tables or [],

            "layout": layout or [],

            "metadata": metadata or {},
        }

    # =========================================================
    # MAIN
    # =========================================================

    def chunk(
        self,
        document: Any,
    ) -> List[Dict[str, Any]]:

        if document is None:
            return []

        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        if isinstance(
            document,
            str,
        ):

            normalized = {
                "source": "",
                "text": document,
                "pages": [],
                "tables": [],
                "images": [],
                "layout": [],
                "metadata": {},
            }

        elif isinstance(
            document,
            dict,
        ):

            normalized = document

        else:

            normalized = {
                "source": "",
                "text": str(document),
                "pages": [],
                "tables": [],
                "images": [],
                "layout": [],
                "metadata": {},
            }

        source = self._text(
            normalized.get(
                "source",
                "",
            )
        )

        chunks = []

        # =====================================================
        # PAGE LEVEL
        # =====================================================

        pages = normalized.get(
            "pages",
            [],
        )

        if isinstance(pages, list):

            for page_index, page in enumerate(
                pages
            ):

                if not isinstance(
                    page,
                    dict,
                ):
                    continue

                page_number = page.get(
                    "page_number",
                    page.get(
                        "page",
                        page_index + 1,
                    ),
                )

                page_text = self._text(
                    page.get(
                        "text",
                        "",
                    )
                )

                page_tables = page.get(
                    "tables",
                    [],
                )

                if not isinstance(
                    page_tables,
                    list,
                ):
                    page_tables = []

                images = page.get(
                    "images",
                    [],
                )

                layout = page.get(
                    "layout",
                    [],
                )

                blocks = page.get(
                    "blocks",
                    [],
                )

                lines = page.get(
                    "lines",
                    [],
                )

                # -------------------------------------------------
                # Detect tables embedded directly inside page text.
                # -------------------------------------------------

                normal_page_text, detected_tables = (
                    self._extract_text_tables(
                        page_text
                    )
                )

                # -------------------------------------------------
                # NORMAL TEXT CHUNKS
                # -------------------------------------------------

                text_chunks = self._split_text(
                    normal_page_text
                )

                for part_index, text_chunk in enumerate(
                    text_chunks
                ):

                    chunks.append(
                        self._make_chunk(
                            chunk_id=(
                                f"{source}:"
                                f"{page_number}:"
                                f"text:"
                                f"{part_index}"
                            ),
                            text=text_chunk,
                            page=page_number,
                            source=source,
                            blocks=blocks,
                            lines=lines,
                            images=images,
                            tables=page_tables,
                            layout=layout,
                            metadata={
                                "type": "text",
                                "page_number": page_number,
                                "page": page_number,
                                "source": source,
                                "has_text": True,
                                "table_count": len(
                                    page_tables
                                )
                                + len(
                                    detected_tables
                                ),
                                "image_count": len(
                                    images
                                ),
                                "layout_object_count": len(
                                    layout
                                ),
                            },
                        )
                    )

                # -------------------------------------------------
                # EMBEDDED TABLE CHUNKS
                # -------------------------------------------------

                for table_index, table_text in enumerate(
                    detected_tables
                ):

                    chunks.append(
                        self._make_chunk(
                            chunk_id=(
                                f"{source}:"
                                f"{page_number}:"
                                f"detected_table:"
                                f"{table_index}"
                            ),
                            text=(
                                "TABLE\n"
                                + table_text
                            ),
                            page=page_number,
                            source=source,
                            blocks=blocks,
                            lines=lines,
                            images=images,
                            tables=page_tables,
                            layout=layout,
                            metadata={
                                "type": "table",
                                "page_number": page_number,
                                "page": page_number,
                                "source": source,
                                "table_index": table_index,
                                "table_source": "page_text",
                            },
                        )
                    )

                # -------------------------------------------------
                # STRUCTURED TABLES
                # -------------------------------------------------

                for table_index, table in enumerate(
                    page_tables
                ):

                    serialized = self._serialize_table(
                        table
                    )

                    if not serialized:
                        continue

                    chunks.append(
                        self._make_chunk(
                            chunk_id=(
                                f"{source}:"
                                f"{page_number}:"
                                f"table:"
                                f"{table_index}"
                            ),
                            text=serialized,
                            page=page_number,
                            source=source,
                            blocks=blocks,
                            lines=lines,
                            images=images,
                            tables=[table],
                            layout=layout,
                            metadata={
                                "type": "table",
                                "page_number": page_number,
                                "page": page_number,
                                "source": source,
                                "table_index": table_index,
                                "table_id": table.get(
                                    "table_id"
                                ),
                                "row_count": len(
                                    table.get(
                                        "rows",
                                        [],
                                    )
                                ),
                                "column_count": len(
                                    table.get(
                                        "columns",
                                        [],
                                    )
                                ),
                            },
                        )
                    )

                # -------------------------------------------------
                # VISUAL-ONLY PAGE
                # -------------------------------------------------

                if (
                    not text_chunks
                    and not detected_tables
                    and not page_tables
                ):

                    chunks.append(
                        self._make_chunk(
                            chunk_id=(
                                f"{source}:"
                                f"{page_number}:"
                                f"visual"
                            ),
                            text="[VISUAL PAGE]",
                            page=page_number,
                            source=source,
                            blocks=blocks,
                            lines=lines,
                            images=images,
                            tables=[],
                            layout=layout,
                            metadata={
                                "type": "visual",
                                "page_number": page_number,
                                "page": page_number,
                                "source": source,
                                "has_text": False,
                            },
                        )
                    )

        # =====================================================
        # DOCUMENT LEVEL TABLES
        # =====================================================

        document_tables = normalized.get(
            "tables",
            [],
        )

        if isinstance(
            document_tables,
            list,
        ):

            for index, table in enumerate(
                document_tables
            ):

                serialized = self._serialize_table(
                    table
                )

                if not serialized:
                    continue

                page = table.get(
                    "page",
                    table.get(
                        "page_number"
                    ),
                )

                chunks.append(
                    self._make_chunk(
                        chunk_id=(
                            f"{source}:"
                            f"document_table:"
                            f"{index}"
                        ),
                        text=serialized,
                        page=page,
                        source=source,
                        tables=[table],
                        metadata={
                            "type": "table",
                            "page": page,
                            "page_number": page,
                            "source": source,
                            "table_id": table.get(
                                "table_id"
                            ),
                            "row_count": len(
                                table.get(
                                    "rows",
                                    [],
                                )
                            ),
                            "column_count": len(
                                table.get(
                                    "columns",
                                    [],
                                )
                            ),
                        },
                    )
                )

        # =====================================================
        # FALLBACK DOCUMENT TEXT
        # =====================================================

        if not chunks:

            document_text = self._text(
                normalized.get(
                    "text",
                    "",
                )
            )

            normal_text, detected_tables = (
                self._extract_text_tables(
                    document_text
                )
            )

            for index, text_chunk in enumerate(
                self._split_text(
                    normal_text
                )
            ):

                chunks.append(
                    self._make_chunk(
                        chunk_id=(
                            f"{source}:"
                            f"fallback:"
                            f"{index}"
                        ),
                        text=text_chunk,
                        page=None,
                        source=source,
                        images=normalized.get(
                            "images",
                            [],
                        ),
                        tables=normalized.get(
                            "tables",
                            [],
                        ),
                        layout=normalized.get(
                            "layout",
                            [],
                        ),
                        metadata={
                            **(
                                normalized.get(
                                    "metadata",
                                    {},
                                )
                                if isinstance(
                                    normalized.get(
                                        "metadata",
                                        {},
                                    ),
                                    dict,
                                )
                                else {}
                            ),
                            "type": "text",
                            "page": None,
                            "source": source,
                        },
                    )
                )

            for index, table_text in enumerate(
                detected_tables
            ):

                chunks.append(
                    self._make_chunk(
                        chunk_id=(
                            f"{source}:"
                            f"fallback_table:"
                            f"{index}"
                        ),
                        text=(
                            "TABLE\n"
                            + table_text
                        ),
                        page=None,
                        source=source,
                        metadata={
                            "type": "table",
                            "page": None,
                            "source": source,
                            "table_source": "document_text",
                        },
                    )
                )

        return chunks


# Backward-compatible name
HybridDocumentChunker = DocumentChunker