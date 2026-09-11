import re

from agents.llm import LocalLLM


class ReasoningAgent:
    """
    Generic grounded document reasoning agent.

    Design:
        1. Retrieve document evidence.
        2. Detect structured/table questions.
        3. Attempt deterministic extraction first.
        4. Use LLM reasoning only when deterministic
           extraction cannot confidently answer.
        5. Never use information outside the document.

    Supports:
        - normal text questions
        - key/value extraction
        - table questions
        - row counting
        - record counting
        - unique categorical values
        - column extraction
        - table cell lookup
        - row/entity lookup
        - highest / lowest
        - maximum / minimum
        - sum
        - average
        - page-specific questions
        - document structure questions
        - relationship questions
        - multi-hop questions
        - negative evidence
        - multiple requested values
        - cross-chunk reasoning

    No document-specific field names are hard-coded.
    """

    FALLBACK = (
        "The document does not contain enough information "
        "to answer this question."
    )

    def __init__(self):
        self.llm = LocalLLM()

    # =========================================================
    # MAIN ANSWER
    # =========================================================

    def answer(self, question, contexts):

        if not question or not str(question).strip():
            return "Please provide a question about the document."

        if not contexts:
            return self.FALLBACK

        normalized_contexts = self._normalize_contexts(contexts)

        if not normalized_contexts:
            return self.FALLBACK

        evidence = self._build_evidence(normalized_contexts)

        # -----------------------------------------------------
        # Explicit negative relationship check
        # -----------------------------------------------------

        negative_answer = self._check_explicit_negative_relationship(
            question=question,
            evidence=evidence,
        )

        if negative_answer:
            return negative_answer

        # -----------------------------------------------------
        # Question type
        # -----------------------------------------------------

        question_type = self._detect_question_type(question)

        # -----------------------------------------------------
        # STRUCTURED / TABLE-FIRST REASONING
        # -----------------------------------------------------

        structured_answer = self._answer_structured_question(
            question=question,
            contexts=normalized_contexts,
        )

        if structured_answer:
            return structured_answer

        # -----------------------------------------------------
        # LLM FALLBACK
        # -----------------------------------------------------

        prompt = f"""
You are the reasoning engine of a generic
Agentic Document Intelligence system.

Answer the user's question using ONLY the
DOCUMENT EVIDENCE provided below.

Do not use outside knowledge.
Do not guess.
Do not invent information.

QUESTION TYPE:
{question_type}

USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{evidence}

RULES:

1. Answer exactly what was asked.

2. Use only information explicitly supported
   by the document evidence.

3. Treat tables as structured data.

4. Preserve table header-to-cell relationships.

5. If the question asks for a count, count the
   actual records/rows supported by the evidence.

6. If the question asks for unique values,
   return the distinct values only.

7. If the question asks for highest, lowest,
   maximum, minimum, largest or smallest,
   compare the relevant numeric values carefully.

8. If the question asks for a specific table cell,
   identify both the row and column before answering.

9. Preserve exact names, dates, amounts,
   identifiers, percentages and other values.

10. For key-value information such as:
        Field: Value
        Field = Value
        Field Value
    return the value associated with the requested field.

11. For page-specific questions, use page information
    attached to the evidence.

12. For document structure questions, use only headings,
    sections, tables, pages and structural information.

13. For relationships, preserve direction.
    Do not infer unsupported relationships.

14. For multi-hop reasoning, every hop must be
    supported by the document.

15. Explicit negative information is authoritative.

16. If multiple values are requested, return all
    supported values.

17. Do not use outside knowledge.

18. Do not hallucinate.

19. Do not add unrelated explanation.

20. If the evidence genuinely does not support
    the answer, return exactly:

The document does not contain enough information to answer this question.

FINAL ANSWER:
Return only the concise answer.
"""

        try:

            answer = self.llm.generate(prompt)

            if not answer or not str(answer).strip():
                return self.FALLBACK

            answer = str(answer).strip()

            post_check = self._check_explicit_negative_relationship(
                question=question,
                evidence=evidence,
            )

            if post_check:
                return post_check

            return answer

        except Exception as exc:

            print(
                "[REASONING] Answer generation failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return (
                "Unable to generate an answer "
                "from the retrieved document evidence."
            )

    # =========================================================
    # NORMALIZE CONTEXTS
    # =========================================================

    @staticmethod
    def _normalize_contexts(contexts):

        normalized_contexts = []

        for index, context in enumerate(contexts, start=1):

            if not isinstance(context, dict):
                continue

            text = str(
                context.get("text", "")
            ).strip()

            if not text:
                continue

            metadata = context.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            page = metadata.get(
                "page",
                context.get(
                    "page",
                    metadata.get(
                        "page_number",
                        "unknown",
                    ),
                ),
            )

            source = metadata.get(
                "source",
                context.get(
                    "source",
                    "unknown",
                ),
            )

            chunk_id = context.get(
                "chunk_id",
                metadata.get(
                    "chunk_id",
                    index,
                ),
            )

            chunk_type = metadata.get(
                "type",
                context.get(
                    "type",
                    "text",
                ),
            )

            normalized_contexts.append(
                {
                    "index": index,
                    "text": text,
                    "metadata": metadata,
                    "page": page,
                    "source": source,
                    "chunk_id": chunk_id,
                    "type": chunk_type,
                }
            )

        return normalized_contexts

    # =========================================================
    # BUILD EVIDENCE
    # =========================================================

    @staticmethod
    def _build_evidence(contexts):

        parts = []

        for context in contexts:

            parts.append(
                f"""
=========================================================
EVIDENCE {context["index"]}
=========================================================

Chunk ID: {context["chunk_id"]}
Page: {context["page"]}
Source: {context["source"]}
Type: {context["type"]}

DOCUMENT CONTENT:
{context["text"]}
"""
            )

        return "\n".join(parts)

    # =========================================================
    # QUESTION TYPE
    # =========================================================

    @staticmethod
    def _detect_question_type(question):

        q = str(question).lower()

        table_terms = [
            "table",
            "row",
            "rows",
            "column",
            "columns",
            "listed",
            "entries",
            "entry",
            "record",
            "records",
            "transaction",
            "transactions",
            "item",
            "items",
            "values in",
            "under the",
            "highest",
            "lowest",
            "maximum",
            "minimum",
            "largest",
            "smallest",
            "most",
            "least",
            "sum",
            "total",
            "average",
            "mean",
            "count",
            "how many",
        ]

        if any(term in q for term in table_terms):
            return "TABLE / STRUCTURED DATA QUESTION"

        if any(
            term in q
            for term in [
                "page",
                "on page",
                "from page",
            ]
        ):
            return "PAGE-SPECIFIC QUESTION"

        if any(
            term in q
            for term in [
                "section",
                "heading",
                "title",
                "structure",
            ]
        ):
            return "DOCUMENT STRUCTURE QUESTION"

        if any(
            term in q
            for term in [
                "father",
                "mother",
                "brother",
                "sister",
                "husband",
                "wife",
                "son",
                "daughter",
                "parent",
                "child",
            ]
        ):
            return "RELATIONSHIP QUESTION"

        return "GENERAL DOCUMENT QUESTION"

    # =========================================================
    # STRUCTURED QUESTION ROUTER
    # =========================================================

    def _answer_structured_question(
        self,
        question,
        contexts,
    ):

        question_lower = str(question).lower().strip()

        # -----------------------------------------------------
        # Extract tables
        # -----------------------------------------------------

        tables = []

        for context in contexts:

            text = context.get("text", "")

            chunk_type = str(
                context.get("type", "")
            ).lower()

            if (
                chunk_type == "table"
                or self._contains_markdown_table(text)
            ):

                parsed = self._parse_markdown_table(text)

                if parsed:
                    tables.append(parsed)

        # -----------------------------------------------------
        # TABLE QUESTIONS
        # -----------------------------------------------------

        if tables:

            # Count
            if self._is_count_question(question_lower):

                answer = self._answer_count_question(
                    question_lower,
                    tables,
                )

                if answer:
                    return answer

            # Sum
            if self._is_sum_question(question_lower):

                answer = self._answer_sum_question(
                    question_lower,
                    tables,
                )

                if answer:
                    return answer

            # Average / mean
            if self._is_average_question(question_lower):

                answer = self._answer_average_question(
                    question_lower,
                    tables,
                )

                if answer:
                    return answer

            # Comparison
            if self._is_comparative_question(question_lower):

                for headers, rows in tables:

                    result = self._answer_comparison_question(
                        question=question_lower,
                        headers=headers,
                        rows=rows,
                    )

                    if result:
                        return result

            # Specific table cell / row lookup
            cell_answer = self._answer_table_lookup_question(
                question_lower,
                tables,
            )

            if cell_answer:
                return cell_answer

            # Column / unique-value question
            column_answer = self._answer_column_question(
                question_lower,
                tables,
            )

            if column_answer:
                return column_answer

        # -----------------------------------------------------
        # KEY-VALUE / LABEL-VALUE QUESTIONS
        # -----------------------------------------------------

        key_value_answer = self._answer_key_value_question(
            question_lower,
            contexts,
        )

        if key_value_answer:
            return key_value_answer

        return None

    # =========================================================
    # COUNT QUESTION
    # =========================================================

    @staticmethod
    def _is_count_question(question):

        q = str(question).lower()

        count_patterns = [
            r"\bhow many\b",
            r"\bnumber of\b",
            r"\bcount of\b",
            r"\bcount\b",
            r"\btotal number\b",
        ]

        return any(
            re.search(pattern, q)
            for pattern in count_patterns
        )

    # =========================================================
    # ANSWER COUNT
    # =========================================================

    def _answer_count_question(
        self,
        question,
        tables,
    ):

        # -----------------------------------------------------
        # Count rows.
        #
        # For questions such as:
        #   How many records are present?
        #   How many rows are there?
        #   How many entries?
        # -----------------------------------------------------

        for headers, rows in tables:

            if not rows:
                continue

            # If a specific column is mentioned and the question
            # asks how many values exist, count non-empty cells.
            column_index = self._find_requested_column(
                question,
                headers,
            )

            # Avoid interpreting generic words such as "record"
            # as an actual table column.
            if (
                column_index is not None
                and self._question_explicitly_mentions_header(
                    question,
                    headers[column_index],
                )
            ):

                count = sum(
                    1
                    for row in rows
                    if (
                        column_index < len(row)
                        and str(row[column_index]).strip()
                    )
                )

                return str(count)

            # Otherwise count actual data rows.
            return str(len(rows))

        return None

    # =========================================================
    # SUM QUESTION
    # =========================================================

    @staticmethod
    def _is_sum_question(question):

        q = str(question).lower()

        return any(
            phrase in q
            for phrase in [
                "sum of",
                "sum ",
                "total ",
                "total amount",
                "total value",
                "grand total",
            ]
        )

    def _answer_sum_question(
        self,
        question,
        tables,
    ):

        for headers, rows in tables:

            column_index = self._find_requested_column(
                question,
                headers,
            )

            if column_index is None:
                continue

            values = []

            for row in rows:

                if column_index >= len(row):
                    continue

                numeric = self._parse_numeric_value(
                    row[column_index]
                )

                if numeric is not None:
                    values.append(numeric)

            if not values:
                continue

            total = sum(values)

            return self._format_numeric_answer(
                total
            )

        return None

    # =========================================================
    # AVERAGE QUESTION
    # =========================================================

    @staticmethod
    def _is_average_question(question):

        q = str(question).lower()

        return any(
            phrase in q
            for phrase in [
                "average",
                "mean",
                "avg",
            ]
        )

    def _answer_average_question(
        self,
        question,
        tables,
    ):

        for headers, rows in tables:

            column_index = self._find_requested_column(
                question,
                headers,
            )

            if column_index is None:
                continue

            values = []

            for row in rows:

                if column_index >= len(row):
                    continue

                numeric = self._parse_numeric_value(
                    row[column_index]
                )

                if numeric is not None:
                    values.append(numeric)

            if not values:
                continue

            average = sum(values) / len(values)

            return self._format_numeric_answer(
                average
            )

        return None

    # =========================================================
    # COMPARATIVE QUESTION
    # =========================================================

    @staticmethod
    def _is_comparative_question(question):

        comparison_terms = [
            "highest",
            "lowest",
            "maximum",
            "minimum",
            "largest",
            "smallest",
            "most",
            "least",
            "max",
            "min",
        ]

        q = str(question).lower()

        return any(
            re.search(
                rf"\b{re.escape(term)}\b",
                q,
            )
            for term in comparison_terms
        )

    # =========================================================
    # ANSWER COMPARISON
    # =========================================================

    def _answer_comparison_question(
        self,
        question,
        headers,
        rows,
    ):

        metric_index = self._find_requested_column(
            question,
            headers,
        )

        if metric_index is None:
            return None

        highest = any(
            re.search(
                rf"\b{term}\b",
                question,
            )
            for term in [
                "highest",
                "maximum",
                "largest",
                "most",
                "max",
            ]
        )

        lowest = any(
            re.search(
                rf"\b{term}\b",
                question,
            )
            for term in [
                "lowest",
                "minimum",
                "smallest",
                "least",
                "min",
            ]
        )

        if not highest and not lowest:
            return None

        entity_index = self._find_entity_column(
            question,
            headers,
            metric_index,
        )

        if entity_index is None:

            for index in range(len(headers)):

                if index != metric_index:
                    entity_index = index
                    break

        if entity_index is None:
            return None

        candidates = []

        for row in rows:

            if (
                metric_index >= len(row)
                or entity_index >= len(row)
            ):
                continue

            entity = str(
                row[entity_index]
            ).strip()

            metric = str(
                row[metric_index]
            ).strip()

            if not entity:
                continue

            numeric_value = self._parse_numeric_value(
                metric
            )

            if numeric_value is None:
                continue

            candidates.append(
                (
                    numeric_value,
                    entity,
                    metric,
                )
            )

        if not candidates:
            return None

        if highest:
            best = max(
                candidates,
                key=lambda x: x[0],
            )
        else:
            best = min(
                candidates,
                key=lambda x: x[0],
            )

        return best[1]

    # =========================================================
    # TABLE LOOKUP
    # =========================================================

    def _answer_table_lookup_question(
        self,
        question,
        tables,
    ):
        """
        Handles questions such as:

            What is the service level in Q3?

            What is Q3 service level?

            What is the amount for REC-1020?

            What is the status of REC-1005?

        """

        for headers, rows in tables:

            if not headers or not rows:
                continue

            # -------------------------------------------------
            # Find column(s) explicitly referenced by question.
            # -------------------------------------------------

            column_candidates = self._find_column_candidates(
                question,
                headers,
            )

            if not column_candidates:
                continue

            # -------------------------------------------------
            # Find row/entity value referenced by question.
            # -------------------------------------------------

            row_match = self._find_row_by_question(
                question,
                headers,
                rows,
                column_candidates,
            )

            if row_match:

                row_index, target_column = row_match

                row = rows[row_index]

                if target_column < len(row):

                    value = str(
                        row[target_column]
                    ).strip()

                    if value:
                        return value

            # -------------------------------------------------
            # Handle questions such as:
            #
            # "What is Q3?"
            #
            # where Q3 itself is a column.
            # -------------------------------------------------

            direct_column = self._find_direct_column_reference(
                question,
                headers,
            )

            if direct_column is not None:

                # If there is exactly one data row,
                # return its value.
                if len(rows) == 1:

                    value = rows[0][direct_column].strip()

                    if value:
                        return value

        return None

    # =========================================================
    # FIND COLUMN CANDIDATES
    # =========================================================

    @staticmethod
    def _find_column_candidates(
        question,
        headers,
    ):

        q = str(question).lower()

        candidates = []

        for index, header in enumerate(headers):

            header_normalized = ReasoningAgent._normalize_text(
                header
            )

            if not header_normalized:
                continue

            header_tokens = header_normalized.split()

            # Exact phrase.
            if header_normalized in q:
                candidates.append(
                    (
                        100 + len(header_tokens),
                        index,
                    )
                )
                continue

            q_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    q,
                )
            )

            overlap = 0

            for token in header_tokens:

                for q_token in q_tokens:

                    if ReasoningAgent._same_word(
                        token,
                        q_token,
                    ):
                        overlap += 1

            if overlap:

                candidates.append(
                    (
                        overlap,
                        index,
                    )
                )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            index
            for _, index in candidates
        ]

    # =========================================================
    # FIND ROW BY QUESTION
    # =========================================================

    def _find_row_by_question(
        self,
        question,
        headers,
        rows,
        column_candidates,
    ):

        q = self._normalize_text(question)

        # -----------------------------------------------------
        # First look for exact values from the table.
        # This is extremely useful for IDs such as:
        #
        # REC-1020
        # SYN-2026-0042
        # -----------------------------------------------------

        for row_index, row in enumerate(rows):

            for cell in row:

                cell_text = str(cell).strip()

                if not cell_text:
                    continue

                normalized_cell = self._normalize_text(
                    cell_text
                )

                if not normalized_cell:
                    continue

                # IDs / codes / exact multi-character values.
                if (
                    len(normalized_cell) >= 3
                    and normalized_cell in q
                ):

                    target_column = (
                        column_candidates[0]
                        if column_candidates
                        else 0
                    )

                    return (
                        row_index,
                        target_column,
                    )

        # -----------------------------------------------------
        # Look for ordinary exact entity values.
        # -----------------------------------------------------

        for row_index, row in enumerate(rows):

            for cell_index, cell in enumerate(row):

                cell_text = str(cell).strip()

                if not cell_text:
                    continue

                normalized_cell = self._normalize_text(
                    cell_text
                )

                if len(normalized_cell) < 3:
                    continue

                if (
                    re.search(
                        rf"\b{re.escape(normalized_cell)}\b",
                        q,
                    )
                ):

                    target_column = (
                        column_candidates[0]
                        if column_candidates
                        else 0
                    )

                    # If the matching cell is itself the
                    # requested column, use it.
                    if cell_index in column_candidates:
                        target_column = cell_index

                    return (
                        row_index,
                        target_column,
                    )

        return None

    # =========================================================
    # DIRECT COLUMN REFERENCE
    # =========================================================

    @staticmethod
    def _find_direct_column_reference(
        question,
        headers,
    ):

        q = ReasoningAgent._normalize_text(
            question
        )

        for index, header in enumerate(headers):

            normalized_header = (
                ReasoningAgent._normalize_text(
                    header
                )
            )

            if not normalized_header:
                continue

            if normalized_header in q:
                return index

        return None

    # =========================================================
    # COLUMN QUESTION
    # =========================================================

    def _answer_column_question(
        self,
        question,
        tables,
    ):

        for headers, rows in tables:

            if not headers or not rows:
                continue

            column_index = self._find_requested_column(
                question,
                headers,
            )

            if column_index is None:

                column_index = self._infer_list_column(
                    question,
                    headers,
                )

            if column_index is None:
                continue

            # Don't return a column merely because the question
            # contains generic words such as "record" or "value".
            if not self._question_explicitly_mentions_header(
                question,
                headers[column_index],
            ):

                # For questions like:
                # "What statuses are present?"
                # the header is explicitly "Status".
                #
                # If no explicit header is found, only infer
                # when the question strongly implies a list.
                if not self._is_list_question(question):
                    continue

            values = []

            for row in rows:

                if column_index >= len(row):
                    continue

                value = str(
                    row[column_index]
                ).strip()

                if not value:
                    continue

                values.append(value)

            if not values:
                continue

            unique_values = []

            seen_normalized = set()

            for value in values:

                normalized = self._normalize_text(
                    value
                )

                if normalized in seen_normalized:
                    continue

                seen_normalized.add(normalized)
                unique_values.append(value)

            return ", ".join(
                unique_values
            )

        return None

    # =========================================================
    # LIST QUESTION
    # =========================================================

    @staticmethod
    def _is_list_question(question):

        q = str(question).lower()

        return any(
            phrase in q
            for phrase in [
                "what ",
                "which ",
                "list",
                "listed",
                "present",
                "available",
                "shown",
                "included",
                "values",
            ]
        )

    # =========================================================
    # KEY/VALUE QUESTION
    # =========================================================

    def _answer_key_value_question(
        self,
        question,
        contexts,
    ):
        """
        Generic extraction from common document formats:

            Reference: SYN-2026-0042
            Status: Pending Review
            Total Amount: 48,750.00

        Also handles:

            Reference = SYN-2026-0042
            Status - Pending Review
        """

        q = self._normalize_text(
            question
        )

        if not q:
            return None

        # -----------------------------------------------------
        # Extract candidate requested terms from question.
        # -----------------------------------------------------

        requested_terms = self._extract_requested_field_terms(
            q
        )

        if not requested_terms:
            return None

        # -----------------------------------------------------
        # Search every evidence chunk.
        # -----------------------------------------------------

        for context in contexts:

            text = str(
                context.get("text", "")
            )

            if not text:
                continue

            # -------------------------------------------------
            # First: line-based key/value extraction.
            # -------------------------------------------------

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            for line in lines:

                match = re.match(
                    r"^\s*([^:|=–—-]{2,80})\s*"
                    r"[:=–—-]\s*(.+?)\s*$",
                    line,
                )

                if not match:
                    continue

                key = self._normalize_text(
                    match.group(1)
                )

                value = match.group(2).strip()

                if not key or not value:
                    continue

                if self._field_matches(
                    key,
                    requested_terms,
                ):

                    return self._clean_answer_value(
                        value
                    )

            # -------------------------------------------------
            # Inline key-value extraction.
            #
            # Useful when OCR/extraction produces:
            #
            # DOCUMENTINTELLIGENCE TEST
            # Reference: SYN-2026-0042 Total Amount ...
            # -------------------------------------------------

            normalized_text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            for term in requested_terms:

                pattern = (
                    rf"\b{re.escape(term)}\b"
                    rf"\s*[:=]\s*"
                    rf"(.+?)"
                    rf"(?=\s+[A-Za-z][A-Za-z0-9 _/&()-]{{1,60}}"
                    rf"\s*[:=]|\s*$)"
                )

                match = re.search(
                    pattern,
                    normalized_text,
                    flags=re.IGNORECASE,
                )

                if match:

                    value = match.group(1).strip()

                    if value:
                        return self._clean_answer_value(
                            value
                        )

        return None

    # =========================================================
    # EXTRACT REQUESTED FIELD TERMS
    # =========================================================

    @staticmethod
    def _extract_requested_field_terms(
        question
    ):

        q = str(
            question
        ).lower().strip()

        # -----------------------------------------------------
        # Remove question phrases.
        # -----------------------------------------------------

        cleaned = q

        remove_patterns = [
            r"\bwhat is\b",
            r"\bwhat was\b",
            r"\bwhat are\b",
            r"\bwhat were\b",
            r"\bwhich is\b",
            r"\bwhich was\b",
            r"\bcan you tell me\b",
            r"\bplease tell me\b",
            r"\btell me\b",
            r"\bshow me\b",
            r"\bfind\b",
            r"\bgive me\b",
            r"\bthe\b",
            r"\bthis\b",
            r"\bthat\b",
            r"\bdocument\b",
            r"\bmentioned\b",
            r"\bprovided\b",
        ]

        for pattern in remove_patterns:

            cleaned = re.sub(
                pattern,
                " ",
                cleaned,
            )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        if not cleaned:
            return []

        # -----------------------------------------------------
        # Generate useful candidate phrases.
        # -----------------------------------------------------

        candidates = []

        candidates.append(cleaned)

        tokens = cleaned.split()

        if len(tokens) > 1:

            # Remove common grammatical words.
            stopwords = {
                "is",
                "was",
                "are",
                "were",
                "of",
                "in",
                "on",
                "for",
                "from",
                "with",
                "to",
                "and",
                "or",
                "present",
                "listed",
                "shown",
                "given",
                "available",
            }

            filtered = [
                token
                for token in tokens
                if token not in stopwords
            ]

            if filtered:
                candidates.append(
                    " ".join(filtered)
                )

        # -----------------------------------------------------
        # Preserve only meaningful terms.
        # -----------------------------------------------------

        result = []

        for candidate in candidates:

            candidate = candidate.strip()

            if not candidate:
                continue

            if candidate not in result:
                result.append(candidate)

        return result

    # =========================================================
    # FIELD MATCHING
    # =========================================================

    @staticmethod
    def _field_matches(
        key,
        requested_terms,
    ):

        normalized_key = ReasoningAgent._normalize_text(
            key
        )

        for term in requested_terms:

            normalized_term = (
                ReasoningAgent._normalize_text(
                    term
                )
            )

            if not normalized_term:
                continue

            if (
                normalized_key == normalized_term
                or normalized_key in normalized_term
                or normalized_term in normalized_key
            ):
                return True

            key_tokens = set(
                normalized_key.split()
            )

            term_tokens = set(
                normalized_term.split()
            )

            if key_tokens and term_tokens:

                overlap = (
                    key_tokens & term_tokens
                )

                if len(overlap) == len(key_tokens):
                    return True

        return False

    # =========================================================
    # CLEAN ANSWER VALUE
    # =========================================================

    @staticmethod
    def _clean_answer_value(
        value
    ):

        value = str(
            value
        ).strip()

        value = value.strip(
            " \t\r\n.,;"
        )

        return value

    # =========================================================
    # REQUESTED COLUMN
    # =========================================================

    @staticmethod
    def _find_requested_column(
        question,
        headers,
    ):

        q = ReasoningAgent._normalize_text(
            question
        )

        best_index = None
        best_score = 0

        q_tokens = set(
            q.split()
        )

        for index, header in enumerate(headers):

            normalized_header = (
                ReasoningAgent._normalize_text(
                    header
                )
            )

            if not normalized_header:
                continue

            header_tokens = normalized_header.split()

            # -------------------------------------------------
            # Exact phrase
            # -------------------------------------------------

            if normalized_header in q:

                score = 100 + len(
                    header_tokens
                )

                if score > best_score:

                    best_score = score
                    best_index = index

                continue

            # -------------------------------------------------
            # Token overlap
            # -------------------------------------------------

            overlap = 0

            for header_token in header_tokens:

                for question_token in q_tokens:

                    if ReasoningAgent._same_word(
                        header_token,
                        question_token,
                    ):

                        overlap += 1

            if overlap:

                # Longer/more specific headers get preference.
                score = overlap * 5 + len(
                    header_tokens
                )

                if score > best_score:

                    best_score = score
                    best_index = index

        return best_index

    # =========================================================
    # INFER LIST COLUMN
    # =========================================================

    @staticmethod
    def _infer_list_column(
        question,
        headers,
    ):

        q = ReasoningAgent._normalize_text(
            question
        )

        q_tokens = set(
            q.split()
        )

        candidates = []

        for index, header in enumerate(headers):

            header_tokens = set(
                ReasoningAgent._normalize_text(
                    header
                ).split()
            )

            overlap = 0

            for header_token in header_tokens:

                for question_token in q_tokens:

                    if ReasoningAgent._same_word(
                        header_token,
                        question_token,
                    ):

                        overlap += 1

            if overlap:

                candidates.append(
                    (
                        overlap,
                        len(header_tokens),
                        index,
                    )
                )

        if candidates:

            candidates.sort(
                reverse=True
            )

            return candidates[0][2]

        return None

    # =========================================================
    # EXPLICIT HEADER MENTION
    # =========================================================

    @staticmethod
    def _question_explicitly_mentions_header(
        question,
        header,
    ):

        q = ReasoningAgent._normalize_text(
            question
        )

        h = ReasoningAgent._normalize_text(
            header
        )

        if not q or not h:
            return False

        if h in q:
            return True

        h_tokens = h.split()
        q_tokens = set(
            q.split()
        )

        matches = 0

        for token in h_tokens:

            if any(
                ReasoningAgent._same_word(
                    token,
                    q_token,
                )
                for q_token in q_tokens
            ):
                matches += 1

        return (
            matches == len(h_tokens)
        )

    # =========================================================
    # FIND ENTITY COLUMN
    # =========================================================

    @staticmethod
    def _find_entity_column(
        question,
        headers,
        metric_index,
    ):

        q = ReasoningAgent._normalize_text(
            question
        )

        best_index = None
        best_score = 0

        for index, header in enumerate(headers):

            if index == metric_index:
                continue

            normalized_header = (
                ReasoningAgent._normalize_text(
                    header
                )
            )

            if not normalized_header:
                continue

            header_tokens = set(
                normalized_header.split()
            )

            question_tokens = set(
                q.split()
            )

            overlap = 0

            for header_token in header_tokens:

                for question_token in question_tokens:

                    if ReasoningAgent._same_word(
                        header_token,
                        question_token,
                    ):

                        overlap += 1

            if overlap:

                score = overlap

                if score > best_score:

                    best_score = score
                    best_index = index

        return best_index

    # =========================================================
    # NUMERIC PARSER
    # =========================================================

    @staticmethod
    def _parse_numeric_value(
        value,
    ):

        if value is None:
            return None

        text = str(
            value
        ).strip()

        if not text:
            return None

        text = text.replace(
            ",",
            "",
        )

        for symbol in [
            "$",
            "₹",
            "€",
            "£",
        ]:

            text = text.replace(
                symbol,
                "",
            )

        match = re.search(
            r"-?\d+(?:\.\d+)?",
            text,
        )

        if not match:
            return None

        try:

            return float(
                match.group(0)
            )

        except ValueError:

            return None

    # =========================================================
    # FORMAT NUMERIC ANSWER
    # =========================================================

    @staticmethod
    def _format_numeric_answer(
        value
    ):

        if value is None:
            return None

        if float(value).is_integer():

            return f"{int(value):,}"

        return f"{value:,.2f}".rstrip(
            "0"
        ).rstrip(
            "."
        )

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        value
    ):

        text = str(
            value or ""
        ).lower()

        text = re.sub(
            r"[^a-z0-9\s.%/-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    # =========================================================
    # WORD MATCHING
    # =========================================================

    @staticmethod
    def _same_word(
        left,
        right,
    ):

        left = str(
            left
        ).lower().strip()

        right = str(
            right
        ).lower().strip()

        if left == right:
            return True

        # Simple singular/plural handling.
        if left.rstrip("s") == right.rstrip("s"):
            return True

        if left.endswith("ies") and right.endswith("y"):

            if left[:-3] + "y" == right:
                return True

        if right.endswith("ies") and left.endswith("y"):

            if right[:-3] + "y" == left:
                return True

        return False

    # =========================================================
    # MARKDOWN TABLE DETECTION
    # =========================================================

    @staticmethod
    def _contains_markdown_table(
        text,
    ):

        if not text:
            return False

        lines = [
            line.strip()
            for line in str(text).splitlines()
            if line.strip()
        ]

        pipe_lines = [
            line
            for line in lines
            if "|" in line
        ]

        return len(pipe_lines) >= 2

    # =========================================================
    # MARKDOWN TABLE PARSER
    # =========================================================

    @staticmethod
    def _parse_markdown_table(
        text,
    ):

        lines = [
            line.strip()
            for line in str(text).splitlines()
            if line.strip()
        ]

        table_lines = []

        for line in lines:

            if "|" not in line:
                continue

            if line.upper().startswith("TABLE"):

                line = line[5:].strip()

            if "|" in line:
                table_lines.append(line)

        if len(table_lines) < 2:
            return None

        # -----------------------------------------------------
        # Find header/separator.
        # -----------------------------------------------------

        header_line = table_lines[0]

        separator_index = None

        for index, line in enumerate(
            table_lines[1:],
            start=1,
        ):

            cells = [
                cell.strip()
                for cell in line.strip("|").split("|")
            ]

            if not cells:
                continue

            is_separator = all(
                re.fullmatch(
                    r":?-+:?",
                    cell.strip(),
                )
                is not None
                for cell in cells
                if cell.strip()
            )

            if is_separator:

                separator_index = index
                break

        if separator_index is None:

            # Some extraction systems may omit the
            # markdown separator. Attempt a simple
            # first-row-as-header fallback.

            if len(table_lines) >= 2:

                headers = [
                    ReasoningAgent._clean_table_cell(
                        cell
                    )
                    for cell in table_lines[0]
                    .strip("|")
                    .split("|")
                ]

                rows = []

                for line in table_lines[1:]:

                    cells = [
                        ReasoningAgent._clean_table_cell(
                            cell
                        )
                        for cell in line
                        .strip("|")
                        .split("|")
                    ]

                    if len(cells) == len(headers):
                        rows.append(cells)

                if rows:
                    return headers, rows

            return None

        headers = [
            ReasoningAgent._clean_table_cell(
                cell
            )
            for cell in header_line
            .strip("|")
            .split("|")
        ]

        headers = [
            header
            for header in headers
            if header != ""
        ]

        if not headers:
            return None

        rows = []

        for line in table_lines[
            separator_index + 1:
        ]:

            cells = [
                ReasoningAgent._clean_table_cell(
                    cell
                )
                for cell in line
                .strip("|")
                .split("|")
            ]

            if not cells:
                continue

            # Ignore repeated separator/header artifacts.
            if all(
                re.fullmatch(
                    r":?-+:?",
                    cell,
                )
                for cell in cells
                if cell
            ):
                continue

            if len(cells) < len(headers):

                cells.extend(
                    [""] * (
                        len(headers) - len(cells)
                    )
                )

            elif len(cells) > len(headers):

                cells = cells[
                    :len(headers)
                ]

            rows.append(cells)

        if not rows:
            return None

        return headers, rows

    # =========================================================
    # CLEAN TABLE CELL
    # =========================================================

    @staticmethod
    def _clean_table_cell(
        value,
    ):

        value = str(
            value
        ).strip()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    # =========================================================
    # NEGATIVE RELATIONSHIP CHECK
    # =========================================================

    def _check_explicit_negative_relationship(
        self,
        question,
        evidence,
    ):

        question_lower = str(
            question or ""
        ).lower().strip()

        evidence_lower = str(
            evidence or ""
        ).lower()

        category = (
            self._detect_relationship_category(
                question_lower
            )
        )

        if not category:
            return None

        patterns = (
            self._get_negative_patterns(
                category
            )
        )

        for pattern in patterns:

            if pattern in evidence_lower:

                return (
                    self._negative_relationship_response(
                        category
                    )
                )

        return None

    # =========================================================
    # RELATIONSHIP CATEGORY
    # =========================================================

    @staticmethod
    def _detect_relationship_category(
        question,
    ):

        relationship_groups = {

            "sibling": [
                "brother",
                "brothers",
                "sister",
                "sisters",
                "sibling",
                "siblings",
            ],

            "parent": [
                "father",
                "mother",
                "parent",
                "parents",
                "dad",
                "mom",
            ],

            "spouse": [
                "husband",
                "wife",
                "spouse",
            ],

            "child": [
                "son",
                "daughter",
                "child",
                "children",
            ],
        }

        padded = f" {question} "

        for category, terms in (
            relationship_groups.items()
        ):

            for term in terms:

                if f" {term} " in padded:
                    return category

        return None

    # =========================================================
    # NEGATIVE PATTERNS
    # =========================================================

    @staticmethod
    def _get_negative_patterns(
        category,
    ):

        if category == "sibling":

            return [
                "brothers & sisters = no",
                "brothers and sisters = no",
                "brothers & sisters: no",
                "brothers and sisters: no",
                "brothers & sisters no",
                "brothers and sisters no",
                "siblings = no",
                "siblings: no",
                "siblings no",
                "brother = no",
                "brother: no",
                "sister = no",
                "sister: no",
                "no brothers",
                "no sisters",
                "no siblings",
                "brothers none",
                "sisters none",
                "siblings none",
                "brothers: none",
                "sisters: none",
                "siblings: none",
                "brothers & sisters = none",
                "brothers and sisters = none",
                "brothers & sisters = nil",
                "brothers and sisters = nil",
            ]

        if category == "child":

            return [
                "children = no",
                "children: no",
                "children no",
                "child = no",
                "child: no",
                "children = none",
                "children: none",
                "children none",
                "no children",
                "no child",
                "children = nil",
                "child = nil",
            ]

        if category == "parent":

            return [
                "parents = no",
                "parents: no",
                "parents no",
                "parent = no",
                "parent: no",
                "parents = none",
                "parents: none",
                "parents none",
            ]

        if category == "spouse":

            return [
                "spouse = no",
                "spouse: no",
                "spouse no",
                "husband = no",
                "husband: no",
                "wife = no",
                "wife: no",
                "spouse = none",
                "spouse: none",
                "husband = none",
                "wife = none",
                "no spouse",
                "no husband",
                "no wife",
            ]

        return []

    # =========================================================
    # NEGATIVE RESPONSE
    # =========================================================

    @staticmethod
    def _negative_relationship_response(
        category,
    ):

        responses = {

            "sibling": (
                "The document explicitly indicates "
                "that there are no brothers or sisters."
            ),

            "child": (
                "The document explicitly indicates "
                "that there are no children."
            ),

            "parent": (
                "The document explicitly indicates "
                "that no parent is provided."
            ),

            "spouse": (
                "The document explicitly indicates "
                "that no spouse is provided."
            ),
        }

        return responses.get(
            category,
            (
                "The document explicitly indicates "
                "that the requested relationship is absent."
            ),
        )