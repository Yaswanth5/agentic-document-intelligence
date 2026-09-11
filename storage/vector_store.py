import re
from typing import Any, Dict, List

import chromadb

from config import CHROMA_DIR


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="documents"
            )
        )

    # =========================================================
    # ADD DOCUMENT CHUNKS
    # =========================================================

    def add_chunks(
        self,
        document_id,
        chunks,
    ):

        if not document_id or not chunks:
            return 0

        document_id = str(
            document_id
        )

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(
            chunks
        ):

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            text = chunk.get(
                "text",
                "",
            )

            if not isinstance(
                text,
                str,
            ):
                text = str(text)

            text = text.strip()

            if not text:
                continue

            original_chunk_id = chunk.get(
                "chunk_id",
                index,
            )

            chunk_id = (
                f"{document_id}_"
                f"{original_chunk_id}"
            )

            metadata = {
                "document_id": document_id,

                "chunk_id": str(
                    original_chunk_id
                ),

                "page": str(
                    chunk.get(
                        "page",
                        "",
                    )
                ),

                "source": str(
                    chunk.get(
                        "source",
                        "unknown",
                    )
                ),

                "type": str(
                    chunk.get(
                        "metadata",
                        {},
                    ).get(
                        "type",
                        "text",
                    )
                    if isinstance(
                        chunk.get(
                            "metadata",
                            {},
                        ),
                        dict,
                    )
                    else "text"
                ),
            }

            chunk_metadata = chunk.get(
                "metadata",
                {},
            )

            if isinstance(
                chunk_metadata,
                dict,
            ):

                for key in [
                    "page_number",
                    "table_id",
                    "table_index",
                    "row_count",
                    "column_count",
                    "table_source",
                ]:

                    value = chunk_metadata.get(
                        key
                    )

                    if value is not None:

                        if isinstance(
                            value,
                            (int, float),
                        ):
                            metadata[key] = value
                        else:
                            metadata[key] = str(
                                value
                            )

            confidence = chunk.get(
                "confidence"
            )

            if confidence is None:

                if isinstance(
                    chunk_metadata,
                    dict,
                ):
                    confidence = chunk_metadata.get(
                        "confidence"
                    )

            if confidence is not None:

                try:
                    metadata["confidence"] = float(
                        confidence
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    metadata["confidence"] = 0.0

            ids.append(
                chunk_id
            )

            documents.append(
                text
            )

            metadatas.append(
                metadata
            )

        if not ids:
            return 0

        # =====================================================
        # REMOVE EXISTING IDS
        # =====================================================

        try:

            existing = self.collection.get(
                ids=ids,
                include=[],
            )

            existing_ids = set(
                existing.get(
                    "ids",
                    [],
                )
            )

        except Exception:

            existing_ids = set()

        new_ids = []
        new_documents = []
        new_metadatas = []

        for index, chunk_id in enumerate(
            ids
        ):

            if chunk_id in existing_ids:
                continue

            new_ids.append(
                chunk_id
            )

            new_documents.append(
                documents[index]
            )

            new_metadatas.append(
                metadatas[index]
            )

        if not new_ids:
            return 0

        self.collection.add(
            ids=new_ids,
            documents=new_documents,
            metadatas=new_metadatas,
        )

        return len(
            new_ids
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query,
        n_results=8,
        document_id=None,
    ):

        if not query:
            return self._empty_result()

        query = str(
            query
        ).strip()

        if not query:
            return self._empty_result()

        if not document_id:

            print(
                "[VECTOR STORE] Search blocked: "
                "document_id was not provided."
            )

            return self._empty_result()

        document_id = str(
            document_id
        )

        count = self.collection.count()

        if count <= 0:
            return self._empty_result()

        try:
            n_results = int(
                n_results
            )
        except (
            TypeError,
            ValueError,
        ):
            n_results = 8

        n_results = max(
            1,
            min(
                n_results,
                20,
            ),
        )

        # =====================================================
        # SEMANTIC SEARCH
        # =====================================================

        semantic_results = self.collection.query(
            query_texts=[
                query
            ],
            n_results=min(
                max(
                    n_results * 3,
                    12,
                ),
                count,
            ),
            where={
                "document_id": document_id
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        # =====================================================
        # BUILD CANDIDATES
        # =====================================================

        candidates = {}

        ids = (
            semantic_results.get(
                "ids",
                [[]],
            )[0]
            if semantic_results.get(
                "ids"
            )
            else []
        )

        documents = (
            semantic_results.get(
                "documents",
                [[]],
            )[0]
            if semantic_results.get(
                "documents"
            )
            else []
        )

        metadatas = (
            semantic_results.get(
                "metadatas",
                [[]],
            )[0]
            if semantic_results.get(
                "metadatas"
            )
            else []
        )

        distances = (
            semantic_results.get(
                "distances",
                [[]],
            )[0]
            if semantic_results.get(
                "distances"
            )
            else []
        )

        for index, chunk_id in enumerate(
            ids
        ):

            candidates[str(
                chunk_id
            )] = {
                "id": str(
                    chunk_id
                ),
                "text": (
                    documents[index]
                    if index < len(
                        documents
                    )
                    else ""
                ),
                "metadata": (
                    metadatas[index]
                    if index < len(
                        metadatas
                    )
                    and isinstance(
                        metadatas[index],
                        dict,
                    )
                    else {}
                ),
                "distance": (
                    distances[index]
                    if index < len(
                        distances
                    )
                    else 999.0
                ),
            }

        # =====================================================
        # LEXICAL RETRIEVAL
        #
        # Fetch chunks belonging only to current document.
        # This catches exact words such as:
        #
        # Region
        # Operational Metrics
        # Units Processed
        # Exception Rate
        #
        # which semantic retrieval can sometimes miss.
        # =====================================================

        try:

            lexical_results = self.collection.get(
                where={
                    "document_id": document_id
                },
                include=[
                    "documents",
                    "metadatas",
                ],
            )

            lexical_ids = lexical_results.get(
                "ids",
                [],
            )

            lexical_documents = lexical_results.get(
                "documents",
                [],
            )

            lexical_metadatas = lexical_results.get(
                "metadatas",
                [],
            )

            query_terms = self._query_terms(
                query
            )

            for index, chunk_id in enumerate(
                lexical_ids
            ):

                text = (
                    lexical_documents[index]
                    if index < len(
                        lexical_documents
                    )
                    else ""
                )

                metadata = (
                    lexical_metadatas[index]
                    if index < len(
                        lexical_metadatas
                    )
                    and isinstance(
                        lexical_metadatas[index],
                        dict,
                    )
                    else {}
                )

                score = self._lexical_score(
                    query,
                    query_terms,
                    text,
                    metadata,
                )

                if score <= 0:
                    continue

                key = str(
                    chunk_id
                )

                if key not in candidates:

                    candidates[key] = {
                        "id": key,
                        "text": text,
                        "metadata": metadata,
                        "distance": 999.0,
                    }

                candidates[key][
                    "lexical_score"
                ] = score

        except Exception as exc:

            print(
                "[VECTOR STORE] "
                f"Lexical retrieval warning: {exc}"
            )

        # =====================================================
        # RANK
        # =====================================================

        ranked = []

        for candidate in candidates.values():

            semantic_score = self._semantic_score(
                candidate.get(
                    "distance",
                    999.0,
                )
            )

            lexical_score = candidate.get(
                "lexical_score",
                0.0,
            )

            metadata = candidate.get(
                "metadata",
                {},
            )

            chunk_type = str(
                metadata.get(
                    "type",
                    "text",
                )
            ).lower()

            # Table chunks receive a small boost because
            # table questions frequently require exact
            # field/value retrieval.
            table_boost = (
                0.35
                if chunk_type == "table"
                else 0.0
            )

            final_score = (
                semantic_score
                + lexical_score
                + table_boost
            )

            ranked.append(
                (
                    final_score,
                    candidate,
                )
            )

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = [
            item[1]
            for item in ranked[
                :n_results
            ]
        ]

        return {
            "ids": [[
                item["id"]
                for item in selected
            ]],

            "documents": [[
                item["text"]
                for item in selected
            ]],

            "metadatas": [[
                item.get(
                    "metadata",
                    {},
                )
                for item in selected
            ]],

            "distances": [[
                item.get(
                    "distance",
                    999.0,
                )
                for item in selected
            ]],
        }

    # =========================================================
    # QUERY TERMS
    # =========================================================

    @staticmethod
    def _query_terms(
        query,
    ):

        stopwords = {
            "what",
            "which",
            "who",
            "where",
            "when",
            "why",
            "how",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "to",
            "from",
            "listed",
            "mentioned",
            "shown",
            "given",
            "document",
            "table",
        }

        words = re.findall(
            r"[a-zA-Z0-9]+",
            str(query).lower(),
        )

        return [
            word
            for word in words
            if len(word) >= 3
            and word not in stopwords
        ]

    # =========================================================
    # LEXICAL SCORE
    # =========================================================

    @staticmethod
    def _lexical_score(
        query,
        query_terms,
        text,
        metadata,
    ):

        if not text:
            return 0.0

        text_lower = str(
            text
        ).lower()

        score = 0.0

        for term in query_terms:

            occurrences = text_lower.count(
                term
            )

            if occurrences:

                score += min(
                    occurrences,
                    5,
                ) * 0.15

        # Exact phrase gets a strong boost.
        query_lower = str(
            query
        ).lower()

        if query_lower in text_lower:
            score += 0.75

        # Table-specific question signals.
        table_words = {
            "table",
            "row",
            "column",
            "listed",
            "regions",
            "region",
            "values",
            "entries",
        }

        if (
            any(
                word in query_lower
                for word in table_words
            )
            and str(
                metadata.get(
                    "type",
                    "",
                )
            ).lower()
            == "table"
        ):
            score += 0.50

        return score

    # =========================================================
    # SEMANTIC SCORE
    # =========================================================

    @staticmethod
    def _semantic_score(
        distance,
    ):

        try:

            distance = float(
                distance
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

        if distance < 0:
            distance = 0

        # Chroma distance is better when smaller.
        return 1.0 / (
            1.0 + distance
        )

    # =========================================================
    # SEARCH ALL
    # =========================================================

    def search_all(
        self,
        query,
        n_results=8,
    ):

        if not query:
            return self._empty_result()

        query = str(
            query
        ).strip()

        if not query:
            return self._empty_result()

        count = self.collection.count()

        if count <= 0:
            return self._empty_result()

        try:
            n_results = int(
                n_results
            )
        except (
            TypeError,
            ValueError,
        ):
            n_results = 8

        n_results = max(
            1,
            min(
                n_results,
                count,
            ),
        )

        return self.collection.query(
            query_texts=[
                query
            ],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    # =========================================================
    # DELETE DOCUMENT
    # =========================================================

    def delete_document(
        self,
        document_id,
    ):

        if not document_id:
            return 0

        document_id = str(
            document_id
        )

        try:

            result = self.collection.get(
                where={
                    "document_id": document_id
                },
                include=[],
            )

            ids = result.get(
                "ids",
                [],
            )

            if not ids:
                return 0

            self.collection.delete(
                ids=ids
            )

            return len(
                ids
            )

        except Exception as exc:

            print(
                f"[VECTOR STORE] Delete failed: {exc}"
            )

            return 0

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self):

        try:

            self.client.delete_collection(
                name="documents"
            )

            self.collection = (
                self.client.get_or_create_collection(
                    name="documents"
                )
            )

            return True

        except Exception as exc:

            print(
                f"[VECTOR STORE] Clear failed: {exc}"
            )

            return False

    # =========================================================
    # EMPTY RESULT
    # =========================================================

    @staticmethod
    def _empty_result():

        return {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
            "distances": [[]],
        }