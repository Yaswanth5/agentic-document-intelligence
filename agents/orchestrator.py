import time
import uuid

from processing.document_loader import (
    DocumentLoader
)

from processing.chunker import (
    DocumentChunker
)

from agents.classifier import (
    DocumentClassifier
)

from agents.extractor import (
    GenericExtractor
)

from agents.merger import (
    ResultMerger
)

from agents.validator import (
    ResultValidator
)

from storage.vector_store import (
    VectorStore
)


class DocumentOrchestrator:

    def __init__(self):

        self.loader = (
            DocumentLoader()
        )

        self.classifier = (
            DocumentClassifier()
        )

        self.chunker = (
            DocumentChunker()
        )

        self.extractor = (
            GenericExtractor()
        )

        self.merger = (
            ResultMerger()
        )

        self.validator = (
            ResultValidator()
        )

        self.vector_store = (
            VectorStore()
        )

    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        file_path: str
    ):

        total_start = time.time()

        document_id = str(
            uuid.uuid4()
        )

        print()
        print(
            "=" * 70
        )

        print(
            "DOCUMENT PROCESSING START"
        )

        print(
            f"Document ID: {document_id}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        start = time.time()

        document = self.loader.load(
            file_path
        )

        load_time = (
            time.time() - start
        )

        print(
            f"[TIMING] "
            f"Document loading/OCR: "
            f"{load_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # CLASSIFY
        # ----------------------------------------------------

        start = time.time()

        classification = (
            self.classifier.classify(
                document
            )
        )

        classification_time = (
            time.time() - start
        )

        print(
            f"[TIMING] "
            f"Classification: "
            f"{classification_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # CHUNK
        # ----------------------------------------------------

        start = time.time()

        chunks = self.chunker.chunk(
            document
        )

        chunk_time = (
            time.time() - start
        )

        print(
            f"[TIMING] "
            f"Chunking: "
            f"{chunk_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # VECTOR STORE
        # ----------------------------------------------------

        start = time.time()

        storage_result = (
            self.vector_store.add_chunks(
                document_id,
                chunks
            )
        )

        vector_time = (
            time.time() - start
        )

        print(
            f"[TIMING] "
            f"Vector store: "
            f"{vector_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # EXTRACTION
        # ----------------------------------------------------

        start = time.time()

        extracted = (
            self.extractor.extract(
                chunks,
                document
            )
        )

        extraction_time = (
            time.time() - start
        )

        print(
            f"[TIMING] "
            f"Semnatic extraction: "
            f"{extraction_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        start = time.time()

        merged = (
            self.merger.merge(
                extracted
            )
            if hasattr(
                self.merger,
                "merge"
            )
            else extracted
        )

        merge_time = (
            time.time() - start
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        start = time.time()

        validation = (
            self.validator.validate_total(
                merged
            )
            if hasattr(
                self.validator,
                "validate_total"
            )
            else {}
        )

        validation_time = (
            time.time() - start
        )

        # ----------------------------------------------------
        # STRUCTURAL SUMMARY
        # ----------------------------------------------------

        structural_summary = {

            "page_count": document.get(
                "page_count",
                len(
                    document.get(
                        "pages",
                        []
                    )
                )
            ),

            "total_text_characters": len(
                document.get(
                    "text",
                    ""
                )
            ),

            "image_count": len(
                document.get(
                    "images",
                    []
                )
            ),

            "table_count": len(
                document.get(
                    "tables",
                    []
                )
            ),

            "layout_object_count": len(
                document.get(
                    "layout",
                    []
                )
            ),

            "pages_with_native_text": sum(
                bool(
                    p.get(
                        "native_text",
                        ""
                    )
                )
                for p in document.get(
                    "pages",
                    []
                )
            ),

            "pages_with_ocr": sum(
                bool(
                    p.get(
                        "ocr_text",
                        ""
                    )
                )
                for p in document.get(
                    "pages",
                    []
                )
            ),

            "tables": [],

        }

        # ----------------------------------------------------
        # TABLE COUNTS
        # ----------------------------------------------------

        for page in document.get(
            "pages",
            []
        ):

            for table in page.get(
                "tables",
                []
            ):

                rows = table.get(
                    "rows",
                    []
                )

                columns = table.get(
                    "columns",
                    []
                )

                structural_summary[
                    "tables"
                ].append(
                    {

                        "table_id": table.get(
                            "table_id"
                        ),

                        "page": page.get(
                            "page_number"
                        ),

                        "row_count": len(
                            rows
                        ),

                        "column_count": len(
                            columns
                        ),

                        "headers": table.get(
                            "headers",
                            []
                        ),
                    }
                )

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total_time = (
            time.time()
            - total_start
        )

        print(
            f"[TIMING] "
            f"TOTAL PROCESSING TIME: "
            f"{total_time:.2f} seconds"
        )

        print(
            "=" * 70
        )

        return {

            "document_id": document_id,

            "document": document,

            "classification": classification,

            "chunks": chunks,

            "storage": storage_result,

            "extracted": extracted,

            "merged": merged,

            "validation": validation,

            "structural_summary": (
                structural_summary
            ),

            "timing": {

                "loading_ocr": load_time,

                "classification": (
                    classification_time
                ),

                "chunking": chunk_time,

                "vector_store": vector_time,

                "extraction": extraction_time,

                "merge": merge_time,

                "validation": validation_time,

                "total": total_time,
            },
        }


# Backward compatibility
Orchestrator = DocumentOrchestrator