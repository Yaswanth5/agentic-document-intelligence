"""
LangGraph document-processing workflow.

This file is responsible only for orchestration.

The actual document-processing logic remains inside:
    processing.document_loader
    processing.chunker
    agents.classifier
    agents.extractor
    agents.merger
    agents.validator
    storage.vector_store

LangGraph controls the execution order and maintains state between nodes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from processing.document_loader import DocumentLoader
from processing.chunker import DocumentChunker

from agents.classifier import DocumentClassifier
from agents.extractor import GenericExtractor
from agents.merger import ResultMerger
from agents.validator import ResultValidator

from storage.vector_store import VectorStore


# ============================================================
# STATE
# ============================================================

class DocumentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Every node receives the current state and returns
    only the fields that it wants to update.
    """

    # Input
    file_path: str

    # Identity
    document_id: str

    # Main document objects
    document: Any
    classification: dict
    chunks: list
    storage: dict

    # Extraction pipeline
    extracted: dict
    merged: dict
    validation: dict

    # Structural information
    structural_summary: dict

    # Timing
    timing: dict

    # Error handling
    error: str


# ============================================================
# COMPONENTS
# ============================================================

loader = DocumentLoader()
classifier = DocumentClassifier()
chunker = DocumentChunker()
extractor = GenericExtractor()
merger = ResultMerger()
validator = ResultValidator()
vector_store = VectorStore()


# ============================================================
# HELPER
# ============================================================

def _safe_dict(value):
    """
    Make sure a value is a dictionary.
    """

    if isinstance(value, dict):
        return value

    return {}


def _build_structural_summary(document):
    """
    Build the same structural summary that the previous
    DocumentOrchestrator generated.
    """

    if not isinstance(document, dict):
        return {}

    pages = document.get("pages", []) or []

    summary = {
        "page_count": document.get(
            "page_count",
            len(pages)
        ),

        "total_text_characters": len(
            document.get("text", "") or ""
        ),

        "image_count": len(
            document.get("images", []) or []
        ),

        "table_count": len(
            document.get("tables", []) or []
        ),

        "layout_object_count": len(
            document.get("layout", []) or []
        ),

        "pages_with_native_text": sum(
            bool(page.get("native_text", ""))
            for page in pages
            if isinstance(page, dict)
        ),

        "pages_with_ocr": sum(
            bool(page.get("ocr_text", ""))
            for page in pages
            if isinstance(page, dict)
        ),

        "tables": [],
    }

    # --------------------------------------------------------
    # Table summaries
    # --------------------------------------------------------

    for page in pages:

        if not isinstance(page, dict):
            continue

        page_number = page.get("page_number")

        for table in page.get("tables", []) or []:

            if not isinstance(table, dict):
                continue

            rows = table.get("rows", []) or []
            columns = table.get("columns", []) or []

            summary["tables"].append(
                {
                    "table_id": table.get("table_id"),

                    "page": page_number,

                    "row_count": len(rows),

                    "column_count": len(columns),

                    "headers": table.get(
                        "headers",
                        []
                    ),
                }
            )

    return summary


def _merge_timing(existing, name, value):
    """
    Update timing information safely.
    """

    timing = dict(existing or {})

    timing[name] = value

    return timing


# ============================================================
# NODE 1 — INITIALIZE
# ============================================================

def initialize_document(state: DocumentState):
    """
    Generate a unique document ID.

    This node is intentionally separate from loading so that
    document identity exists before chunks are written into
    ChromaDB.
    """

    document_id = str(uuid.uuid4())

    print()
    print("=" * 70)
    print("LANGGRAPH DOCUMENT PROCESSING START")
    print(f"Document ID: {document_id}")
    print("=" * 70)

    return {
        "document_id": document_id,
        "timing": {},
    }


# ============================================================
# NODE 2 — LOAD
# ============================================================

def load_document(state: DocumentState):
    """
    Load the document.

    Existing DocumentLoader performs:
        - PDF extraction
        - OCR
        - image extraction
        - table extraction
        - layout extraction
        - DOCX/XLSX/PPTX/TXT/etc. handling
    """

    file_path = state["file_path"]

    start = time.time()

    print("[LANGGRAPH] Loading document...")

    document = loader.load(file_path)

    elapsed = time.time() - start

    print(
        f"[TIMING] Document loading/OCR: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "document": document,
        "timing": _merge_timing(
            state.get("timing"),
            "loading_ocr",
            elapsed,
        ),
    }


# ============================================================
# NODE 3 — CLASSIFICATION
# ============================================================

def classify_document(state: DocumentState):
    """
    Classify the document using the existing
    DocumentClassifier.
    """

    document = state["document"]

    start = time.time()

    print("[LANGGRAPH] Classifying document...")

    classification = classifier.classify(document)

    elapsed = time.time() - start

    print(
        f"[TIMING] Classification: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "classification": classification,
        "timing": _merge_timing(
            state.get("timing"),
            "classification",
            elapsed,
        ),
    }


# ============================================================
# NODE 4 — CHUNKING
# ============================================================

def chunk_document(state: DocumentState):
    """
    Convert the normalized document into chunks.

    Existing DocumentChunker remains unchanged.
    """

    document = state["document"]

    start = time.time()

    print("[LANGGRAPH] Chunking document...")

    chunks = chunker.chunk(document)

    elapsed = time.time() - start

    print(
        f"[TIMING] Chunking: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "chunks": chunks,
        "timing": _merge_timing(
            state.get("timing"),
            "chunking",
            elapsed,
        ),
    }


# ============================================================
# NODE 5 — VECTOR STORAGE
# ============================================================

def store_chunks(state: DocumentState):
    """
    Store chunks in ChromaDB.

    Each chunk is associated with document_id.
    """

    document_id = state["document_id"]
    chunks = state["chunks"]

    start = time.time()

    print("[LANGGRAPH] Writing chunks to vector store...")

    storage_result = vector_store.add_chunks(
        document_id,
        chunks,
    )

    elapsed = time.time() - start

    print(
        f"[TIMING] Vector store: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "storage": storage_result,
        "timing": _merge_timing(
            state.get("timing"),
            "vector_store",
            elapsed,
        ),
    }


# ============================================================
# NODE 6 — EXTRACTION
# ============================================================

def extract_document(state: DocumentState):
    """
    Perform deterministic + LLM-based extraction.
    """

    document = state["document"]
    chunks = state["chunks"]

    start = time.time()

    print("[LANGGRAPH] Extracting information...")

    extracted = extractor.extract(
        chunks,
        document,
    )

    elapsed = time.time() - start

    print(
        f"[TIMING] Semantic extraction: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "extracted": extracted,
        "timing": _merge_timing(
            state.get("timing"),
            "extraction",
            elapsed,
        ),
    }


# ============================================================
# NODE 7 — MERGE
# ============================================================

def merge_results(state: DocumentState):
    """
    Merge extraction results coming from multiple chunks.
    """

    extracted = state.get("extracted", {})

    start = time.time()

    print("[LANGGRAPH] Merging extraction results...")

    if hasattr(merger, "merge"):
        merged = merger.merge(extracted)
    else:
        merged = extracted

    elapsed = time.time() - start

    print(
        f"[TIMING] Merge: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "merged": merged,
        "timing": _merge_timing(
            state.get("timing"),
            "merge",
            elapsed,
        ),
    }


# ============================================================
# NODE 8 — VALIDATION
# ============================================================

def validate_results(state: DocumentState):
    """
    Validate merged extraction results.
    """

    merged = state.get("merged", {})

    start = time.time()

    print("[LANGGRAPH] Validating results...")

    if hasattr(validator, "validate_total"):
        validation = validator.validate_total(
            merged
        )
    else:
        validation = {}

    elapsed = time.time() - start

    print(
        f"[TIMING] Validation: "
        f"{elapsed:.2f} seconds"
    )

    return {
        "validation": validation,
        "timing": _merge_timing(
            state.get("timing"),
            "validation",
            elapsed,
        ),
    }


# ============================================================
# NODE 9 — FINALIZE
# ============================================================

def finalize_document(state: DocumentState):
    """
    Build final structural information and total timing.
    """

    start = time.time()

    document = state.get("document", {})

    structural_summary = _build_structural_summary(
        document
    )

    timing = dict(
        state.get("timing", {})
    )

    # Total time is measured from the beginning of the
    # graph. Since individual node times are already stored,
    # calculate total from their sum as a stable approximation.
    timing["total"] = sum(
        value
        for key, value in timing.items()
        if isinstance(value, (int, float))
        and key != "total"
    )

    print(
        f"[TIMING] TOTAL PROCESSING TIME: "
        f"{timing['total']:.2f} seconds"
    )

    print("=" * 70)
    print("LANGGRAPH DOCUMENT PROCESSING COMPLETE")
    print("=" * 70)

    return {
        "structural_summary": structural_summary,
        "timing": timing,
    }


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_document_graph():
    """
    Build and compile the LangGraph document pipeline.
    """

    workflow = StateGraph(DocumentState)

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    workflow.add_node(
        "initialize",
        initialize_document,
    )

    workflow.add_node(
        "load",
        load_document,
    )

    workflow.add_node(
        "classify",
        classify_document,
    )

    workflow.add_node(
        "chunk",
        chunk_document,
    )

    workflow.add_node(
        "store",
        store_chunks,
    )

    workflow.add_node(
        "extract",
        extract_document,
    )

    workflow.add_node(
        "merge",
        merge_results,
    )

    workflow.add_node(
        "validate",
        validate_results,
    )

    workflow.add_node(
        "finalize",
        finalize_document,
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    workflow.add_edge(
        START,
        "initialize",
    )

    workflow.add_edge(
        "initialize",
        "load",
    )

    workflow.add_edge(
        "load",
        "classify",
    )

    workflow.add_edge(
        "classify",
        "chunk",
    )

    workflow.add_edge(
        "chunk",
        "store",
    )

    workflow.add_edge(
        "store",
        "extract",
    )

    workflow.add_edge(
        "extract",
        "merge",
    )

    workflow.add_edge(
        "merge",
        "validate",
    )

    workflow.add_edge(
        "validate",
        "finalize",
    )

    workflow.add_edge(
        "finalize",
        END,
    )

    return workflow.compile()


# ============================================================
# COMPILED GRAPH
# ============================================================

document_graph = build_document_graph()