import os
import time
from pathlib import Path

import streamlit as st

from config import (
    RETRIEVAL_TOP_K,
    RETRIEVAL_MAX_K,
)

from agents.reasoning import ReasoningAgent
from agents.retriever import DocumentRetriever
from agents.orchestrator import DocumentOrchestrator


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Agentic Document Intelligence",
    page_icon="📄",
    layout="wide",
)


# =========================================================
# SESSION STATE
# =========================================================

if "processed_result" not in st.session_state:
    st.session_state.processed_result = None

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# =========================================================
# CACHED COMPONENTS
# =========================================================

@st.cache_resource
def get_orchestrator():
    return DocumentOrchestrator()


@st.cache_resource
def get_retriever():
    return DocumentRetriever()


@st.cache_resource
def get_reasoning_agent():
    return ReasoningAgent()


# =========================================================
# APPLICATION OBJECTS
# =========================================================

orchestrator = get_orchestrator()
retriever = get_retriever()
reasoning_agent = get_reasoning_agent()


# =========================================================
# HEADER
# =========================================================

st.title("📄 Agentic Document Intelligence Platform")

st.caption(
    "Document-agnostic extraction, multimodal understanding, "
    "retrieval and grounded reasoning"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("System")

    st.write(
        "Upload a document and the platform will "
        "automatically process its structure, text, "
        "tables, images and other available content."
    )

    st.divider()

    st.subheader("Supported formats")

    st.write(
        """
        • PDF  
        • PNG / JPG / JPEG / WEBP / BMP  
        • TIFF  
        • DOCX  
        • XLSX / XLS  
        • PPTX  
        • TXT  
        • Markdown  
        • CSV
        """
    )

    st.divider()

    st.subheader("Retrieval")

    st.write(
        f"Default candidates: **{RETRIEVAL_TOP_K}**"
    )

    st.write(
        f"Maximum candidates: **{RETRIEVAL_MAX_K}**"
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "docx",
        "xlsx",
        "xls",
        "pptx",
        "txt",
        "md",
        "csv",
    ],
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None:

    upload_dir = Path("storage/uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        upload_dir /
        uploaded_file.name
    )

    # -----------------------------------------------------
    # Save uploaded document
    # -----------------------------------------------------

    try:

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

    except Exception as exc:

        st.error(
            f"Unable to save uploaded document: {exc}"
        )

        st.stop()

    # -----------------------------------------------------
    # Avoid reprocessing same upload
    # -----------------------------------------------------

    previous_result = (
        st.session_state.processed_result
    )

    should_process = True

    if previous_result:

        previous_path = (
            previous_result.get(
                "file_path"
            )
        )

        if previous_path:

            try:

                if (
                    Path(previous_path).name
                    == uploaded_file.name
                    and
                    Path(previous_path).stat().st_size
                    == uploaded_file.size
                ):

                    should_process = False

            except Exception:
                pass

    # -----------------------------------------------------
    # Process
    # -----------------------------------------------------

    if should_process:

        st.session_state.processed_result = None
        st.session_state.active_document_id = None
        st.session_state.qa_history = []

        st.divider()

        st.subheader(
            "Processing document"
        )

        progress = st.progress(0)

        status = st.empty()

        try:

            status.info(
                "Running document intelligence pipeline..."
            )

            progress.progress(10)

            start_time = time.perf_counter()

            result = orchestrator.process(
                str(file_path)
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            progress.progress(100)

            status.success(
                f"Processing completed in "
                f"{elapsed:.2f} seconds."
            )

            st.session_state.processed_result = result

            st.session_state.active_document_id = (
                result.get("document_id")
            )

        except Exception as exc:

            progress.empty()

            status.empty()

            st.error(
                "Document processing failed."
            )

            st.exception(exc)

            st.stop()


# =========================================================
# GET CURRENT RESULT
# =========================================================

result = (
    st.session_state.processed_result
)

if result is None:

    st.info(
        "Upload a document to begin."
    )

    st.stop()


# =========================================================
# DOCUMENT ID
# =========================================================

document_id = (
    result.get("document_id")
)

st.divider()

st.subheader(
    "Document processed"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Document ID",
        str(document_id)
        if document_id
        else "N/A"
    )

with col2:

    chunks = result.get(
        "chunks",
        []
    )

    st.metric(
        "Chunks",
        len(chunks)
        if isinstance(chunks, list)
        else 0
    )

with col3:

    timing = result.get(
        "timing",
        {}
    )

    st.metric(
        "Processing time",
        f"{timing.get('total_seconds', 0):.2f}s"
    )


# =========================================================
# CLASSIFICATION
# =========================================================

classification = result.get(
    "classification",
    {}
)

with st.expander(
    "📑 Document Classification",
    expanded=True
):

    if isinstance(
        classification,
        dict
    ):

        display_classification = {
            key: value
            for key, value in classification.items()
            if value is not None
        }

        st.json(
            display_classification
        )

    else:

        st.write(
            classification
        )


# =========================================================
# EXTRACTED INFORMATION
# =========================================================

st.divider()

st.subheader(
    "🔎 Extracted Information"
)

extracted = result.get(
    "extracted",
    []
)

if extracted:

    if isinstance(
        extracted,
        list
    ):

        for index, item in enumerate(
            extracted,
            start=1
        ):

            with st.container(
                border=True
            ):

                if isinstance(
                    item,
                    dict
                ):

                    field = item.get(
                        "field",
                        item.get(
                            "name",
                            "Information"
                        )
                    )

                    value = item.get(
                        "value",
                        item.get(
                            "text",
                            ""
                        )
                    )

                    st.markdown(
                        f"**{field}**"
                    )

                    st.write(
                        value
                    )

                    evidence = item.get(
                        "evidence"
                    )

                    if evidence:

                        st.caption(
                            f"Evidence: {evidence}"
                        )

                else:

                    st.write(
                        item
                    )

    elif isinstance(
        extracted,
        dict
    ):

        st.json(
            extracted
        )

    else:

        st.write(
            extracted
        )

else:

    st.info(
        "No structured information was extracted."
    )


# =========================================================
# MERGED RESULT
# =========================================================

merged = result.get(
    "merged"
)

if merged:

    with st.expander(
        "🔗 Merged Information"
    ):

        if isinstance(
            merged,
            (dict, list)
        ):

            st.json(
                merged
            )

        else:

            st.write(
                merged
            )


# =========================================================
# MULTIMODAL / VISUAL CONTENT
# =========================================================

document = result.get(
    "document",
    {}
)

with st.expander(
    "🖼️ Multimodal / Visual Content",
    expanded=False
):

    images = []

    if isinstance(
        document,
        dict
    ):

        images = document.get(
            "images",
            []
        )

    if images:

        st.write(
            f"Detected {len(images)} image(s)."
        )

        for index, image in enumerate(
            images,
            start=1
        ):

            if not isinstance(
                image,
                dict
            ):
                continue

            image_path = image.get(
                "path"
            )

            page = image.get(
                "page",
                "unknown"
            )

            caption = image.get(
                "caption",
                ""
            )

            if image_path and Path(
                image_path
            ).exists():

                st.image(
                    image_path,
                    caption=(
                        f"Image {index} | "
                        f"Page {page}"
                        + (
                            f" | {caption}"
                            if caption
                            else ""
                        )
                    ),
                )

            elif image.get("image"):

                st.image(
                    image.get("image"),
                    caption=(
                        f"Image {index} | "
                        f"Page {page}"
                    ),
                )

            else:

                st.write(
                    image
                )

    else:

        st.info(
            "No embedded images were detected."
        )


# =========================================================
# DOCUMENT STRUCTURE
# =========================================================

with st.expander(
    "🧩 Document Structure",
    expanded=False
):

    if isinstance(
        document,
        dict
    ):

        pages = document.get(
            "pages",
            []
        )

        tables = document.get(
            "tables",
            []
        )

        st.write(
            f"Pages/sections: "
            f"{len(pages) if isinstance(pages, list) else 0}"
        )

        st.write(
            f"Tables: "
            f"{len(tables) if isinstance(tables, list) else 0}"
        )

        if tables:

            st.subheader("Tables")

            for index, table in enumerate(
                tables,
                start=1
            ):

                st.write(
                    f"Table {index}"
                )

                if isinstance(
                    table,
                    dict
                ):

                    st.json(table)

                else:

                    st.write(table)

        if pages:

            st.subheader(
                "Pages / Sections"
            )

            for index, page in enumerate(
                pages,
                start=1
            ):

                with st.expander(
                    f"Page / Section {index}"
                ):

                    if isinstance(
                        page,
                        dict
                    ):

                        st.json(page)

                    else:

                        st.write(page)

    else:

        st.write(
            document
        )


# =========================================================
# COMPLETE EXTRACTED TEXT
# =========================================================

with st.expander(
    "📜 Complete Document Text",
    expanded=False
):

    complete_text = ""

    if isinstance(
        document,
        dict
    ):

        complete_text = document.get(
            "text",
            document.get(
                "markdown",
                ""
            )
        )

    if complete_text:

        st.text_area(
            "Extracted text",
            complete_text,
            height=500,
        )

    else:

        st.info(
            "No plain text representation available."
        )


# =========================================================
# CHUNKS
# =========================================================

with st.expander(
    "🧱 Document Chunks",
    expanded=False
):

    if chunks:

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            if not isinstance(
                chunk,
                dict
            ):

                st.write(chunk)

                continue

            chunk_id = chunk.get(
                "chunk_id",
                index
            )

            page = chunk.get(
                "page",
                "unknown"
            )

            chunk_type = chunk.get(
                "type",
                chunk.get(
                    "metadata",
                    {}
                ).get(
                    "type",
                    "text"
                )
                if isinstance(
                    chunk.get("metadata"),
                    dict
                )
                else "text"
            )

            text = chunk.get(
                "text",
                ""
            )

            st.markdown(
                f"### Chunk {chunk_id}"
            )

            st.caption(
                f"Page: {page} | Type: {chunk_type}"
            )

            st.text(text)

            images = chunk.get(
                "images",
                []
            )

            if images:

                st.caption(
                    f"Images: {len(images)}"
                )

            tables = chunk.get(
                "tables",
                []
            )

            if tables:

                st.caption(
                    f"Tables: {len(tables)}"
                )

    else:

        st.info(
            "No chunks available."
        )


# =========================================================
# SPATIAL INFORMATION
# =========================================================

with st.expander(
    "📐 Spatial / Layout Information",
    expanded=False
):

    spatial_found = False

    if chunks:

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            if not isinstance(
                chunk,
                dict
            ):
                continue

            blocks = chunk.get(
                "blocks",
                []
            )

            lines = chunk.get(
                "lines",
                []
            )

            if blocks:

                spatial_found = True

                st.markdown(
                    f"**Chunk {index} — Blocks**"
                )

                st.json(blocks)

            if lines:

                spatial_found = True

                st.markdown(
                    f"**Chunk {index} — Lines**"
                )

                st.json(lines)

    if not spatial_found:

        st.info(
            "No explicit spatial information available."
        )


# =========================================================
# VALIDATION
# =========================================================

validation = result.get(
    "validation"
)

with st.expander(
    "✅ Validation",
    expanded=False
):

    if validation:

        if isinstance(
            validation,
            (dict, list)
        ):

            st.json(validation)

        else:

            st.write(validation)

    else:

        st.info(
            "No validation information available."
        )


# =========================================================
# ASK QUESTIONS
# =========================================================

st.divider()

st.subheader(
    "💬 Ask Questions About This Document"
)

st.caption(
    "Questions are answered only from the currently "
    "processed document."
)

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What are the important details "
        "mentioned in this document?"
    ),
)


# =========================================================
# QUESTION ANSWERING
# =========================================================

if st.button(
    "Ask Question",
    type="primary",
):

    if not question or not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not document_id:

        st.error(
            "No active document is available."
        )

    else:

        with st.spinner(
            "Searching document and reasoning over evidence..."
        ):

            try:

                # =================================================
                # RETRIEVAL CONFIGURATION
                # =================================================

                retrieval_k = min(
                    RETRIEVAL_TOP_K,
                    RETRIEVAL_MAX_K,
                )

                # =================================================
                # RETRIEVE
                # =================================================

                contexts = retriever.retrieve(
                    query=question,
                    document_id=document_id,
                    top_k=retrieval_k,
                )

                # =================================================
                # RETRIEVAL DEBUG
                # =================================================

                print(
                    "\n"
                    "========== RETRIEVAL DEBUG =========="
                )

                print(
                    "QUESTION:",
                    question
                )

                print(
                    "DOCUMENT ID:",
                    document_id
                )

                print(
                    "RETRIEVAL K:",
                    retrieval_k
                )

                print(
                    "CONTEXT COUNT:",
                    len(contexts)
                )

                for i, ctx in enumerate(
                    contexts,
                    start=1
                ):

                    print(
                        f"\n--- CONTEXT {i} ---"
                    )

                    if not isinstance(
                        ctx,
                        dict
                    ):

                        print(
                            "INVALID CONTEXT:",
                            ctx
                        )

                        continue

                    print(
                        "CHUNK ID:",
                        ctx.get(
                            "chunk_id"
                        )
                    )

                    print(
                        "DISTANCE:",
                        ctx.get(
                            "distance"
                        )
                    )

                    print(
                        "METADATA:",
                        ctx.get(
                            "metadata"
                        )
                    )

                    print(
                        "TEXT:"
                    )

                    print(
                        ctx.get(
                            "text",
                            ""
                        )[:5000]
                    )

                print(
                    "=====================================\n"
                )

                # =================================================
                # SAFETY FILTER
                # =================================================
                # Never allow evidence belonging to another
                # document into the reasoning layer.
                # =================================================

                safe_contexts = []

                for context in contexts:

                    if not isinstance(
                        context,
                        dict
                    ):
                        continue

                    metadata = context.get(
                        "metadata",
                        {}
                    )

                    if not isinstance(
                        metadata,
                        dict
                    ):
                        continue

                    returned_document_id = str(
                        metadata.get(
                            "document_id",
                            ""
                        )
                    )

                    if (
                        returned_document_id
                        == str(document_id)
                    ):

                        safe_contexts.append(
                            context
                        )

                # =================================================
                # DISPLAY RETRIEVAL STATUS
                # =================================================

                st.info(
                    f"Retrieved {len(contexts)} context(s); "
                    f"{len(safe_contexts)} passed document safety filtering."
                )

                # =================================================
                # REASONING
                # =================================================

                answer = reasoning_agent.answer(
                    question=question,
                    contexts=safe_contexts,
                )

                # =================================================
                # SAVE HISTORY
                # =================================================

                st.session_state.qa_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "contexts": safe_contexts,
                    }
                )

            except Exception as exc:

                st.error(
                    "Question answering failed."
                )

                st.exception(exc)


# =========================================================
# DISPLAY QA HISTORY
# =========================================================

if st.session_state.qa_history:

    st.subheader(
        "Answers"
    )

    for item in reversed(
        st.session_state.qa_history
    ):

        question_text = item.get(
            "question",
            ""
        )

        answer_text = item.get(
            "answer",
            ""
        )

        contexts = item.get(
            "contexts",
            []
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"**Question:** {question_text}"
            )

            st.markdown(
                f"**Answer:** {answer_text}"
            )

            # =====================================================
            # EVIDENCE
            # =====================================================

            if contexts:

                with st.expander(
                    "🔍 Evidence used"
                ):

                    for index, context in enumerate(
                        contexts,
                        start=1
                    ):

                        if not isinstance(
                            context,
                            dict
                        ):
                            continue

                        metadata = context.get(
                            "metadata",
                            {}
                        )

                        text = context.get(
                            "text",
                            ""
                        )

                        if not isinstance(
                            metadata,
                            dict
                        ):

                            metadata = {}

                        page = metadata.get(
                            "page",
                            metadata.get(
                                "page_number",
                                "unknown"
                            )
                        )

                        source = metadata.get(
                            "source",
                            "unknown"
                        )

                        chunk_type = metadata.get(
                            "type",
                            "text"
                        )

                        distance = context.get(
                            "distance"
                        )

                        chunk_id = context.get(
                            "chunk_id"
                        )

                        st.markdown(
                            f"**Evidence {index}**"
                        )

                        st.caption(
                            f"Chunk: {chunk_id} | "
                            f"Type: {chunk_type} | "
                            f"Page: {page} | "
                            f"Source: {source} | "
                            f"Distance: {distance}"
                        )

                        st.text(text)

                        st.divider()

            else:

                st.warning(
                    "No evidence was retrieved for this question."
                )


# =========================================================
# DEBUG INFORMATION
# =========================================================

with st.expander(
    "🛠️ Debug Information",
    expanded=False
):

    debug_info = {
        "document_id": document_id,
        "file_path": result.get(
            "file_path"
        ),
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "retrieval_max_k": RETRIEVAL_MAX_K,
        "chunks": len(chunks)
        if isinstance(chunks, list)
        else 0,
        "qa_count": len(
            st.session_state.qa_history
        ),
        "timing": result.get(
            "timing",
            {}
        ),
    }

    st.json(
        debug_info
    )