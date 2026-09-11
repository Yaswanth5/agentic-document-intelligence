from storage.vector_store import VectorStore


class DocumentRetriever:
    """
    Generic document retrieval agent.

    Retrieval is isolated by document_id so that a question asked
    against the current document can NEVER retrieve chunks from
    previously processed documents.
    """

    def __init__(self):
        self.store = VectorStore()

    # =========================================================
    # RETRIEVE RELEVANT DOCUMENT EVIDENCE
    # =========================================================

    def retrieve(
        self,
        query,
        document_id,
        top_k=5,
    ):

        if not query:
            return []

        if not document_id:
            print(
                "[RETRIEVER] Retrieval blocked: "
                "document_id was not provided."
            )
            return []

        result = self.store.search(
            query=query,
            n_results=top_k,
            document_id=document_id,
        )

        if not isinstance(result, dict):
            return []

        documents = result.get(
            "documents",
            [[]],
        )

        metadatas = result.get(
            "metadatas",
            [[]],
        )

        distances = result.get(
            "distances",
            [[]],
        )

        ids = result.get(
            "ids",
            [[]],
        )

        # -----------------------------------------------------
        # Chroma returns nested lists because we query one
        # question at a time.
        # -----------------------------------------------------

        documents = (
            documents[0]
            if documents
            and isinstance(documents[0], list)
            else documents
        )

        metadatas = (
            metadatas[0]
            if metadatas
            and isinstance(metadatas[0], list)
            else metadatas
        )

        distances = (
            distances[0]
            if distances
            and isinstance(distances[0], list)
            else distances
        )

        ids = (
            ids[0]
            if ids
            and isinstance(ids[0], list)
            else ids
        )

        contexts = []

        # =====================================================
        # BUILD GROUNDED CONTEXT OBJECTS
        # =====================================================

        for index, document in enumerate(documents):

            if document is None:
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                and isinstance(metadatas[index], dict)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            chunk_id = (
                ids[index]
                if index < len(ids)
                else None
            )

            # -------------------------------------------------
            # EXTRA SAFETY CHECK
            #
            # Chroma already filters by document_id, but we
            # verify the returned metadata as well.
            # -------------------------------------------------

            returned_document_id = str(
                metadata.get(
                    "document_id",
                    ""
                )
            )

            if returned_document_id != str(document_id):
                print(
                    "[RETRIEVER] Ignoring chunk from "
                    f"different document: {returned_document_id}"
                )
                continue

            contexts.append(
                {
                    "text": str(document),
                    "metadata": metadata,
                    "distance": distance,
                    "chunk_id": chunk_id,
                }
            )

        return contexts


# =============================================================
# BACKWARD COMPATIBILITY
# =============================================================

class RetrievalAgent(DocumentRetriever):
    """
    Backward-compatible alias for existing code.
    """
    pass