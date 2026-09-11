"""
Document orchestrator.

The actual workflow is implemented using LangGraph
in agents.document_graph.

This class is retained so that the existing Streamlit
application does not need major changes.
"""

from agents.document_graph import document_graph


class DocumentOrchestrator:
    """
    Backward-compatible interface around the LangGraph
    document-processing workflow.
    """

    def __init__(self):
        self.graph = document_graph

    def process(self, file_path: str):
        """
        Process a document through the LangGraph workflow.

        Parameters
        ----------
        file_path : str
            Path of the uploaded document.

        Returns
        -------
        dict
            Final LangGraph state.
        """

        if not file_path:
            raise ValueError(
                "file_path is required"
            )

        result = self.graph.invoke(
            {
                "file_path": file_path
            }
        )

        return result


# Backward compatibility
Orchestrator = DocumentOrchestrator