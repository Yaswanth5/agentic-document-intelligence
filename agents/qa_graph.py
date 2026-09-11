"""
LangGraph question-answering workflow.

Flow:

Question
   |
   v
Retrieve
   |
   v
Analyze Question
   |
   +--------------------+
   |                    |
structured              llm
   |                    |
   v                    v
Structured          LLM Reasoning
Reasoning               |
   |                    |
   +---------+----------+
             |
             v
           Answer
"""

from __future__ import annotations

import re
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from agents.retriever import DocumentRetriever
from agents.reasoning import ReasoningAgent


# ============================================================
# STATE
# ============================================================

class QAState(TypedDict, total=False):

    question: str

    document_id: str

    top_k: int

    contexts: list

    question_type: str

    route: str

    answer: str

    error: str


# ============================================================
# COMPONENTS
# ============================================================

retriever = DocumentRetriever()
reasoning_agent = ReasoningAgent()


# ============================================================
# NODE 1 — RETRIEVE
# ============================================================

def retrieve_context(state: QAState):

    question = state["question"]
    document_id = state["document_id"]

    top_k = state.get(
        "top_k",
        5,
    )

    print(
        "[LANGGRAPH QA] Retrieving context..."
    )

    contexts = retriever.retrieve(
        query=question,
        document_id=document_id,
        top_k=top_k,
    )

    return {
        "contexts": contexts
    }


# ============================================================
# NODE 2 — QUESTION CLASSIFICATION
# ============================================================

def classify_question(state: QAState):

    question = state["question"]

    q = question.lower().strip()

    # --------------------------------------------------------
    # Table-related questions
    # --------------------------------------------------------

    table_patterns = [
        r"\btable\b",
        r"\bcolumn\b",
        r"\bcolumns\b",
        r"\brow\b",
        r"\brows\b",
        r"\bcell\b",
        r"\bheader\b",
        r"\bheaders\b",
    ]

    # --------------------------------------------------------
    # Counting
    # --------------------------------------------------------

    count_patterns = [
        r"\bhow many\b",
        r"\bcount\b",
        r"\bnumber of\b",
    ]

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    aggregation_patterns = [
        r"\baverage\b",
        r"\bmean\b",
        r"\bsum\b",
        r"\btotal\b",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\bmax\b",
        r"\bmin\b",
    ]

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    lookup_patterns = [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwhich\b",
        r"\blist\b",
        r"\bshow\b",
    ]

    if any(
        re.search(pattern, q)
        for pattern in table_patterns
    ):
        question_type = "structured"

    elif any(
        re.search(pattern, q)
        for pattern in count_patterns
    ):
        question_type = "structured"

    elif any(
        re.search(pattern, q)
        for pattern in aggregation_patterns
    ):
        question_type = "structured"

    elif any(
        re.search(pattern, q)
        for pattern in lookup_patterns
    ):
        question_type = "lookup"

    else:
        question_type = "semantic"

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    if question_type in {
        "structured",
        "lookup",
    }:
        route = "structured"

    else:
        route = "llm"

    print(
        f"[LANGGRAPH QA] Question type: "
        f"{question_type}"
    )

    print(
        f"[LANGGRAPH QA] Route: "
        f"{route}"
    )

    return {
        "question_type": question_type,
        "route": route,
    }


# ============================================================
# NODE 3 — STRUCTURED REASONING
# ============================================================

def structured_reasoning(state: QAState):

    question = state["question"]

    contexts = state.get(
        "contexts",
        [],
    )

    print(
        "[LANGGRAPH QA] "
        "Running structured reasoning..."
    )

    # Your existing ReasoningAgent already contains
    # deterministic structured-question handling.
    #
    # We intentionally reuse that implementation rather
    # than duplicating all table/count/aggregation logic.

    answer = reasoning_agent.answer(
        question,
        contexts,
    )

    return {
        "answer": answer
    }


# ============================================================
# NODE 4 — LLM REASONING
# ============================================================

def llm_reasoning(state: QAState):

    question = state["question"]

    contexts = state.get(
        "contexts",
        [],
    )

    print(
        "[LANGGRAPH QA] "
        "Running LLM reasoning..."
    )

    answer = reasoning_agent.answer(
        question,
        contexts,
    )

    return {
        "answer": answer
    }


# ============================================================
# ROUTER
# ============================================================

def route_question(state: QAState):

    route = state.get(
        "route",
        "llm",
    )

    if route == "structured":
        return "structured"

    return "llm"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_qa_graph():

    workflow = StateGraph(QAState)

    # Nodes

    workflow.add_node(
        "retrieve",
        retrieve_context,
    )

    workflow.add_node(
        "classify_question",
        classify_question,
    )

    workflow.add_node(
        "structured_reasoning",
        structured_reasoning,
    )

    workflow.add_node(
        "llm_reasoning",
        llm_reasoning,
    )

    # START -> retrieve

    workflow.add_edge(
        START,
        "retrieve",
    )

    # retrieve -> classify

    workflow.add_edge(
        "retrieve",
        "classify_question",
    )

    # Conditional routing

    workflow.add_conditional_edges(
        "classify_question",
        route_question,
        {
            "structured": "structured_reasoning",
            "llm": "llm_reasoning",
        },
    )

    # Both paths terminate

    workflow.add_edge(
        "structured_reasoning",
        END,
    )

    workflow.add_edge(
        "llm_reasoning",
        END,
    )

    return workflow.compile()


# ============================================================
# COMPILED GRAPH
# ============================================================

qa_graph = build_qa_graph()