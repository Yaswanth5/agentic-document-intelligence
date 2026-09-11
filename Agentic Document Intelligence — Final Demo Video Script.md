# AGENTIC DOCUMENT INTELLIGENCE
## Final Demo Video Script

**Target length:** 7–10 minutes  
**Documents:** Synthetic documents only

---

# 0. BEFORE RECORDING

### Start

- Start Ollama.
- Start Streamlit.
- Keep the terminal visible during at least one document-processing run.
- Keep the following synthetic documents ready:

```text
01_Synthetic_Business_Report.docx
02_Synthetic_Operations_Data.xlsx
03_Synthetic_Floor_Plan.png
04_Synthetic_Text_Image.png
05_Synthetic_Multi_Page_Report.pdf
```

### Final architecture to show

```text
                         STREAMLIT UI
                              |
                              v
                   DocumentOrchestrator
                              |
                              v
                  ┌─────────────────────┐
                  │  LANGGRAPH DOCUMENT │
                  │       GRAPH         │
                  └─────────────────────┘
                              |
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
     Loader               Classifier             Chunker
        |                     |                     |
        └─────────────────────┼─────────────────────┘
                              ↓
                         ChromaDB
                              |
                              ↓
                         Extractor
                              |
                              ↓
                           Merger
                              |
                              ↓
                         Validator
                              |
                              ↓
                         Final State


                         USER QUESTION
                              |
                              v
                    ┌──────────────────┐
                    │   LANGGRAPH QA   │
                    │      GRAPH       │
                    └──────────────────┘
                              |
                              v
                          Retriever
                              |
                              v
                     Question Routing
                        /          \
                       /            \
                Structured         LLM
                  Route           Reasoning
                       \            /
                        \          /
                         Reasoning
                            |
                            v
                          Answer
                            +
                         Evidence
```

---

# 1. INTRODUCTION — 30–45 SEC

### SAY

> "Hello, this is my Agentic Document Intelligence Platform.
>
> The objective of this project is to build a generic system that can ingest heterogeneous documents, preserve their text and structural information, create searchable representations, extract useful information, and answer natural-language questions using document-grounded evidence.
>
> The system is designed to be document-agnostic. I am not hard-coding a fixed schema for a particular document type.
>
> The final architecture combines conventional document-processing components with agentic workflow orchestration using LangGraph."

### SHOW

- Streamlit application
- Upload section
- Supported document formats
- Project folder structure

### IMPORTANT

Don't spend too much time explaining every technology here. The architecture section will cover that.

---

# 2. FINAL ARCHITECTURE — 60–75 SEC

### SAY

> "The final architecture has two LangGraph workflows: one for document processing and one for question answering.
>
> When a document is uploaded, the Streamlit application sends it to the Document Orchestrator.
>
> The orchestrator invokes the LangGraph document-processing graph.
>
> The graph maintains a shared document state and moves through loading, classification, chunking, vector storage, extraction, merging, validation and finalization.
>
> Importantly, LangGraph is being used as the orchestration layer. I am reusing the existing document loader, classifier, chunker, extractor, merger, validator and vector-store components rather than rewriting them as LangChain components.
>
> Once the document is processed, the user can ask questions. The QA graph retrieves relevant chunks from ChromaDB, classifies the question, selects a reasoning route, and generates an answer using the retrieved document context."

### SHOW

Your architecture diagram:

```text
UPLOAD
   ↓
Streamlit
   ↓
DocumentOrchestrator
   ↓
LANGGRAPH DOCUMENT GRAPH
   ↓
Load → Classify → Chunk → Store
                         ↓
                  Extract → Merge → Validate
                         ↓
                       END


QUESTION
   ↓
LANGGRAPH QA GRAPH
   ↓
Retrieve
   ↓
Question Classification
   ↓
Route
 ┌───────────────┐
 ↓               ↓
Structured      LLM
Reasoning       Reasoning
 └───────┬───────┘
         ↓
       Answer
         +
      Evidence
```

### SHOW PROJECT FOLDERS

Highlight:

```text
agents/
    classifier.py
    extractor.py
    merger.py
    reasoning.py
    retriever.py
    validator.py
    orchestrator.py
    document_graph.py
    qa_graph.py

processing/
    document_loader.py
    chunker.py
    ocr.py

storage/
    vector_store.py
```

### SAY

> "This separation also makes the architecture easier to extend because document processing and question answering are independent workflows."

---

# 3. DOCUMENT PROCESSING GRAPH — 45–60 SEC

Before demonstrating individual files, show one processing run.

### SAY

> "Let me first show what happens internally when a document is uploaded.
>
> The document graph starts by initializing the document state and generating a document ID.
>
> The loader processes the document. Depending on the document type, native text, OCR output, tables, images and structural information can be captured.
>
> The classifier adds document-level information to the state.
>
> The chunker then creates retrieval-friendly chunks while retaining metadata.
>
> These chunks are stored in ChromaDB.
>
> The extractor processes the chunks, the merger combines the results, and the validator performs the final validation step.
>
> The graph then produces a consolidated document result containing the processed document, classification, chunks, extraction results, validation results, structural summary and timing information."

### SHOW

Terminal output.

Then show:

- Document ID
- Classification
- Chunk count
- Tables
- Images
- Structural summary
- Processing timings

---

# 4. DOCX DEMO — 60–90 SEC

## UPLOAD

`01_Synthetic_Business_Report.docx`

### SAY

> "I will start with a synthetic Word document.
>
> This document contains narrative content as well as a structured operational metrics table.
>
> The important point here is that I am not writing a special extractor specifically for this table. It enters the same generic processing and retrieval pipeline."

### SHOW

- Extracted document
- Table
- Chunks
- Structural information

### ASK

> "What regions are listed in the operational metrics table?"

Pause and show answer.

### ASK

> "Which region has the highest exception rate?"

Pause and show answer.

### SAY

> "The questions are answered from the processed document context rather than from a hard-coded answer.
>
> The QA workflow retrieves the relevant document chunks and passes the evidence to the reasoning layer."

### SHOW

QA debug information if available:

```text
Question type
Route
Retrieved contexts
Document ID
```

---

# 5. XLSX DEMO — 45–60 SEC

## UPLOAD

`02_Synthetic_Operations_Data.xlsx`

### SAY

> "Next, I am testing a spreadsheet.
>
> This demonstrates that the system is not conceptually limited to PDF documents.
>
> The goal of the architecture is to normalize different document modalities into representations that can participate in the downstream retrieval and reasoning workflow."

### ASK

> "How many records are present?"

### ASK

> "What statuses are present?"

### SAY

> "The important architectural point is that the downstream question-answering layer does not need to be rewritten for every document format.
>
> The ingestion and processing layer handles the document-specific representation, while retrieval and reasoning operate on the resulting document state and chunks."

---

# 6. IMAGE / OCR DEMO — 60–90 SEC

## UPLOAD

`04_Synthetic_Text_Image.png`

### SAY

> "Now I am testing an image document.
>
> Unlike a normal text-based document, an image does not provide a native text layer that can simply be extracted.
>
> Therefore OCR is required to convert the visual text into machine-readable content."

### SHOW

- Original image
- OCR output
- Extracted text

### ASK

> "What is the reference?"

### ASK

> "What is the status?"

### SAY

> "The OCR result becomes part of the document representation and can subsequently participate in chunking, storage, retrieval and reasoning.
>
> The original visual asset is also preserved separately from the recognized OCR text."

### OPTIONAL SHOW

OCR processing details if available.

---

# 7. VISUAL DOCUMENT DEMO — 60–90 SEC

## UPLOAD

`03_Synthetic_Floor_Plan.png`

### SAY

> "This is a synthetic floor-plan image.
>
> This demonstrates an important limitation as well as an architectural opportunity.
>
> The current system can preserve the visual document and process any extractable text through the existing pipeline.
>
> However, a question such as 'What labels are visible?' is fundamentally different from a spatial question such as 'What room is adjacent to the reception area?'
>
> Text retrieval alone is not sufficient for deeper geometric reasoning."

### ASK

> "What labels are visible?"

### SAY

> "The current system can use the extracted textual evidence where available.
>
> A dedicated visual or geometry reasoning agent would be the natural next extension for questions involving spatial relationships, coordinates or shapes."

### IMPORTANT

This is a good point to demonstrate that you understand the system's **current capability versus future capability**.

Do not claim that a visual reasoning agent already exists if it does not.

---

# 8. MULTI-PAGE PDF DEMO — 90–120 SEC

## UPLOAD

`05_Synthetic_Multi_Page_Report.pdf`

### SAY

> "Finally, I will demonstrate the PDF pipeline.
>
> PDF processing is particularly important because PDFs can contain native machine-readable text, scanned pages, tables, images and other layout information in the same document."

### SHOW

Document processing output:

```text
Page count
Native text pages
OCR pages
Table count
Image count
Chunk count
Processing time
```

### SAY

> "The loader attempts to use the available native document information where possible and uses OCR when required.
>
> This is important for performance because unnecessarily running OCR over every page can significantly increase processing time."

### ASK

> "How many columns are in the metrics table?"

### ASK

> "What was the Q3 service level?"

### SAY

> "These questions also demonstrate why retrieval and reasoning should not be treated as one single operation.
>
> A structural question such as table dimensions is best answered from structural or table metadata when that information is available.
>
> A semantic question such as the Q3 service level can be answered by retrieving the relevant table or text context and reasoning over that evidence."

### SHOW

QA graph/debug information.

Point out:

```text
Document ID
Question type
Route
Retrieved contexts
Answer
```

---

# 9. LANGGRAPH QA WORKFLOW — 45–60 SEC

This is important because it is one of the major final-architecture changes.

### SAY

> "Let me briefly show what happens when I ask a question.
>
> The question enters the QA LangGraph.
>
> First, the retriever searches for relevant chunks using the current document ID.
>
> This document ID is important because it prevents the QA workflow from accidentally mixing information from another processed document.
>
> The question is then classified.
>
> Structured or lookup-style questions can follow the structured route, while more semantic questions are sent through the LLM reasoning route.
>
> Both routes ultimately produce the answer using the available document context."

### SHOW

```text
Question
   ↓
Retriever
   ↓
Question Classification
   ↓
Route
 ┌───────────┐
 │           │
 ↓           ↓
Structured   LLM
Reasoning   Reasoning
 │           │
 └─────┬─────┘
       ↓
     Answer
```

### SAY

> "This is the agentic part of the final architecture: the workflow is stateful and can be extended with additional specialized agents and routing decisions."

---

# 10. PERFORMANCE — 45–60 SEC

### SAY

> "Performance was also an important part of development.
>
> During testing, large documents took significantly longer to process.
>
> Profiling showed that OCR, document parsing and repeated local LLM inference can become major contributors to processing time.
>
> Therefore, the architecture separates deterministic document processing from LLM-based reasoning wherever possible.
>
> Processing timings are captured for individual stages such as loading, classification, chunking, vector storage, extraction, merging and validation."

### SHOW

Terminal:

```text
Loading / OCR
Classification
Chunking
Vector Store
Extraction
Merge
Validation
Total
```

### SAY

> "This stage-level timing makes it easier to identify the actual bottleneck instead of treating the entire pipeline as a black box."

### IF YOU HAVE BEFORE/AFTER TIMINGS

Show them here.

---

# 11. ISSUES AND LESSONS — 45–60 SEC

### SAY

> "Several important lessons came from testing the system.
>
> First, OCR quality directly affects downstream language detection and extraction, so OCR output quality needs to be treated as part of the document-processing problem.
>
> Second, retrieval is not the answer to every type of question.
>
> Text questions require text evidence.
>
> Table questions benefit from table and structural evidence.
>
> Visual questions may require image or geometry reasoning.
>
> Third, the LLM should not be responsible for every deterministic operation. Where reliable structural metadata is available, it should be preferred over asking the LLM to infer the same information."

### PAUSE

Then:

> "These lessons motivated the explicit query-routing architecture in the QA graph."

---

# 12. WHY LANGGRAPH — 30–45 SEC

### SAY

> "LangGraph was introduced as an orchestration layer rather than replacing the existing processing components.
>
> This allows the project to retain the working document-processing modules while providing explicit workflow state, node-based execution and conditional routing.
>
> The same architecture can now be extended with additional nodes such as table reasoning, visual reasoning, confidence evaluation, human review or document comparison."

### SHOW

```text
Current

Document
   ↓
LangGraph
   ↓
Existing Processing Agents


Future

Document
   ↓
Router
   ├── Text Agent
   ├── Table Agent
   ├── Visual Agent
   ├── Comparison Agent
   └── Validation Agent
```

---

# 13. FINAL SUMMARY — 30–45 SEC

### SAY

> "To summarize, this project evolved from a basic document-to-LLM pipeline into a modular agentic document intelligence architecture.
>
> It supports heterogeneous document ingestion, native text extraction, OCR, document structure, tables, images, chunking, embeddings, ChromaDB retrieval, semantic reasoning and validation.
>
> The final architecture uses LangGraph for both document-processing orchestration and question-answering orchestration.
>
> Most importantly, the system is designed to be generic rather than tied to a single document schema.
>
> The architecture can be extended with visual reasoning, advanced table reasoning, document comparison, compliance agents, confidence scoring and enterprise workflow automation.
>
> Thank you."

---

# 14. FINAL SCREEN

Keep the architecture on screen for approximately 10 seconds.

```text
                AGENTIC DOCUMENT INTELLIGENCE

                         DOCUMENT
                            |
                            v
                  ┌─────────────────┐
                  │   LangGraph     │
                  │ Document Graph  │
                  └─────────────────┘
                            |
                            v
             Load → Classify → Chunk
                            |
                            v
                       ChromaDB
                            |
                            v
                  Extract → Merge
                            |
                            v
                        Validate
                            |
                            v
                     READY STATE
                            |
                            v
                       QUESTION
                            |
                            v
                  ┌─────────────────┐
                  │   LangGraph     │
                  │     QA Graph    │
                  └─────────────────┘
                            |
                            v
                       Retrieve
                            |
                            v
                     Route Question
                      /          \
                     /            \
             Structured          LLM
               Route            Reasoning
                     \            /
                      \          /
                       v        v
                         ANSWER
                            |
                            v
                  GROUNDED EVIDENCE
```

---

# 15. ONE-LINE FINAL ARCHITECTURE

If the evaluator asks:

**"Can you explain your architecture in one sentence?"**

Say:

> "It is a generic, multimodal document intelligence system where LangGraph orchestrates document ingestion, classification, chunking, storage, extraction and validation, and a separate LangGraph QA workflow performs document-scoped retrieval, question routing and evidence-grounded reasoning."

---

# 16. KEY POINTS TO EMPHASIZE DURING THE DEMO

Make sure you explicitly communicate these five points:

### 1. Generic

> "I am not hard-coding a schema for one document type."

### 2. Multimodal

> "The architecture can work with text, tables, scanned content and visual documents."

### 3. Agentic

> "LangGraph provides explicit stateful orchestration and routing."

### 4. Grounded

> "The answer is generated using retrieved evidence from the selected document."

### 5. Extensible

> "Specialized table, visual, comparison and validation agents can be added without redesigning the complete system."

---

# 17. DEMO FLOW — QUICK CHEAT SHEET

Keep this beside you while recording:

```text
INTRO
 ↓
ARCHITECTURE
 ↓
DOCUMENT GRAPH
 ↓
DOCX
 ↓
XLSX
 ↓
IMAGE/OCR
 ↓
FLOOR PLAN
 ↓
PDF
 ↓
QA GRAPH
 ↓
PERFORMANCE
 ↓
LESSONS
 ↓
LANGGRAPH
 ↓
CLOSING
```

### Documents

```text
01 → DOCX → Table + Text
02 → XLSX → Spreadsheet
04 → PNG  → OCR
03 → PNG  → Visual
05 → PDF  → Multi-page + Tables
```

### Questions

```text
DOCX
"What regions are listed?"
"Which region has the highest exception rate?"

XLSX
"How many records are present?"
"What statuses are present?"

OCR
"What is the reference?"
"What is the status?"

Floor Plan
"What labels are visible?"

PDF
"How many columns are in the metrics table?"
"What was the Q3 service level?"
```

### Architecture keywords

```text
Streamlit
    ↓
DocumentOrchestrator
    ↓
LangGraph
    ↓
Loader
Classifier
Chunker
Vector Store
Extractor
Merger
Validator
    ↓
ChromaDB
    ↓
QA LangGraph
    ↓
Retriever
Question Routing
Reasoning
    ↓
Answer + Evidence
```