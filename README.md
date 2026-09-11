Agentic Document Intelligence System
End-to-End Technical Project Report
1. Executive Summary
The Agentic Document Intelligence System is an intelligent document processing and questionanswering platform designed to process heterogeneous business documents and extract, organize,
retrieve, and reason over their contents.
Traditional document processing systems often depend on rigid templates or document-specific
extraction rules. Such approaches work well for highly standardized documents but become difficult to
maintain when document layouts, formats, languages, tables, or structures change.
This project addresses that challenge by building a generic document intelligence pipeline capable of
processing different types of documents without being tightly coupled to a particular document format.
The system combines:
Document parsing
Native text extraction
OCR
Layout understanding
Table extraction
Language detection
Intelligent chunking
Document classification
Vector storage
Semantic retrieval
Question classification
LLM-based reasoning
Validation
Agentic orchestration
LangGraph workflow management
Streamlit-based user interface
The architecture separates the document processing pipeline from the question-answering pipeline. A
document first passes through ingestion, classification, chunking, storage, extraction, merging, and
validation. Once processed, users can ask natural-language questions about the document. The system
retrieves the most relevant information and generates an answer using the available document
evidence.
The system is designed to be generic, modular, explainable, extensible, and suitable for enterprise
document intelligence use cases.
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
1
2. Problem Statement
Organizations generate large volumes of documents in different formats, including:
PDF reports
Scanned PDFs
Text-based PDFs
Invoices
Financial statements
Operational reports
Business documents
Tables
Forms
Presentations
Word documents
Image-based documents
The information contained in these documents is often difficult to access programmatically.
A conventional keyword search system has several limitations:
It cannot reliably understand document meaning.
It struggles with scanned documents.
It does not understand document structure.
It performs poorly on tables and spatial relationships.
It cannot easily answer semantic questions.
It may retrieve information from the wrong document.
It does not provide an intelligent reasoning layer.
Therefore, the objective of this project is to build a generic document intelligence system that
converts unstructured and semi-structured documents into searchable, structured, and
queryable knowledge.
3. Project Objectives
The major objectives are:
3.1 Generic document processing
Build a system that can process documents without requiring document-specific extraction rules.
3.2 Text extraction
Extract text from digitally generated documents while preserving document structure wherever
possible.
•
•
•
•
•
•
•
•
•
•
•
•
1.
2.
3.
4.
5.
6.
7.
2
3.3 OCR support
Process scanned and image-based documents using OCR when native text is unavailable or insufficient.
3.4 Layout understanding
Capture document-level structures such as:
Pages
Paragraphs
Tables
Images
Layout objects
Spatial information
3.5 Intelligent chunking
Divide documents into meaningful chunks suitable for retrieval and downstream reasoning.
3.6 Vector-based retrieval
Store document chunks in a vector database and retrieve semantically relevant content.
3.7 Natural-language question answering
Allow users to ask questions using normal language rather than requiring SQL, keywords, or predefined
queries.
3.8 Agentic reasoning
Introduce decision-making components that determine how a question should be processed and which
reasoning path should be used.
3.9 Workflow orchestration
Use LangGraph to explicitly model the document-processing and question-answering workflows as
graphs.
3.10 Explainability
Provide evidence and debugging information so users can understand where an answer came from.
4. High-Level Architecture
The system consists of two major workflows.
•
•
•
•
•
•
3
4.1 Document Processing Workflow
 User Uploads Document
 |
 v
 Document Orchestrator
 |
 v
 LangGraph
 |
 +----------------+----------------+
 | | |
 v v v
 Document Document Document
 Loader Classifier Chunker
 | | |
 +----------------+----------------+
 |
 v
 Vector Store
 ChromaDB
 |
 v
 Extraction
 |
 v
 Merger
 |
 v
 Validator
 |
 v
 Processed Document
5. Question Answering Architecture
After document processing, users can ask questions about the uploaded document.
 User Question
 |
 v
 QA Graph
 |
 v
 Retriever
 |
4
 v
 Relevant Contexts
 |
 v
 Question Classification
 |
 +----------+----------+
 | |
 v v
 Structured / Lookup Semantic
 Route Route
 | |
 +----------+----------+
 |
 v
 Reasoning Agent
 |
 v
 Answer
 |
 v
 Evidence / Debug
This separation allows the document ingestion process and question-answering process to evolve
independently.
6. Technology Stack
6.1 Programming Language
Python
Python is used as the primary development language because of its mature ecosystem for:
Machine learning
NLP
OCR
Document processing
Vector databases
LLM integration
Web applications
•
•
•
•
•
•
•
5
7. Core Technologies
7.1 Streamlit
Streamlit is used to create the interactive application interface.
The application allows users to:
Upload documents
Process documents
View extracted information
View document structure
Ask questions
View answers
Inspect retrieved evidence
View debugging information
Inspect processing metrics
7.2 Docling
Docling is used for document understanding and structured document parsing.
It helps identify document components such as:
Text
Tables
Images
Layout elements
Page structure
This is particularly useful because document intelligence requires more than simply extracting raw text.
7.3 PyMuPDF
PyMuPDF is used for PDF processing.
The Python module is imported as:
import fitz
It provides functionality for:
Opening PDFs
Reading pages
Extracting text
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
6
Working with PDF structure
Rendering PDF pages when required
7.4 Tesseract OCR
Tesseract is used as an OCR engine for scanned or image-based content.
OCR becomes important when a PDF does not contain a usable native text layer.
The system therefore supports both:
Native PDF Text
 +
OCR Text
This hybrid approach improves robustness across different document types.
7.5 OpenCV
OpenCV is used for image processing and OCR preprocessing.
Potential operations include:
Image conversion
Thresholding
Noise reduction
Image preprocessing
Page/image handling
7.6 Pillow
Pillow is used for image manipulation and conversion.
7.7 ChromaDB
ChromaDB is used as the vector store.
Document chunks are converted into embeddings and stored so that semantically related content can
later be retrieved.
Conceptually:
•
•
•
•
•
•
•
7
Document
 |
 v
Chunks
 |
 v
Embeddings
 |
 v
ChromaDB
 |
 v
Semantic Retrieval
7.8 Sentence Transformers
Sentence Transformers can be used to generate semantic embeddings for document chunks and
queries.
This allows retrieval based on meaning rather than exact keyword matching.
For example, a document may contain:
"Regional operational performance"
while a user asks:
"Which regions processed the most units?"
Semantic retrieval can identify the relevant section even when the wording differs.
7.9 Ollama / LLM Layer
Ollama is used to provide local LLM capabilities.
The LLM is responsible for reasoning over retrieved document context and generating natural-language
answers.
The system does not rely solely on the LLM's internal knowledge. Instead, the intended approach is:
Question
 |
Retrieval
 |
Relevant Document Context
8
 |
LLM
 |
Evidence-grounded Answer
This reduces hallucination and makes the answer more document-specific.
8. Agent Architecture
The project follows a modular agent-oriented design.
The major components include:
agents/
├── classifier.py
├── extractor.py
├── llm.py
├── merger.py
├── orchestrator.py
├── reasoning.py
├── retriever.py
├── validator.py
├── document_graph.py
└── qa_graph.py
Each component has a specific responsibility.
9. Document Classifier
The document classifier determines the general characteristics of an uploaded document.
The purpose is to provide downstream components with document-level information that can influence
processing.
Examples of possible document characteristics include:
PDF
Report
Financial document
Structured document
Scanned document
Text-heavy document
Table-heavy document
•
•
•
•
•
•
•
9
The classifier is deliberately separated from the loader so that classification logic can evolve
independently.
10. Document Loader
The document loader is responsible for loading the source document.
It acts as the entry point into the processing pipeline.
For PDF documents, PyMuPDF and document parsing tools are used.
The loader attempts to capture information such as:
Page count
Native text
Images
Tables
Layout objects
OCR requirements
A major design principle is:
Do not assume that every document has a usable native text layer.
Therefore, OCR can be used when necessary.
11. Hybrid Text Extraction
One of the important design aspects of the project is the use of both native extraction and OCR.
Native extraction
For digitally generated PDFs:
PDF
 |
Native text layer
 |
Text extraction
OCR extraction
For scanned PDFs:
•
•
•
•
•
•
10
PDF
 |
Rendered page/image
 |
OCR
 |
Recognized text
Combined strategy
 PDF
 |
 +--------+--------+
 | |
 Native Text Image/Scan
 | |
 | OCR
 | |
 +--------+--------+
 |
 Unified Content
This makes the system more generic.
12. Document Chunking
Large documents cannot always be sent directly to an LLM.
Therefore, the document is divided into chunks.
A chunk may contain:
Text
Page information
Table information
Spatial information
Metadata
The objective is not simply to split the document after a fixed number of characters.
Meaningful chunking should preserve document context wherever possible.
For example:
•
•
•
•
•
11
Document
 |
Page
 |
Section
 |
Paragraph / Table
 |
Chunk
This makes downstream retrieval more accurate.
13. Vector Store
After chunking, the chunks are stored in the vector database.
Each chunk can conceptually be represented as:
Chunk
 ├── Text
 ├── Embedding
 ├── Document ID
 ├── Page
 └── Metadata
The Document ID is particularly important.
It ensures that when the user asks a question about a particular uploaded document, the system can
retrieve information associated with that document rather than mixing content from unrelated
documents.
14. Document ID Isolation
A document-level identifier is generated during processing.
Conceptually:
document_id = str(uuid.uuid4())
This ID is associated with:
Processed document
Chunks
•
•
12
Vector-store records
Retrieval requests
QA requests
Debug information
This provides an important isolation mechanism.
For example:
Document A
 |
document_id = A
 |
Chunks A
Document B
 |
document_id = B
 |
Chunks B
A question for Document A should only retrieve chunks associated with Document A.
15. Extraction Agent
The extraction agent processes document chunks and extracts useful information.
It is implemented as a generic extractor rather than being hardcoded for a specific document type.
This is important for a general-purpose document intelligence system.
The extraction layer can deal with:
Text
Tables
Structured information
Page-level information
Metadata
16. Result Merger
Individual extraction results may be generated at chunk or page level.
The merger combines these results into a coherent document-level representation.
•
•
•
•
•
•
•
•
•
13
Conceptually:
Chunk 1 Extraction
Chunk 2 Extraction
Chunk 3 Extraction
 |
 v
 Merger
 |
 v
Document-level result
This separation keeps extraction and aggregation logically independent.
17. Validation Layer
The validation agent provides a final quality-control stage.
Validation can identify:
Missing information
Inconsistent extracted results
Invalid totals
Structural inconsistencies
Extraction issues
For example, if a table contains totals, validation can check whether extracted values are internally
consistent.
The purpose is not merely to produce an answer but to increase confidence in the processing pipeline.
18. LangGraph Integration
LangGraph was introduced as the workflow orchestration layer.
Before LangGraph, the processing flow could be represented as a traditional sequential Python pipeline.
The LangGraph implementation makes the workflow explicit.
The graph contains:
START
 |
Initialize
•
•
•
•
•
14
 |
Load
 |
Classify
 |
Chunk
 |
Store
 |
Extract
 |
Merge
 |
Validate
 |
Finalize
 |
END
Each step is implemented as a graph node.
19. Document Graph State
The document graph maintains a shared state.
Conceptually:
DocumentState
contains fields such as:
file_path
document_id
document
classification
chunks
storage
extracted
merged
validation
structural_summary
timing
error
This state allows different nodes to communicate without tightly coupling them.
For example:
•
•
•
•
•
•
•
•
•
•
•
•
15
Loader
 |
 v
document
 |
Classifier
 |
 v
classification
 |
Chunker
 |
 v
chunks
Each node receives the current state and contributes additional information.
20. Why LangGraph Was Used
LangGraph provides several benefits.
Explicit workflow
The complete processing flow becomes visible and easier to understand.
State management
All workflow stages can share a structured state.
Conditional routing
Future versions can branch dynamically.
For example:
Document
 |
Is scanned?
 /
Yes No
 | |
OCR Native
 | |
 +----+----+
 |
Continue
16
Error handling
A graph makes it easier to introduce recovery and fallback nodes.
Extensibility
Additional agents can be added without rewriting the entire application.
21. QA Graph
A separate LangGraph workflow is used for question answering.
The QA flow is:
START
 |
Retrieve Context
 |
Classify Question
 |
Route
 /
Structured LLM
Reasoning Reasoning
 \ /
 Answer
 |
 END
This separates question retrieval from reasoning.
22. Question Classification
The QA system determines the type of question.
Examples:
Structured questions
What regions are listed?
How many units were processed?
What is the total?
Which region has the highest value?
17
Semantic questions
Why did operational performance decline?
What are the major risks discussed?
Summarize the management concerns.
This classification allows the system to evolve toward specialized reasoning strategies.
23. Retrieval
When a user asks a question, the retriever searches the vector store for relevant chunks.
For example:
Question:
"What regions are listed in the operational metrics table?"
 |
 v
Semantic Retrieval
 |
 v
Relevant table chunk
 |
 v
Reasoning
 |
 v
Answer
The system can retrieve the top-K relevant chunks.
The value of K can be controlled through configuration.
24. Retrieval Safety
One important implementation detail is document-level filtering.
The retrieval request contains:
18
document_id
This ensures that retrieval is scoped to the currently selected document.
An additional safety filtering step can be applied at the application layer before evidence is shown to the
user.
This provides defense against accidental cross-document retrieval.
25. Reasoning Agent
The reasoning agent receives:
User Question
+
Retrieved Context
and generates an answer.
The intended reasoning pattern is:
Question
 +
Evidence
 |
 v
Reasoning
 |
 v
Answer
The system should prefer retrieved evidence over unsupported model knowledge.
26. Evidence-Based Answering
The application exposes the evidence used for answering.
This is important because users should be able to inspect why a particular answer was generated.
A typical interaction becomes:
19
Question
 |
Retrieved evidence
 |
Reasoning
 |
Answer
rather than:
Question
 |
LLM guess
 |
Answer
This is a core principle of retrieval-augmented document intelligence.
27. Streamlit Application
The Streamlit application acts as the user-facing interface.
The major sections include:
Document upload
Users can upload supported files.
Processing
The uploaded document is passed to the document orchestrator.
Document information
The application displays:
Document ID
Classification
Processing time
Chunk count
Document structure
Extracted results
Users can inspect extracted information.
•
•
•
•
•
20
Question answering
Users can ask questions about the uploaded document.
Evidence
Relevant retrieved contexts can be displayed.
Debug information
Technical information can be inspected for troubleshooting.
28. Processing Metrics
The system records timing information for different processing stages.
Example stages include:
Loading / OCR
Classification
Chunking
Vector Store
Extraction
Merge
Validation
Total
This makes performance bottlenecks easier to identify.
For large documents, this is particularly useful because OCR and document parsing can be significantly
more expensive than simple text processing.
29. Structural Summary
The system produces a structural summary containing information such as:
Page count
Total text characters
Image count
Table count
Layout object count
Pages with native text
21
Pages with OCR
Table summaries
This provides a compact representation of the document structure.
It is useful for both debugging and downstream document analytics.
30. Error Handling
The architecture includes error information in the graph state.
Conceptually:
Node
 |
 +---- Success ----> Next Node
 |
 +---- Failure ----> Error State
This is important because document processing involves multiple external libraries and potentially
unreliable inputs.
Potential failures include:
Corrupt PDFs
Unsupported file formats
OCR failures
Missing dependencies
LLM failures
Vector-store failures
Invalid document structures
31. Generic Design Philosophy
A major objective of the project is to avoid document-specific logic.
For example, the system should not contain logic such as:
If document == Aadhaar:
 extract father name
or:
•
•
•
•
•
•
•
22
If document == invoice:
 read fixed coordinate
Instead, it should work from generic document characteristics:
Document
 |
Layout
 |
Text
 |
Tables
 |
Chunks
 |
Retrieval
 |
Reasoning
This makes the architecture reusable across multiple document categories.
32. Supported Document Intelligence
Capabilities
The architecture can support use cases such as:
Financial reports
Questions:
What was revenue?
Which segment performed best?
What were the major risks?
Operational reports
Questions:
Which regions are listed?
Which region has the highest exception rate?
What is the average cycle time?
•
•
•
•
•
•
23
Invoices
Questions:
What is the invoice amount?
What is the invoice date?
Who is the supplier?
Business reports
Questions:
What are the key findings?
What recommendations are provided?
What are the major operational issues?
Tables
Questions:
What columns exist?
What are the values for a particular row?
Which row has the maximum value?
33. Key Design Principles
The project follows several architectural principles.
Modularity
Each component performs one primary responsibility.
Reusability
Existing document-processing modules can be reused by LangGraph nodes.
Separation of concerns
Document ingestion, extraction, storage, retrieval, reasoning, and UI are separate layers.
Evidence grounding
Answers are based on retrieved document context.
Document isolation
Document IDs prevent unintended cross-document retrieval.
•
•
•
•
•
•
•
•
•
24
Extensibility
New agents and graph branches can be introduced without redesigning the complete system.
Observability
Timing, structural information, evidence, and debug information are exposed.
34. End-to-End Execution Flow
The complete workflow is:
1. User uploads document
 |
 v
2. Streamlit saves document
 |
 v
3. DocumentOrchestrator invoked
 |
 v
4. LangGraph Document Workflow
 |
 v
5. Document initialization
 |
 v
6. Document loading
 |
 v
7. Native text / OCR processing
 |
 v
8. Document classification
 |
 v
9. Document chunking
 |
 v
10. Chunks stored in ChromaDB
 |
 v
11. Extraction agent
 |
 v
12. Result merger
 |
25
 v
13. Validation
 |
 v
14. Final document result
 |
 v
15. User asks question
 |
 v
16. QA LangGraph
 |
 v
17. Retrieve relevant chunks
 |
 v
18. Classify question
 |
 v
19. Select reasoning route
 |
 v
20. Reasoning agent
 |
 v
21. Evidence-grounded answer
 |
 v
22. Display answer + evidence
35. Example Interaction
Consider a document containing an operational metrics table:
Region Units Processed Avg. Cycle Time Exception Rate
North 12,500 4.2 2.1%
South 15,200 3.8 1.8%
East 9,800 4.6 2.7%
The user asks:
What regions are listed in the operational metrics table?
The system performs:
26
Question
 |
Retrieve relevant table chunk
 |
Question classified as structured/lookup
 |
Reasoning over retrieved evidence
 |
Answer:
North, South and East
The answer is therefore grounded in the actual document rather than generated from general model
knowledge.
36. Why a Graph-Based Architecture Is Better
Than a Simple Pipeline
A traditional implementation could simply call:
load()
classify()
chunk()
store()
extract()
merge()
validate()
However, this becomes increasingly difficult to manage when the system needs:
Conditional branches
Retry mechanisms
Multiple agents
Human review
Dynamic routing
Specialized reasoning
Error recovery
Observability
LangGraph provides a structured framework for these requirements.
For example, future routing can be:
 Document
 |
 Document Classifier
•
•
•
•
•
•
•
•
27
 |
 +-----------+-----------+
 | |
 Scanned Digital
 | |
 OCR Native Parser
 | |
 +-----------+-----------+
 |
 Chunking
 |
 Retrieval
This makes the architecture naturally extensible.
37. Performance Considerations
Large documents can be expensive to process because of:
OCR
PDF parsing
Layout analysis
Embedding generation
Vector storage
LLM inference
Performance optimization opportunities include:
Avoid unnecessary OCR
Use native PDF text whenever reliable text is already available.
Cache processed documents
The application checks whether an uploaded document has already been processed to avoid
unnecessary repeated processing.
Parallel processing
Independent pages or chunks can potentially be processed in parallel.
Batch embeddings
Multiple chunks can be embedded in batches.
Retrieval limits
The number of retrieved chunks should be controlled through configurable top-K parameters.
•
•
•
•
•
•
28
Local LLM inference
Using Ollama avoids requiring every request to be sent to a remote API.
38. Security and Data Privacy Considerations
The architecture can be particularly useful for sensitive enterprise documents because processing can
be performed locally.
Potential benefits include:
Local OCR
Local document processing
Local vector storage
Local LLM inference through Ollama
This can reduce unnecessary exposure of sensitive document contents to external services.
For production deployment, additional controls would still be required, including:
Authentication
Authorization
Encryption
Audit logs
Secure file storage
Access control
Data retention policies
39. Limitations
The current system has several limitations.
OCR accuracy
OCR accuracy depends on:
Image quality
Font
Scan quality
Language
Page orientation
Noise
Table understanding
Complex tables containing merged cells, nested headers, or irregular layouts remain challenging.
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
29
Semantic retrieval
Retrieval quality depends on embedding quality and chunk design.
LLM hallucination
Even with retrieval, an LLM can generate unsupported information. Evidence-grounding and validation
therefore remain important.
Question classification
The current question routing logic uses deterministic heuristics and can be improved using a dedicated
classifier or LLM-based router.
Large document latency
OCR and layout processing can become expensive for very large documents.
40. Future Enhancements
The architecture provides a strong foundation for future development.
40.1 Advanced Agent Routing
Introduce specialized agents:
Question
 |
Router
 |
+---------+---------+----------+
| | | |
Table Numeric Semantic Summary
Agent Agent Agent Agent
40.2 True Structured Table Reasoning
Instead of relying entirely on semantic retrieval, tables can be converted into structured
representations.
For example:
30
DataFrame
 |
SQL / Python reasoning
 |
Exact answer
This would improve accuracy for questions involving:
Sum
Average
Maximum
Minimum
Sorting
Filtering
40.3 Human-in-the-Loop Validation
Low-confidence results could be routed to human review.
Extraction
 |
Confidence
 /
High Low
 | |
Continue Human Review
40.4 Confidence Scoring
Each answer could contain:
Answer
Confidence
Evidence
Source pages
40.5 Citation-Aware Answers
Answers could explicitly cite:
•
•
•
•
•
•
31
Page 5
Table 2
Section: Operational Metrics
This would improve enterprise usability.
40.6 Multi-Document Question Answering
The system can be extended from:
One document → questions
to:
Multiple documents
 |
Unified retrieval
 |
Cross-document reasoning
This enables questions such as:
Compare the operational performance reported in Q1 and Q2.
40.7 Multilingual Document Intelligence
OCR and language detection can be expanded to support:
English
Hindi
Telugu
Tamil
Kannada
Other regional languages
This would make the system more applicable to Indian enterprise documents.
40.8 Production Deployment
The current Streamlit interface can eventually be separated into:
•
•
•
•
•
•
32
Frontend
 |
REST API
 |
Agent Orchestrator
 |
Document Processing
 |
Vector Database
 |
LLM
FastAPI can be used as the backend API layer.
41. Testing Strategy
Testing should cover multiple layers.
Unit testing
Test:
Loader
Chunker
Classifier
Retriever
Validator
Individual graph nodes
Integration testing
Test:
Upload
→ Process
→ Store
→ Retrieve
→ Answer
Document diversity testing
Use:
Text PDFs
Scanned PDFs
•
•
•
•
•
•
•
•
33
Tables
Images
Multi-page documents
Different layouts
Retrieval testing
Evaluate:
Recall@K
Precision@K
Retrieval relevance
Answer evaluation
Evaluate:
Correctness
Groundedness
Completeness
Hallucination rate
42. Observability
The system provides useful debugging information including:
Document ID
File path
Retrieval K
Number of chunks
Question type
QA route
Number of retrieved contexts
Processing timing
Structural document information
This is important during development because a wrong final answer may originate from different
stages:
Bad answer
 |
 +-- Bad OCR
 |
 +-- Bad extraction
 |
 +-- Bad chunking
 |
 +-- Bad retrieval
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
34
 |
 +-- Bad reasoning
The modular architecture makes these failures easier to diagnose.
43. Project Folder Structure
The final architecture can be represented as:
agentic-document-intelligence/
│
├── agents/
│ ├── __init__.py
│ ├── classifier.py
│ ├── extractor.py
│ ├── llm.py
│ ├── merger.py
│ ├── orchestrator.py
│ ├── reasoning.py
│ ├── retriever.py
│ ├── validator.py
│ ├── document_graph.py
│ └── qa_graph.py
│
├── processing/
│ ├── __init__.py
│ ├── chunker.py
│ ├── document_loader.py
│ ├── language.py
│ └── ocr.py
│
├── schemas/
│ ├── __init__.py
│ └── document.py
│
├── storage/
│ ├── __init__.py
│ ├── metadata.py
│ └── vector_store.py
│
├── tests/
│
├── app.py
├── config.py
├── requirements.txt
└── test_document.py
35
44. Dependency Stack
The major dependencies include:
Python
Streamlit
Docling
PyMuPDF
Tesseract
OpenCV
Pillow
ChromaDB
Sentence Transformers
Ollama
LangGraph
LangChain Core
NumPy
Pandas
OpenPyXL
Python-docx
Python-pptx
Requests
LangGraph is specifically used for workflow orchestration, while the existing processing components
remain responsible for their original tasks.
45. Important Architectural Decision
A key implementation decision was not to rewrite the entire application using LangChain/
LangGraph abstractions.
Instead, LangGraph is introduced as an orchestration layer around the existing components.
For example:
Existing Loader
Existing Classifier
Existing Chunker
Existing Extractor
Existing Merger
Existing Validator
Existing Retriever
Existing Reasoning Agent
 |
36
 v
 LangGraph
This approach provides two major benefits:
Existing working code is preserved.
The application gains explicit agentic workflow orchestration.
This is preferable to rewriting stable components unnecessarily.
46. Difference Between Traditional Pipeline and
Agentic Workflow
Traditional pipeline
Input
 ↓
Load
 ↓
Process
 ↓
Extract
 ↓
Output
The sequence is mostly fixed.
Agentic workflow
Input
 ↓
State
 ↓
Decision
 ↓
Specialized Node
 ↓
Decision
 ↓
Specialized Node
 ↓
Validation
 ↓
Output
1.
2.
37
The second architecture allows the system to make processing decisions based on document
characteristics and question characteristics.
47. Business Value
A document intelligence platform can significantly reduce manual effort involved in reviewing large
volumes of documents.
Potential business applications include:
Banking
Credit documents
Financial statements
Customer documents
Risk reports
Compliance documents
Insurance
Claims documents
Policy documents
Underwriting reports
Healthcare
Medical reports
Insurance documents
Administrative records
Legal
Contracts
Agreements
Case documents
Manufacturing
Operational reports
Quality reports
Inspection documents
Enterprise operations
Management reports
KPI reports
Business reviews
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
•
38
48. Key Innovations
The main innovative aspects of the project are:
1. Generic document processing
The system is not designed around one fixed document template.
2. Hybrid OCR and native extraction
The system can process both digital and scanned documents.
3. Layout-aware document understanding
Tables and document structures are considered in addition to plain text.
4. Vector-based semantic retrieval
Questions can be answered based on meaning rather than only keyword matching.
5. Agentic reasoning
Different processing and reasoning components are represented as independent agents/nodes.
6. LangGraph orchestration
The workflow is represented explicitly as a stateful graph.
7. Evidence-based QA
The system exposes retrieved context to improve transparency.
8. Document-level isolation
Document IDs prevent accidental cross-document retrieval.
49. Overall System Flow
The entire solution can be summarized as:
 USER
 |
 v
 Streamlit UI
 |
 v
 Document Orchestrator
39
 |
 v
 DOCUMENT GRAPH
 |
 +----------------+----------------+
 | | |
 v v v
 Load Classify Chunk
 | | |
 +----------------+----------------+
 |
 v
 ChromaDB
 |
 v
 Extract
 |
 v
 Merge
 |
 v
 Validate
 |
 v
 Ready Document
 |
 |
 USER QUESTION
 |
 v
 QA GRAPH
 |
 v
 Retrieve
 |
 v
 Question Router
 /
 /
 Structured Semantic
 Route Route
 \ /
 \ /
 Reasoning
 |
 v
 ANSWER
 |
 +--------+--------+
 | |
40
 v v
 Evidence Debug Info
50. Conclusion
The Agentic Document Intelligence System provides an end-to-end framework for converting
heterogeneous documents into an intelligent, searchable, and question-answerable knowledge source.
The project combines document parsing, OCR, layout understanding, chunking, vector retrieval, LLM
reasoning, validation, and graph-based orchestration into a single modular architecture.
A key strength of the solution is that it does not depend on document-specific hardcoded extraction
rules. Instead, it uses generic processing components that can be extended to different document types
and business domains.
The introduction of LangGraph further improves the architecture by converting the processing pipeline
into an explicit stateful workflow. This provides a foundation for future capabilities such as dynamic
routing, specialized agents, human-in-the-loop validation, retries, confidence scoring, structured table
reasoning, and multi-document reasoning.
The resulting architecture is therefore not simply an OCR or document extraction application. It is a
foundation for a broader agentic document intelligence platform capable of understanding
documents, retrieving relevant information, reasoning over evidence, and providing users with
transparent answers.
51. Final Project Summary
Component Purpose
Streamlit User interface
Docling Document understanding and layout processing
PyMuPDF PDF processing and native text extraction
Tesseract OCR
OpenCV Image preprocessing
Pillow Image handling
Chunker Creates retrieval-friendly document chunks
Classifier Identifies document characteristics
Extractor Extracts information from document chunks
ChromaDB Vector storage and semantic retrieval
41
Component Purpose
Sentence Transformers Embeddings
Retriever Finds relevant document context
Reasoning Agent Generates evidence-grounded answers
Validator Performs quality checks
Merger Combines extraction results
Ollama Local LLM inference
LangGraph Agent/workflow orchestration
Streamlit Interactive application and visualization
52. Final Outcome
The completed system provides the following end-to-end capability:
Upload document → Understand document → Extract text/OCR → Identify structure → Classify →
Chunk → Embed → Store → Retrieve → Reason → Validate → Answer user questions with
evidence.
The architecture is modular and can be extended toward a production-grade enterprise document
intelligence platform.
42