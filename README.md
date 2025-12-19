# Mini RAG-Powered Assistant

## Project Objective
The goal of this project is to design and implement a basic Retrieval-Augmented Generation (RAG) assistant that can answer user questions using a custom document corpusThe problem statement focuses on building an AI assistant that can accurately answer user questions from large amounts of unstructured data such as PDFs and web content. Traditional keyword search is inefficient, and using a pure language model can lead to hallucinated or unreliable answers.

By this project we as a team demonstrates our hands-on understanding of:

- Generative AI concepts  
- Retrieval-Augmented Generation (RAG)  
- Vector databases and semantic search  
- LLM integration  
- AI-assisted development tools (GitHub Copilot)
- Our creative ideas
- End to end Pipeline
- Deployment

---

## 🚀 Key Features & Architectural Highlights

This project implements a state-of-the-art RAG pipeline with 9 distinct layers of optimization to ensure accuracy, speed, and reliability.

### 1. Hybrid Search (Ensemble Retrieval)
**The Problem:** Standard vector search misses exact keyword matches (like specific error codes or acronyms).
**Our Solution:** We combine **Dense Vector Index** (OpenAI Embeddings) with **Sparse Keyword Index** (BM25). The system weights results from both, capturing both conceptual meaning and exact terminology.

### 2. Advanced Re-Ranking (Cross-Encoder)
**The Problem:** The "Lost in the Middle" phenomenon where relevant context is buried in the retrieved list.
**Our Solution:** We retrieve a broad set of documents (Top 25) and pass them through a **Cross-Encoder Model**. This re-scores every document-query pair for relevance, ensuring only the highest-quality chunks reach the LLM.

### 3. Graph-Enhanced Context (GraphRAG)
**The Problem:** Flat text chunks lose relationships between entities across different documents.
**Our Solution:** We extract entities (People, Places, Concepts) and build a lightweight relationship map. Queries retrieve not just text chunks, but also connected "neighbor" nodes, providing deep contextual awareness.

### 4. Query Expansion & Decomposition
**The Problem:** Users often ask vague or complex multi-part questions.
**Our Solution:** The system uses an LLM call to break down complex queries into sub-questions and generates synonyms (hypothetical document embeddings) to broaden the search scope and improve retrieval recall.

### 5. Hallucination Guardrails & Citations
**The Problem:** Models making up facts.
**Our Solution:**
* **Strict Adherence:** The prompt forces the model to say "I don't know" if the context is missing.
* **Citations:** Every response explicitly links to the source file and page number for full auditability.

### 6. Metadata Filtering (Self-Querying)
**The Problem:** Searching purely by text when the user actually wants to filter (e.g., "Show me reports from 2023").
**Our Solution:** We use a Self-Querying Retriever that extracts metadata filters (Date, Author, File Type) from the user's natural language and applies them *before* performing the vector search.

### 7. Intelligent Chunking Strategy
**The Problem:** Fixed-size chunking often cuts sentences in half or separates headers from their content.
**Our Solution:** We use a **Recursive Character Text Splitter** with semantic awareness. It respects document structure (paragraphs, headers) to keep related text together, preserving semantic integrity.

### 8. Multi-Modal Ingestion Support
**The Problem:** Information is often trapped in images or tables within PDFs.
**Our Solution:** The pipeline supports OCR (Optical Character Recognition) to extract text from images and specifically parses tables to preserve their row/column structure before embedding.

### 9. Conversation Memory (Session State)
**The Problem:** Users cannot ask follow-up questions in standard RAG scripts.
**Our Solution:** We maintain a chat history buffer. The system rephrases follow-up questions (e.g., "What about the second point?") into standalone queries using the previous context, ensuring a smooth conversational flow.

---

## System Architecture

The system follows a standard **RAG pipeline**:

1. Document ingestion and preprocessing  
2. Text chunking with overlap  
3. Embedding generation  
4. Vector storage using FAISS  
5. Query embedding and similarity search  
6. Context-aware response generation using an LLM  

### High-Level Flow

User Query → Retriever → Vector Database → Relevant Context → LLM → Final Answer


---

## Technology Stack

- **Programming Language:** Python  
- **LLM:** OpenAI / HuggingFace (configurable)  
- **Embedding Model:** OpenAI / HuggingFace embeddings  
- **Vector Store:** FAISS  
- **Framework:** LangChain  
- **Version Control:** GitHub  
- **AI Coding Assistant:** GitHub Copilot  

---

## Role of GitHub Copilot in This Project

GitHub Copilot was used strictly as an **AI-assisted development tool** to improve productivity and reduce boilerplate coding.

### How GitHub Copilot Was Used

- Suggested boilerplate code for document loading and preprocessing  
- Assisted with repetitive coding tasks (embeddings, FAISS setup)  
- Provided inline suggestions while working with LangChain APIs  
- Helped speed up debugging and syntax corrections  

### Developer Control

All **architectural decisions, system design, and logic flow** were implemented manually by the developer.  
GitHub Copilot acted only as a **supportive coding assistant**, not a replacement for reasoning or design.

---

## Project Workflow

1. Select a custom document corpus (PDFs / articles)  
2. Split documents into smaller overlapping chunks  
3. Generate vector embeddings for each chunk  
4. Store embeddings in a FAISS vector database  
5. Accept user queries via CLI or UI  
6. Retrieve top-k relevant chunks using similarity search  
7. Generate grounded answers using the LLM  

---

## Setup Instructions

To run the project locally:

1. Clone the repository  
2. Create and activate a Python virtual environment  
3. Install required dependencies  
4. Configure API keys using environment variables  
5. Run document ingestion  
6. Start querying the assistant  

*(Detailed steps are available in the project documentation.)*

---

## Key Learnings

- Practical understanding of Retrieval-Augmented Generation  
- Importance of grounding LLM responses with retrieved context  
- Efficient use of vector databases for semantic search  
- Responsible usage of AI-assisted coding tools  
- End-to-end GenAI system integration  

---

## Challenges and Solutions

### Selecting Optimal Chunk Size
**Challenge:** Preserving semantic context  
**Solution:** Used overlapping chunks to maintain continuity  

### Avoiding Hallucinations
**Challenge:** Unverified LLM outputs  
**Solution:** Restricted response generation strictly to retrieved document context  

---

## Future Enhancements

- Web-based frontend using Streamlit or Node.js  
- Cloud deployment (AWS / Azure)  
- Improved evaluation metrics for retrieval accuracy  
- Support for additional document formats  

---

## Repository Usage

The repository is maintained using GitHub to ensure:

- Code traceability  
- Collaboration readiness  
- Reproducibility  

