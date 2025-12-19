# Mini RAG-Powered Assistant

## Project Objective
The goal of this project is to design and implement a basic Retrieval-Augmented Generation (RAG) assistant that can answer user questions using a custom document corpus. The problem statement focuses on building an AI assistant that can accurately answer user questions from large amounts of unstructured data such as PDFs and web content. Traditional keyword search is inefficient, and using a pure language model can lead to hallucinated or unreliable answers.

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

## ⚡ Key Features & Architectural Highlights

This project implements a state-of-the-art RAG pipeline optimized for real-world interaction, transparency, and multi-format intelligence.

### 1. Multilingual Response (Cross-Lingual Support)
* **The Problem:** Knowledge bases are often in English, but users may speak Hindi, Spanish, or French.
* **Our Solution:** The system uses a **Cross-Lingual Chain**. It accepts queries in any language, translates them to English for retrieval against the database, and generates the final response back in the user's native language, breaking language barriers.

### 2. Citation & Transparency Work
* **The Problem:** Users do not trust "black box" AI models that hallucinate facts.
* **Our Solution:** We enforce **Strict Evidence Binding**. Every response includes an expandable "Source" tab that links directly to the specific document name and page number used, ensuring 100% auditability and trust.

### 3. YouTube Context Integration
* **The Problem:** Valuable information is often locked inside video tutorials or recorded meetings, inaccessible to text search.
* **Our Solution:** The pipeline integrates `YouTubeAudioLoader` and OpenAI Whisper. It downloads audio, transcribes it into text, and indexes the transcript with timestamps, allowing users to "search" inside videos.

### 4. Query Expansion Layer
* **The Problem:** Users often ask vague or overly simple questions (e.g., "slow wifi") that miss technical keywords.
* **Our Solution:** We use an intermediate LLM step to **decompose and expand** the query. The system generates synonyms and sub-questions (Hypothetical Document Embeddings) to broaden the search scope and drastically improve retrieval recall.

### 5. Multi-Modal Retrieval (Images & Tables)
* **The Problem:** Critical data is often trapped in charts, infographics, or scanned tables within PDFs.
* **Our Solution:** The ingestion pipeline utilizes **OCR (Optical Character Recognition)** and multi-modal models to extract text from images and structure tables into JSON before embedding, ensuring no data is left behind.

### 6. Intelligent Summarization
* **The Problem:** Users don't always have time to read detailed answers; sometimes they need a "TL;DR" of a 50-page report.
* **Our Solution:** We implemented a **Map-Reduce Summarization Chain**. The system can take large sets of retrieved documents and recursively condense them into a concise executive summary without losing key details.

### 7. Conversation Memory (Session State)
* **The Problem:** Standard RAG systems treat every question as a new interaction, failing to understand "What about the second point?"
* **Our Solution:** We maintain a `ConversationBufferMemory`. The system rephrases follow-up questions into standalone queries using the previous chat history, ensuring a smooth, context-aware conversational flow.

### 8. User Feedback Loop (RLHF Lite)
* **The Problem:** Developers rarely know when the RAG system fails or gives a bad answer.
* **Our Solution:** We included a **Thumbs Up/Down** feedback mechanism. Negative feedback logs the query and retrieved context to a file, creating a dataset for future fine-tuning and system improvement (Reinforcement Learning from Human Feedback).

### 9. Auto-Generated Follow-Up
* **The Problem:** Users often don't know what to ask next after getting an answer.
* **Our Solution:** After generating a response, the LLM analyzes the context to suggest **3 relevant follow-up questions**. This guides the user deeper into the document and improves engagement.

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

### High-Level Flow

User Query → Retriever → Vector Database → Relevant Context → LLM → Final Answer

**Figure 1: High-Level RAG Architecture**  
![High-Level Architecture Diagram](Pipeline1.png)

**Figure 2: Retrieval and Generation Flow**  
![RAG Pipeline Flow](Picture2.png)


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




