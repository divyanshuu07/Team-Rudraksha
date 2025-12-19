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

## Innovative Features

- Multilingual query support  
- Multimodal-ready architecture  
- Query expansion layer for improved retrieval  
- Whole-document summarization  
- Conversational memory across interactions  
- User feedback loop for answer quality  
- Auto-generated follow-up questions  
- Citation and transparency for retrieved content  

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
