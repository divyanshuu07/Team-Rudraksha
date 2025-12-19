import streamlit as st
import os
import tempfile

# --- IMPORTS (UPDATED FOR OPENAI) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # <--- NEW IMPORTS
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="OpenAI RAG: Multi-Doc Edition", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .source-box {
        border-left: 4px solid #10a37f;  /* OpenAI Green */
        background-color: #f1f1f1;
        color: black;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. SETUP API KEY ---
def load_api_key():
    try:
        with open("api_open.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

OPENAI_API_KEY = load_api_key()

# --- 2. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "full_text_content" not in st.session_state:
    st.session_state.full_text_content = ""

# --- 3. BACKEND FUNCTIONS (UPDATED) ---

def get_llm():
    # Switched to ChatOpenAI
    # 'gpt-4o' or 'gpt-3.5-turbo' are good options
    return ChatOpenAI(
        model="gpt-4o",  
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

def get_embeddings():
    # Switched to OpenAIEmbeddings
    # 'text-embedding-3-small' is efficient and high quality
    return OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=OPENAI_API_KEY
    )

def process_documents(uploaded_files, youtube_url):
    docs = []
    status_text = st.empty()
    
    # 1. Process PDFs
    if uploaded_files:
        for i, file in enumerate(uploaded_files):
            status_text.text(f"📄 Processing PDF {i+1}/{len(uploaded_files)}: {file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = file.name 
                docs.extend(loaded_docs)
            finally:
                os.remove(tmp_path) 

    # 2. Process YouTube
    if youtube_url:
        status_text.text("🎥 Reading YouTube Transcript...")
        try:
            loader = YoutubeLoader.from_youtube_url(
                youtube_url, 
                add_video_info=True,
                language=["en", "hi"],
                translation="en"
            )
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading YouTube: {e}")
            if not docs: return None

    if not docs:
        st.error("No content found.")
        return None
    
    st.session_state.full_text_content = "\n\n".join([d.page_content for d in docs])

    # Splitting
    status_text.text(f"✂️ Chunking {len(docs)} pages & Embedding...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Create Vector DB (Chroma)
    # Note: If switching embedding models, we usually need a fresh DB or collection name
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=get_embeddings(),
        collection_name="openai_rag_collection" # <--- Changed collection name to avoid conflicts
    )
    
    # Create Keyword Retriever
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = 3
    
    status_text.empty()
    return vectorstore, bm25

def summarize_content(language):
    llm = get_llm()
    full_text = st.session_state.full_text_content
    
    if not full_text:
        return "No document text found to summarize."
        
    prompt = f"""
    You are a helpful expert assistant. 
    Please provide a comprehensive summary of the provided content.
    
    LANGUAGE INSTRUCTION: Output the summary in {language}.

    Content:
    {full_text}
    """
    
    response = llm.invoke(prompt)
    return response.content

# --- 4. MANUAL RAG LOGIC ---

def run_hybrid_rag(query, language, chat_history, vector_db, bm25):
    llm = get_llm()
    
    final_query = query
    if chat_history:
        history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-4:]])
        rewrite_prompt = f"""
        Given the chat history below and the user's new question, 
        rewrite the question to be a standalone sentence.
        STRICTLY OUTPUT ENGLISH ONLY.
        
        Chat History:
        {history_text}
        
        User's New Question: {query}
        
        Standalone Question:
        """
        response = llm.invoke(rewrite_prompt)
        final_query = response.content.strip()
    
    vector_docs = vector_db.similarity_search(final_query, k=3)
    keyword_docs = bm25.invoke(final_query)
    
    all_docs = vector_docs + keyword_docs
    unique_docs = []
    seen_content = set()
    
    for doc in all_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
    
    final_context_docs = unique_docs[:5]
    
    context_text = "\n\n".join([d.page_content for d in final_context_docs])
    
    answer_prompt = f"""
    You are a helpful assistant. Answer the user's question using ONLY the context provided below.
    If the context comes from multiple documents, mention which document you are referencing.
    
    LANGUAGE INSTRUCTION: Answer in {language}.
    
    Context:
    {context_text}
    
    Question: {final_query}
    
    Answer:
    """
    
    final_response = llm.invoke(answer_prompt)
    return final_response.content, final_context_docs

# --- 5. UI LOGIC ---

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API Key Missing! Please put it in 'api_key.txt'.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    target_language = st.radio("Select Language:", ("English", "Hindi"))
    
    st.divider()
    
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type="pdf", 
        accept_multiple_files=True 
    )
    
    yt_url = st.text_input("YouTube URL")
    
    if st.button("Build Knowledge Base"):
        if uploaded_files or yt_url:
            with st.spinner("Processing..."):
                v_db, bm25 = process_documents(uploaded_files, yt_url)
                if v_db:
                    st.session_state.vector_db = v_db
                    st.session_state.bm25_retriever = bm25
                    st.success("✅ Ready!")
        else:
            st.warning("Provide a file or URL.")
            
    st.divider()
    
    if st.button("📝 Summarize All Content"):
        if not st.session_state.full_text_content:
            st.error("Please build the knowledge base first!")
        else:
            with st.spinner("Generating summary..."):
                st.session_state.chat_history.append(HumanMessage(content="Please summarize the content."))
                summary = summarize_content(target_language)
                st.session_state.chat_history.append(AIMessage(content=summary))

st.title("OpenAI RAG (Multi-Doc Edition)")

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

user_input = st.chat_input("Ask something...")

if user_input:
    if not st.session_state.vector_db:
        st.warning("Please upload a document first!")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking in {target_language}..."):
                answer, sources = run_hybrid_rag(
                    user_input, 
                    target_language, 
                    st.session_state.chat_history,
                    st.session_state.vector_db,
                    st.session_state.bm25_retriever
                )
                
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.markdown(answer)
                
                with st.expander("🔎 View Sources"):
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'Video')
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source {i+1}:</b> {source_name} (Page: {page})<br>
                            <i>{doc.page_content[:300]}...</i>
                        </div>
                        """, unsafe_allow_html=True)