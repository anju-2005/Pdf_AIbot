"""
AI-Powered RAG Chatbot Suite - Streamlit Application
----------------------------------------------------
This is the front-end user interface designed using Streamlit.
It provides a professional, beginner-friendly dark-themed layout where 
students can configure their Gemini API Key, upload multi-format documents,
build indices, and chat with local materials while observing retrieved source chunks.
"""

import os
import streamlit as st
from utils.document_processor import process_uploaded_file, chunk_documents
from utils.rag_engine import build_vector_store, load_vector_store, retrieve_relevant_snippets, generate_rag_answer

# Define directory structures
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Page Setup & Tab Configuration
st.set_page_config(
    page_title="AI RAG Chatbot Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for an exceptionally polished, ultra-modern slate dark interface
st.markdown("""
<style>
    /* Main Background & Root Settings */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #334155;
    }
    
    /* Header Styles & Modern Fonts */
    h1, h2, h3 {
        color: #38bdf8 !important; /* Sky Blue */
        font-family: 'Inter', system-ui, sans-serif;
        font-weight: 700;
    }
    
    /* Chat Message Bubbles styling */
    .user-bubble {
        background-color: #0284c7;
        color: #ffffff;
        padding: 14px 18px;
        border-radius: 18px 18px 2px 18px;
        margin-bottom: 12px;
        display: inline-block;
        max-width: 80%;
    }
    
    .assistant-bubble {
        background-color: #1e293b;
        color: #f1f5f9;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 2px;
        margin-bottom: 12px;
        display: inline-block;
        max-width: 80%;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.image("logo.png", width=100) # Or placeholder unspash
    st.title("Settings Panel")
    
    st.subheader("🔑 1. API Configuration")
    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=env_gemini_key,
        type="password",
        help="Get an API key from https://aistudio.google.com/"
    )

    st.subheader("📂 2. Document Uploads")
    uploaded_files = st.file_uploader(
        "Upload doc files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if st.button("🔄 Commit & Build Store", use_container_width=True):
        if not user_api_key:
            st.error("Please supply a valid Gemini API key first!")
        elif not uploaded_files:
            st.error("No documents loaded yet.")
        else:
            with st.spinner("Processing documents..."):
                all_chunks = []
                for file in uploaded_files:
                    raw_text = process_uploaded_file(file)
                    chunks = chunk_documents(raw_text, source_name=file.name)
                    all_chunks.extend(chunks)
                
                db = build_vector_store(all_chunks)
                st.success("✅ FAISS Vector store indexed successfully!")
                st.session_state["vector_db_loaded"] = True

# --- MAIN CHAT PANEL ---
st.title("🤖 AI-powered RAG Chatbot")
