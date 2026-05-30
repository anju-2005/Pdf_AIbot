"""
RAG Engine Utility
------------------
Manages local FAISS vector stores, generating local MiniLM embeddings,
and prompting Gemini using retrieved contexts.
"""

import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_DIR = "vectorstore"

def get_embedding_model():
    # Uses HuggingFace all-MiniLM-L6-v2 which runs offline on CPU with zero tokens
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(documents):
    db = FAISS.from_documents(documents, get_embedding_model())
    db.save_local(VECTORSTORE_DIR)
    return db

def generate_rag_answer(query, retrieved_docs, chat_history, api_key):
    genai.configure(api_key=api_key)
    
    context_str = ""
    for idx, doc in enumerate(retrieved_docs):
        context_str += f"[Source {idx+1} | {doc.metadata.get('source')}]: {doc.page_content}\n\n"
        
    full_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(full_prompt)
    return response.text, full_prompt
