
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# ---------------------------------------------------

import os
import streamlit as st
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Streamlit UI ---
st.set_page_config(page_title="PDFPal", page_icon="📄")
st.title("📄 PDF Chatbot with Gemini + LangChain")

# --- API Key ---
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # --- Load PDF ---
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()

    # --- Split Text ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = splitter.split_documents(data)
    texts = [doc.page_content for doc in docs if doc.page_content.strip()]

    # --- Embeddings + Vector Store ---
    try:
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_texts(texts=texts, embedding=embedding)
        retriever = vectorstore.as_retriever(search_type="similarity")
    except Exception as e:
        st.error("❌ Embedding failed. Please check your API key or PDF content.")
        st.stop()

    # --- LLM + Prompt ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_template(
        "You are my personal assistant to help me talk with the PDF. "
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {input}"
    )

    # --- RAG Chain ---
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # --- Chat UI ---
    query = st.chat_input("Ask me anything about the PDF:")
    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})
            st.write(response["answer"])
else:
    st.info("👆 Upload a PDF to get started")
