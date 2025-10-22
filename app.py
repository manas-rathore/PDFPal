_import_("pysqlite3")
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

# --- API Key ---
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

st.title("📄 PDF Chatbot with Gemini + LangChain")

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # --- PDF Loader ---
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()

    # --- Text Splitting ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    docs = text_splitter.split_documents(data)

    # --- Embeddings + Vector Store ---
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )

    retriever = vectorstore.as_retriever(search_type="similarity")

    # --- LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # --- Prompt Template  ---
    prompt = ChatPromptTemplate.from_template(
        "You are my personal assistant to help me talk with the PDF. "
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {input}"
    )

    # --- RAG Chain ---
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- Chat UI ---
    query = st.chat_input("Ask me anything about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})
            st.write(response["answer"])
else:
    st.info("👆 Upload a PDF to get started")
