import os
import tempfile
import streamlit as st
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Function to process the uploaded document
def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.close()  # Explicitly close the file

    # Load the document using PyPDFLoader
    loader = PyPDFLoader(tmp_file.name)
    documents = loader.load()

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    return texts

# Streamlit UI for document upload and query
st.title("Document Retrieval System")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Process the document
    texts = process_document(uploaded_file)
    st.success("Document successfully processed!")

    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

    st.info("Document indexed into the vector store. Ready for querying!")

    # User query
    query = st.text_input("Enter your query")
    if st.button("Search"):
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # Top 5 results
        results = retriever.get_relevant_documents(query)
        print(results)

        # Display the results
        for i, result in enumerate(results):
            st.write(f"### Result {i + 1}")
            st.write(result.page_content)
