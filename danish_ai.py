import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
# Initialize the LLM (HuggingFace-based model)
llm = CTransformers(
    model="tiiuae/falcon-7b-instruct",
    model_type="falcon",
    max_new_tokens=512,
    temperature=0.7
)
# Load environment variables
load_dotenv('.env')

# Define the custom prompt template
custom_prompt_template = """
Context: {context}
Question: {question}

Utilize the retrieved information to provide a detailed and informative response. Ensure that the answer is comprehensive and covers all relevant aspects based on the provided context and question.
Also, remember to give every single detail about the answer, including the source page number and lines where the answer can be found.
Use the given information only; do not frame answers if the information is not relevant. Say you can't find the answer for the given question.
Helpful answer:
"""


def set_custom_prompt(context, question):
    return custom_prompt_template.format(context=context, question=question)


# QA Model Function
def train_pdf(pdf_documents):
    """
    Train the retrieval model on the provided PDF documents.
    """
    persist_directory = "./db/trained_data"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and persist the Chroma vectorstore
    vectordb = Chroma.from_texts(
        texts=[doc.page_content for doc in pdf_documents],
        embedding=embeddings,
        metadatas=[{"page": doc.metadata["page"]} for doc in pdf_documents],
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


def qa_bot(vectordb, query):
    """
    Answer questions using the trained retrieval model.
    """
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    result = qa_chain({"query": query})
    return result


# Streamlit App
st.title('🤖 MARUTI AI Q&A ANSWERING SYSTEM')

# Upload PDF document and train the model
pdf_uploaded = st.file_uploader("Upload PDF Document", type="pdf")

if pdf_uploaded:
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_uploaded.read())
            tmp_file.close()

        # Load the PDF file
        loader = PyPDFLoader(tmp_file.name)
        pdf_documents = loader.load()

        # Train the model
        vectordb = train_pdf(pdf_documents)
        st.success("Training completed. You can now ask questions.")

        # Ask questions after training
        query = st.text_input("Enter your question")
        if query:
            result = qa_bot(vectordb, query)

            # Display the result
            st.subheader("Answer:")
            st.write(result["result"])

            # Display metadata (source pages)
            st.subheader("Source Information:")
            for doc in result["source_documents"]:
                st.write(f"Page: {doc.metadata['page']}")
                st.write(doc.page_content)

    except Exception as e:
        st.error(f"An error occurred: {e}")
