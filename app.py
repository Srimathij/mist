import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context", 
    don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain()
    resp = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("**Response:**", resp["output_text"])


def clear_question():
    st.session_state.question_input = ""


def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with Doc LLM")

    # Initialize session state
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    # CSS to match the input height to the button
    st.markdown(
        """
        <style>
        div.stTextInput > div > div > input {
            height: 2.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Layout: 90% for input, 10% for clear button
    col1, col2 = st.columns([9, 1], gap="small")

    with col1:
        user_question = st.text_input(
            "",  # hide default label
            value=st.session_state.question_input,
            placeholder="Ask a Question from the PDF Files",
            key="question_input",
            label_visibility="collapsed",
        )

    with col2:
        st.button(
            "×",
            key="clear_button",
            help="Clear input",
            on_click=clear_question
        )

    # Process question
    if user_question and user_question.strip():
        user_input(user_question)

    # Sidebar for PDF upload & processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
