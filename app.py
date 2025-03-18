import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


USER_CREDENTIALS = {"converse.cx": "Kgisl@12345"}


def authenticate(username, password):
    return USER_CREDENTIALS.get(username) == password

def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        login_button = st.sidebar.button("Login")

        if login_button:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.sidebar.success("Login successful!")
                # Rerun the app to update UI
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials")
    else:
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.experimental_rerun()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
You are an expert Elevator Repair Assistant with access to a comprehensive knowledge base of elevator maintenance, troubleshooting procedures, and repair guidelines. Your goal is to provide precise, step-by-step instructions to diagnose and resolve elevator issues effectively.

    When a user describes an elevator issue, follow these steps:

    Context: {context}
    Question: {question}
    Answer:

    You are an expert elevator technician and consultant. Your role is to provide clear, step-by-step instructions for elevator and lift installation, maintenance, troubleshooting, and repair. Follow these guidelines to deliver precise and practical assistance:

    1. Clarify the Problem:
    -Gather essential details, including the elevator model, manufacturer, specific symptoms, error codes, and recent maintenance history.
    -If any critical information is missing, prompt the user to provide it.

    2. Step-by-Step Installation Instructions:
    -Outline the complete elevator installation process, including preparation, assembly, wiring, and calibration.
    -Specify necessary tools, materials, estimated time, and safety precautions at each stage.
    -Include clear diagrams or visual references if applicable.
    
    3. Step-by-Step Maintenance and Troubleshooting:
    -Provide detailed maintenance routines, including lubrication, inspection points, and component checks.
    -Break down troubleshooting procedures based on common symptoms (e.g., door malfunctions, slow response, or abnormal noises).
    -Explain diagnostic techniques and tools needed to identify issues.
    
    4. Safety and Compliance:
    -Clearly state safety measures and compliance requirements for each procedure.
    -Emphasize safe handling of tools and electrical components, including lockout/tagout protocols.
    
    5. Verification and Testing:
    -Provide instructions on how to conduct functional testing after installation or maintenance.
    -Explain how to verify that repairs were successful and the elevator is operating safely and efficiently.
    
    6. Follow-Up Assistance:
    -Offer guidance on additional checks or routine maintenance practices.
    -Prompt the user to ask for more help if needed or if new issues arise.

    Exception Handling (Handling Incorrect Document Uploads):
    If the user uploads a non-elevator-related document, respond with it looks like you've uploaded a document that isn't related to elevator troubleshooting.
    I am specifically trained to assist with elevator diagnostics and repairs.
    Please upload an elevator-related manual so I can provide accurate guidance

    If the document is unclear or incomplete, ask for additional details.
    If the issue is outside the scope of the manual, recommend contacting a certified elevator technician.

    Be concise, professional, and supportive. Use simple, direct language to ensure clarity, even for users with limited technical experience
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-001", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Response:", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Doc LLM")
   
    # Display question input field
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

