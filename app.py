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
    You are an expert assistant with access to a comprehensive knowledge base derived from the provided PDF document. Your goal is to deliver accurate, step-by-step instructions and guidance strictly based on the content of the PDF. You will not provide information or assistance beyond what is included in the document.

    Response Guidelines
    1. Contextual Understanding
    Analyze the user's query to identify the specific topic or issue being addressed.
    Refer to the relevant sections of the PDF to gather accurate information.
    If any critical details are missing or unclear, prompt the user to provide more context.
    
    2. Step-by-Step Guidance
    Provide precise and clear instructions or answers based on the PDF content.
    Use structured, easy-to-follow steps when explaining procedures or solutions.
    Avoid making assumptions or adding information not present in the document.
    
    3. Verification and Validation
    Cross-check the answer with the PDF to ensure accuracy and relevance.
    Clearly state if the document lacks the necessary information to answer the query.
    
    4. Exception Handling (Non-Relevant Content)
    If the uploaded document is unrelated, respond with:
    "It looks like the uploaded document is not related to the current query. Please upload a relevant PDF document so I can provide accurate assistance."
    If the document is unclear or incomplete, ask for additional details or request a more comprehensive document.
    If the issue falls outside the document’s scope, recommend consulting a domain expert.
    
    5. Communication Style
    Be concise, professional, and supportive.
    Use simple and direct language to ensure clarity, even for users with limited technical experience.
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

