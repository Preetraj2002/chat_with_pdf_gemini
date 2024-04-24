import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain

from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
st.session_state.pdfuploaded = False

def get_pdf_text(pdf_docs):
    text =''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("faiss_index1")
    


def get_conversational_chain():
    
    # from template loads the prompt text as f-string
    prompt = PromptTemplate.from_template(
    """
    Answer the question as detailed as possible from the provided context. 
    Also make sure to provide the details. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context:\n{context}\n
    Question: \n{question}\n
    
    Answer:
    """
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    
    chain = load_qa_chain(llm,chain_type='stuff',prompt=prompt)
    
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()  #get the chain
    
    response = chain({"input_documents": docs, "question": user_question})
    
    print(response)
    
    st.write("Reply: ", response["output_text"])
    
    
def main():
    st.set_page_config(page_title="PDF Chat QA App", page_icon=":memo:")
    st.title(":memo: PDF Chat QA App")
    
    # Input field for user question
    question = st.text_input("Ask a question:", "")
    
    if question:
        user_input(question)

    # if not question:
    #     st.warning("Please enter a question.")

    with st.sidebar:
        st.header("Instructions")
        st.markdown(
            "1. **Upload PDFs**: Upload one or more PDF files.\n"
            "2. **Ask Questions**: Enter your questions in the chat box.\n"
            "3. **Search**: Click 'Search' to find answers in the PDFs."
        )
        # Upload PDF files
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

        if uploaded_files:
            st.session_state.pdfuploaded = True
            
        if st.session_state.pdfuploaded:    
            # Button to create faiss index
            if st.button("Submit and Process"):
                
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(uploaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done!") 
                    
        if not st.session_state.pdfuploaded:            
            if question:
                st.info("Using the previouly stored faiss index to answer the question")
                return
        
            st.warning("Please upload one or more PDF files.")
            
        

if __name__ == "__main__":
    main()
    
    