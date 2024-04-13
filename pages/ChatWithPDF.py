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
import io
from PIL import Image
import requests

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

    
def main():
    st.set_page_config("CHAT WITH PDF💁")
    st.header("SMART TUTOR CHATBOT 📚")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)
    
    if user_question == "magnetic separation" or user_question=="what is magnetic separation" or user_question=="what is magnetic separation?" or user_question=="explain magnetic separation":
        st.write("Reference video:","https://www.youtube.com/watch?v=cs-IZOFY53A")
    
    elif user_question == "froth floatation method" or user_question == "froth floatation" or user_question=="what is froth floatation method"  or user_question=="what is froth floatation?" or user_question=="what is froth floatation method?" or user_question=="explain froth floatation method":
        st.write("Reference video:","https://www.youtube.com/watch?v=zFzD1wAwldU")
        
    elif user_question == "valence bond theory" or user_question=="what is valence bond theory" or user_question=="what is valence bond theory?" or user_question=="explain valence bond theory":
        st.write("Reference video:","https://www.youtube.com/watch?v=86rNPVAtj0Y")
    
    elif user_question == "coulomb's law" or user_question=="what is coulomb's law" or user_question=="what is coulomb's law?" or user_question=="explain coulomb's law":
        st.write("Reference video:","https://www.youtube.com/watch?v=ECBEJnAVFMk")
    
    elif user_question == "electric field" or user_question=="what is electric field" or user_question=="what is electric field?" or user_question=="explain electric field":
        st.write("Reference video:","https://www.youtube.com/watch?v=DThrw4cvpe8")
    
    elif user_question == "electric field lines" or user_question=="what is electric field lines" or user_question=="what is electric field lines?" or user_question=="explain electric field lines":
        st.write("Reference video:","https://www.youtube.com/watch?v=cs-IZOFY53A")

    
        
    if len(user_question)>= 1:
        image_bytes = query({"inputs": user_question,})
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image)

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
