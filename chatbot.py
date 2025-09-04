from http.client import responses

import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

st.header('Chatbot')

with st.sidebar:
    st.title('PDF Doc')
    file = st.file_uploader('Upload PDF file and start asking questions', type="pdf")



if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()


    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks=text_splitter.split_text(text)

    # embedding logic here

    #embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings()

    #vector store logic here

    vector_store = FAISS.from_texts(chunks, embeddings)

    # get question from the user
    user_input = st.text_input('Enter your question')

    if user_input != '':
        match = vector_store.similarity_search(user_input)

        llm = ChatOllama(
            temperature = 0,
            model = 'gpt-oss:20b'
        )

        chain = load_qa_chain(llm, chain_type='stuff')
        responses = chain.run(input_documents = match, question = user_input)
        st.write(responses)
