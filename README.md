Readme:
Project Description
This project implements a PDF-based chatbot using Streamlit and Langchain. It allows users to upload a PDF document, processes its content by splitting into chunks, creates embeddings, and answers user questions based on the document's content.

Key Features
Text extraction from PDF files using PyPDF2.

Text chunking with Langchain's RecursiveCharacterTextSplitter.

Embedding generation using OllamaEmbeddings (not OpenAI embeddings).

Vector search with FAISS.

Question answering using ChatOllama, leveraging Ollama's local large language models (LLMs) instead of OpenAI.

User interface built with Streamlit.


Required Packages
Please install the following dependencies to run the project:

bash
pip install streamlit
pip install PyPDF2
pip install langchain
pip install faiss-cpu
pip install langchain_community


Special Notes
This project uses Ollama's LLM and embeddings (ChatOllama and OllamaEmbeddings from langchain_community), which run locally or through Ollama server instead of OpenAI API.

Make sure you have Ollama installed and the required models pulled on your machine for inference.

Replace any OpenAIEmbeddings usage with OllamaEmbeddings.
