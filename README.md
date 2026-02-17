# IPC RAG Chatbot ⚖️🤖

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain 1.x that enables conversational querying over the Indian Penal Code (IPC).

## 🚀 Features

- PDF ingestion and text extraction
- Semantic chunking using RecursiveCharacterTextSplitter
- OpenAI embeddings
- Chroma vector database
- Conversational RAG pipeline
- Persistent vector storage

## 🏗 Architecture

PDF → Chunking → Embeddings → ChromaDB → Retriever → LLM → Response

## 🛠 Tech Stack

- Python
- LangChain 1.x
- OpenAI Embeddings
- ChromaDB

## 💬 Sample Queries

- What is the section related to unlawful assembly?
- What is the punishment for that?
- Which section deals with counterfeiting currency-notes?

## 🧪 How to Run

1. Clone this repository
2. Install dependencies:
   pip install -r requirements.txt
3. Set your OpenAI API key
4. Run:
   python app.py

## 📸 Sample Output

<img width="848" height="245" alt="image" src="https://github.com/user-attachments/assets/23d6afae-063b-42e8-beaa-8e515ba79cfb" />

## ⚠ Disclaimer

This chatbot is for educational purposes only and does not provide legal advice.
