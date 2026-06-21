# Document Question Answering Bot

## Project Description

This project is a Document Question Answering Bot built using LangChain, FAISS, HuggingFace Embeddings, and Google Gemini. The bot answers user questions based on the contents of uploaded PDF documents and displays the source file and page number.

## Tech Stack

* Python
* LangChain
* FAISS
* HuggingFace Embeddings
* Google Gemini API
* PyPDF

## Architecture

Documents → Chunking → Embeddings → FAISS Vector Store → Retriever → Gemini → Answer

## Chunking Strategy

* Chunk Size: 500
* Chunk Overlap: 100
* RecursiveCharacterTextSplitter

## Setup Instructions

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create vector database

```bash
python index.py
```

3. Run the bot

```bash
python app.py
```

## Environment Variables

Create a `.env` file and add:

```text
GOOGLE_API_KEY=your_api_key
```

## Example Queries

* What is Machine Learning?
* What is Artificial Intelligence?
* What is Cloud Computing?
* What is Data Science?
* Explain Python programming language.
