import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.title("📄 Document Question Answering Bot")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

question = st.text_input("Ask a question")

if question:
    docs = db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer only from the context below.

If the answer is not present, say:
I could not find this information in the documents.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)