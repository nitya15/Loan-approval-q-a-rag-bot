import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from groq import Groq

# Streamlit UI
st.set_page_config(page_title="Loan Approval RAG Chatbot")
st.title("🤖 Loan Approval Q&A Chatbot (Groq + RAG)")

# Groq API Key input
groq_api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")

# Load CSV dataset
@st.cache_data
def load_dataset():
    return pd.read_csv("Training Dataset.csv")

# Convert each row into a document-like sentence
def row_to_doc(row):
    return (
        f"Gender: {row['Gender']}, Married: {row['Married']}, Education: {row['Education']}, "
        f"Self Employed: {row['Self_Employed']}, Income: {row['ApplicantIncome']} + {row['CoapplicantIncome']}, "
        f"Loan Amount: {row['LoanAmount']}, Term: {row['Loan_Amount_Term']}, Credit History: {row['Credit_History']}, "
        f"Area: {row['Property_Area']}, Approved: {row['Loan_Status']}"
    )

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Embed docs
@st.cache_data
def embed_docs(docs, _embedder):
    return _embedder.encode(docs, show_progress_bar=True)

# Groq response generator
def generate_answer_with_groq(query, context_docs, client):
    prompt = f"""Answer the question using the context below.

Context:
{chr(10).join(context_docs)}

Question:
{query}

Answer:"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # or llama3-70b-8192 if desired
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

# Main logic
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    client = Groq()  # auto-loads API key from env var

    df = load_dataset()
    docs = df.apply(row_to_doc, axis=1).tolist()

    embedder = load_embedder()
    doc_embeddings = embed_docs(docs, _embedder=embedder)

    dim = doc_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(doc_embeddings))

    user_query = st.text_area("💬 Ask a question about loan approvals:")

    if st.button("🔍 Get Answer") and user_query:
        with st.spinner("Searching for relevant examples..."):
            q_embedding = embedder.encode([user_query])
            _, indices = index.search(np.array(q_embedding), k=3)
            context = [docs[i] for i in indices[0]]

        with st.spinner("Thinking with Groq..."):
            answer = generate_answer_with_groq(user_query, context, client)

        st.subheader("✅ Answer")
        st.write(answer)
else:
    st.info("Please enter your Groq API key to begin.")