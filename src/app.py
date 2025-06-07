import streamlit as st
import json
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import os
import gdown

# Load .env file and get API key
load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key not found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Configure Gemini with API key
genai.configure(api_key=api_key)

# Select Gemini model (adjust if needed)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Download faiss_index.bin if not exists
FAISS_FILE = "faiss_index.bin"
FAISS_FILE_ID = "1xxYYhMBkYypXTVNzhlcOdYQ2KxC7g-Gp"

if not os.path.exists(FAISS_FILE):
    st.info("Downloading FAISS index from Google Drive...")
    url = f"https://drive.google.com/uc?id={FAISS_FILE_ID}"
    gdown.download(url, FAISS_FILE, quiet=False)

# Load FAISS index (make sure this file is uploaded or accessible)
@st.cache_resource
def load_faiss_index():
    return faiss.read_index(FAISS_FILE)

index = load_faiss_index()

# Load labeled comments JSON (make sure this file is uploaded or accessible)
@st.cache_data
def load_labeled_comments():
    with open('sentiment_labeled_comments.json', 'r', encoding='utf-8') as f:
        return json.load(f)

labeled_data = load_labeled_comments()

texts = [item['cleaned_text'] for item in labeled_data if item['cleaned_text'].strip()]

# Load embedding model once
@st.cache_resource
def load_embedder():
    return SentenceTransformer('intfloat/multilingual-e5-large')

embedder = load_embedder()

# Retrieve function to find similar comments from FAISS
def retrieve_similar_comments(query, k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        results.append({
            "text": texts[idx],
            "sentiment_label": labeled_data[idx]['sentiment_label']
        })
    return results

# Streamlit UI

st.title("Amharic Sentiment Analysis with Gemini & FAISS RAG")

query = st.text_area("Enter your query in Amharic:", height=100)

if st.button("Analyze Sentiment"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Retrieving relevant comments..."):
            retrieved = retrieve_similar_comments(query, k=5)

        sentiment_summary = {"positive": 0, "neutral": 0, "negative": 0}
        for item in retrieved:
            sentiment_summary[item["sentiment_label"]] += 1
        dominant_sentiment = max(sentiment_summary, key=sentiment_summary.get)

        context = "\n".join([f"- ({item['sentiment_label']}) {item['text']}" for item in retrieved])

        prompt = f"""
በአማርኛ ስሜት ተያይዞ በተሰጠው እና በቀጣዩ ጥያቄ ላይ መረጃ እንደሚከተሉት ያቀርቡ::

ጥያቄ: {query}

ተመራጭ አስተያየቶች:
{context}

ብዛት ያለው ስሜት: {dominant_sentiment}

መልስ:
"""

        with st.spinner("Generating answer with Gemini..."):
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=256
                )
            )

        st.subheader("Generated Answer:")
        st.write(response.text)

        st.subheader("Sentiment Summary of Retrieved Comments:")
        st.json(sentiment_summary)

        st.subheader("Retrieved Comments:")
        for i, item in enumerate(retrieved, 1):
            st.markdown(f"{i}. **({item['sentiment_label']})** {item['text']}")
