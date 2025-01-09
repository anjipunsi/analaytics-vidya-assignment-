import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the FAISS index and dataset
faiss_index_path = "courses_faiss_index.bin"  # Ensure this path matches your saved FAISS index
faiss_index = faiss.read_index(faiss_index_path)

data = pd.read_csv("courses_data.csv")  # Ensure this path matches your dataset with embeddings

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to search for relevant courses
def search_courses(query, top_k=5):
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = data.iloc[indices[0]].copy()
    results['Distance'] = distances[0]
    return results

# Streamlit interface
st.title("Smart Search for Analytics Vidhya Free Courses")
st.write("Type a query to find the most relevant courses.")

query = st.text_input("Search query", "")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    if query:
        results = search_courses(query, top_k)
        st.write("Top Results:")
        st.table(results[['Title', 'Category', 'Review Count', 'Price', 'Distance']])
    else:
        st.warning("Please enter a query!")
