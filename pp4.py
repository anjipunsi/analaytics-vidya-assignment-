import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the FAISS index and dataset
faiss_index = faiss.read_index("courses_faiss_index.bin")
data = pd.read_csv("courses_data.csv")

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def search_courses(query, top_k=5):
    """
    Search for the most relevant courses based on the query.

    Args:
    query (str): The search query.
    top_k (int): Number of top results to retrieve.

    Returns:
    pd.DataFrame: The top-k relevant courses with their details.
    """
    # Generate embedding for the query
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)

    # Perform search using FAISS
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Retrieve the top-k courses
    results = data.iloc[indices[0]].copy()
    results['Distance'] = distances[0]
    
    return results

# Example usage
query = "Learn machine learning algorithms"
top_k_results = search_courses(query, top_k=5)


# Display full details
print("Top Results:")
print(top_k_results[['Title', 'Category', 'Review Count', 'Price', 'Distance']])
