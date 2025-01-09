from sentence_transformers import SentenceTransformer
import pandas as pd

# Load the cleaned data
data = pd.read_csv("cleaned_courses_data.csv")  # Replace with your cleaned data file path

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
data['Embeddings'] = data['Text For Embedding'].apply(lambda x: model.encode(x).tolist())

# Save the data with embeddings
data.to_csv("data_with_embeddings.csv", index=False)
print("Embeddings generated and saved to data_with_embeddings.csv")
