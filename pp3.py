import faiss
import numpy as np
import pandas as pd

# Load the data with embeddings
data = pd.read_csv("data_with_embeddings.csv")

# Convert the embeddings from string format to a NumPy array
embeddings = np.array(data['Embeddings'].apply(eval).tolist()).astype('float32')

# Initialize a FAISS index
dimension = embeddings.shape[1]  # The dimensionality of the embeddings
faiss_index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
faiss_index.add(embeddings)

# Save the FAISS index
faiss.write_index(faiss_index, "courses_faiss_index.bin")
print("FAISS index saved successfully!")

# Save the original data (for lookup purposes during search)
data.to_csv("courses_data.csv", index=False)
