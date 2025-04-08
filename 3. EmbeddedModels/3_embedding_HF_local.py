# Import the HuggingFaceEmbeddings class from langchain_huggingface
# This lets you use embedding models from Hugging Face (like sentence transformers)
from langchain_huggingface import HuggingFaceEmbeddings

# Create an embedding object using a pre-trained model from Hugging Face
# "all-MiniLM-L6-v2" is a lightweight model good for semantic similarity and search
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the text you want to convert into an embedding (vector)
text = "Delhi is the capital of India"


# A list of text documents to embed (turn into vectors)
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]


# Generate the embedding vector for the text
# (NOTE: There's a small typo here – should be embed_query not enbed_query)
vector = embedding.embed_query(text)

# Print the generated vector (a list of numbers that represents the meaning of the text)
print(str(vector))

# Get the embeddings for multiple documents (each document will be converted into a separate vector)
vector1 = embedding.embed_documents(documents)

# Print the embedding vectors of all the documents (this will be a list of lists)
print(str(vector1))
