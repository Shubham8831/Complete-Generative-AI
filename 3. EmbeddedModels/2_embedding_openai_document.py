# Importing OpenAIEmbeddings class from langchain_openai
# This allows us to use OpenAI's embedding models easily
from langchain_openai import OpenAIEmbeddings

# Importing load_dotenv to read variables like API keys from a .env file
from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY) from .env file into the script
load_dotenv()

# Create an embedding object using OpenAI's 'text-embedding-3-large' model
# Setting dimensions to 32 means the model will return a 32-number vector for each input text
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',  # Name of the embedding model to use
    dimensions=32                    # Optional: reduce size of output vector for performance
)

# A list of text documents to embed (turn into vectors)
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Get the embeddings for multiple documents (each document will be converted into a separate vector)
result1 = embedding.embed_documents(documents)

# Print the embedding vectors of all the documents (this will be a list of lists)
print(str(result1))
