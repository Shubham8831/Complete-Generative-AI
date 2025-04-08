# Importing OpenAIEmbeddings class from langchain_openai
# This allows us to use OpenAI's embedding models easily
from langchain_openai import OpenAIEmbeddings

# Imports the load_dotenv function to read environment variables from a .env file
# This is how your OpenAI API key gets securely loaded
from dotenv import load_dotenv

# Loads environment variables (like OPENAI_API_KEY) from the .env file into your script
load_dotenv()

# Creating an embedding object using OpenAI's 'text-embedding-3-large' model
# 'dimensions=32' means we want the output embedding vector to have only 32 numbers (default is 1536 or higher)
embedding = OpenAIEmbeddings(
    model = 'text-embedding-3-large',  # The specific embedding model from OpenAI
    dimensions = 32  # Compress the vector to 32 dimensions to make it lighter
)

# Convert the text "Delhi is the capital of India" into an embedding (vector of numbers)
result = embedding.embed_query("Delhi is the capital of India")

# Print the resulting vector (list of 32 floating-point numbers)
print(str(result))
