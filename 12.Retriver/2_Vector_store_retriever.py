from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS  # Use FAISS instead of Chroma
from langchain_core.documents import Document

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Set up your embedding model (nomic-embed-text)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Step 3: Create FAISS vector store in memory
vector_store = FAISS.from_documents(documents, embedding_model)

# Step 4: Convert vectorstore into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Step 5: Query the retriever
query = "What is Chroma used for?"
results = retriever.get_relevant_documents(query)

# Step 6: Print the results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
