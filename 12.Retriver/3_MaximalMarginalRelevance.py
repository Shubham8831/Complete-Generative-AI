# it finds relevant yet diverse result 

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS  # Use FAISS instead of Chroma
from langchain_core.documents import Document

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Step 2: Create the FAISS vector store from documents
vector_store = FAISS.from_documents(
    documents= docs,
    embedding=embedding
)

retriver = vector_store.as_retriever(
    search_type = "mmr",  #Can be "similarity" (default), "mmr", or "similarity_score_threshold"
    search_kwargs = {"k": 3, # Amount of documents to return (Default: 4)
            "lambda_mult": 0.5 # Diversity of results returned by MMR; 
                # 1 for minimum diversity(behaves like similarity search) and 0 for maximum.(more diverse results) (Default: 0.5)
                })


query = "what is langchain?"
results = retriver.invoke(query)

for i,doc in enumerate(results):
    print(f"\n-- Result {i+1} --")
    print(doc.page_content)
