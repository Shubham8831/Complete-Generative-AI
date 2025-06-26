from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

model = OllamaLLM(model = 'gemma3:1b')

parser = StrOutputParser()
embedding_model = OllamaEmbeddings(model = "nomic-embed-text")


loader = TextLoader(file_path="16.Gen AI Udemy Course/test.txt")
doc = loader.load()
# print(doc[0].page_content)
texts = doc[0].page_content # taking only page content not meta data as it is not useful


# TEXT SPLITTER 
splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 10)
chunks = splitter.create_documents([texts])


#vector store

vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)


#making a retriver
retriver = vector_db.as_retriever()


# if we want to first laod a context document from the large chunk of document and answer the query from this contextual chunk then,
from langchain.chains.combine_documents import create_stuff_documents_chain
prompt = ChatPromptTemplate.from_template(
    """ Answer the following question based on the provided context.
    <context>
    {context}
    </context>

"""
)
document_chain = create_stuff_documents_chain(model, prompt)


from langchain.chains.retrieval import create_retrieval_chain
retriver_chain = create_retrieval_chain(retriver, document_chain)

response = retriver_chain.invoke({"input":"what is machine learning"})
print(response["answer"])
print(response["context"])