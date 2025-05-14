from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import  PromptTemplate
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled 
import re

#BELOW CODE FINDS OUT YOUTUBE VIDEO ID FORM A URL
# https://youtu.be/wjZofJX0v4M?si=605cGBymapYE-jHC
# https://youtu.be/Gfr50f6ZBvo?si=RBdhGcT8a_984CAy
youtube_url = input("Enter YouTube URL: ") # Prompt user for a YouTube URL and store it for later use
def extract_video_id(url: str) -> str:  # Function to extract the video ID
    match = re.search(r"(?:youtu\.be/|v=|embed/)([^&?#]+)", url)
    return match.group(1) if match else ""

video_id = extract_video_id(youtube_url) # Extract and store the video ID in a variable for future operations


# STEP - 1 DATA INGESTION( INDEXING)

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id= video_id, languages=['en'])
    #flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)
except TranscriptsDisabled:
    print("No Caption avilable for this video.")

# TEXT SPLITTER 
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
chunks = splitter.create_documents([transcript])

#VECTOR STORE
embedding_model = OllamaEmbeddings(model = "nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# RETRIEVER
retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = { "k":4})

# print(retriever.invoke("what is llm"))

llm = ChatOllama(model = "gemma3:1b")
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = input("Aks your Que : ")
retrived_docs = retriever.invoke(question) # this will give 4 relvant documents 
context_text = "\n\n".join(doc.page_content for doc in retrived_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question}) # this one is final prompt

#GENERATION
answer = llm.invoke(final_prompt)
print(answer.content)


# #MAKING A CHAIN
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# def format_docs(retrieved_docs):
#   context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#   return context_text

# parallel_chain = RunnableParallel({
#     'context': retriever | RunnableLambda(format_docs),
#     'question': RunnablePassthrough()
# })

# parser = StrOutputParser()

# main_chain = parallel_chain | prompt | llm | parser

# main_chain.invoke('Can you summarize the video')    