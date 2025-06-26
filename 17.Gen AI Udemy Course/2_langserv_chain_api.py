# LangServe turns your LangChain apps into first-class, deployable APIs, complete with validation, scaling, and monitoring,
#  so you can focus on building capabilities rather than infrastructure plumbing.
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


model = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)


system_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", '{text}')
])

parser = StrOutputParser()


chain = prompt | model | parser



# print(result)

# app definition
app = FastAPI(
    title = "Langchain Server",
    version="0.11",
    description="This a simple API using langchain runnable interfaces "
)

# adding chain routs
add_routes(
    app,
    chain,
    path = "/chain"
)

if __name__  == "__main__":
    result = chain.invoke({
    "language":"French",
    "text":"hello shubham"
})
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8001)