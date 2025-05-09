from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model1 = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)

model2 = OllamaLLM(
    model = "gemma3:1b"
)

prompt1 = PromptTemplate(
    template="write a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="write a small linkedin post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain1 = RunnableSequence(prompt1, model1, parser)
chain2 = prompt2 | model2 | parser

chain = RunnableParallel(
    {"tweet" : chain1,
     "linkedin":chain2
    }
)


result = chain.invoke({"topic": "war"})
print(result)
print(result['tweet'])
print()
print(result['linkedin'])