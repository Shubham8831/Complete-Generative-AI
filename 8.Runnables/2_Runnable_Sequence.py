from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence 

load_dotenv()

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

model = ChatOpenAI(
    model = "mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.8
)

# chain = RunnableSequence(prompt1, model, parser) # here we will sequentially give all the runnables
# print(chain.invoke({'topic':"artificial intelligence"}))

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))