from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template1 = PromptTemplate(
    template="generate a detailed report on the {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="tell 5 important points from the report. \n {text}",
    input_variables=["text"]
)

model = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)
parser = StrOutputParser()



chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Future of Generative ai"})
# print(result)

chain.get_graph().print_ascii() # visualizing the chian