from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate(
    template="generate intresting facts about {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)

parser = StrOutputParser()

chain = prompt | model | parser # langchian expression language

result = chain.invoke({'topic': 'cricket'})
# print(result)

# to visualize your chian

chain.get_graph().print_ascii()