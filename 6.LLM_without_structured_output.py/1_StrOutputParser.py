from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


# Choose a Together-supported model
llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)



# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

result = llm.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = llm.invoke(prompt2)

print(result1.content)
