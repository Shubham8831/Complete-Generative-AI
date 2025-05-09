from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain 
#LLMChain is one of the simplest chain

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model = "mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.8
)

prompt = PromptTemplate(
    template="Suggest a catchy blog title on this topic: {topic}",
    input_variables=['topic']
)

topic = input("Enter a topic: ")

# this formats the prompt according to input topic
formatted_prompt = prompt.format(topic = topic)

blog_title = llm.predict(formatted_prompt) # this calls the llm directely

print(blog_title)