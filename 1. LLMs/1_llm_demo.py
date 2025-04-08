from langchain_openai import OpenAI # importing libraries
from dotenv import load_dotenv

load_dotenv() # loading the api key

llm = OpenAI(model = 'gpt-3.5-turbo-instruct') # object of openai

result = llm.invoke("whai is capital of India") # prompt

print(result) # response

# this is not used now (it is just for learning purpose)
# currently chatmodels are used 