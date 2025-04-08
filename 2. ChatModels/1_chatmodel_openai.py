from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model ="gpt-4" , temperature= 0, max_completion_tokens=10) 
# temperature is a creativity parameter. it controlls the randomness of a language model.

# lower value(0.0 - 0.3) -> more deterministic and predictable
# higher value(0.7-0.5) -> more random, creative and diverse

# factual answer math, code, facts [0.0-0.3]
# balanced response general QA, explanation [0.5-0.7]
#creative writing, storytellingm jokes [0.9-1.2]
# maximum randomness wild ideas, brainstorming [1.5+]


# max_completion_tokens : how much tokens we get in output

result = model.invoke("what is the capital of India")
print(result) # this will give content[actual answer] + many keyword argument [metadata]
print(result.content) # only for the result