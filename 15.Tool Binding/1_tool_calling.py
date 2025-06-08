import os
os.environ["OPEN_API_KEY"] = ""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

# creating a tool
@tool
def multiply(a:int, b:int) -> int:
    """given 2 numbers a and b this tool returns their product"""
    return a*b

print(multiply.name) # this gives tool name
print(multiply.description) # this prints tool description
print(multiply.args) # prints argument
print(multiply.invoke({"a":2, "b":3}))

# tool binding 
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([multiply]) # give list of tools


query = HumanMessage("can you multiply 3 with 10") # make your querry as human message

messages = [query] 
result = llm_with_tools.invoke(messages) # this is will give ai message, with empty content

# we will add this result(ai message) in message list
messages.append(result)

#result.tool_calls[0] # there can be multiple tool calls but we have only one so [0] # this will give tool name, input args, unique id
# llms does not do the executions

tool_result = multiply.invoke(result.tool_call[0]) # This will give a tool message: (special message) we get this message when we execute a tool with the help of tool call

#now we will add this tool_result(tool message ) to message result (this will create a chat history)
# and now i will give all this message list (history) in llm

print(llm_with_tools.invoke(messages).content)