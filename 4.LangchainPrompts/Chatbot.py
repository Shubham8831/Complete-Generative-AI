# #1 making first chatbot:
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm = llm)


# while True:
#     user_input = input("you : ")
#     if user_input=="exit":
#         break
#     result = model.invoke(user_input)
#     print("AI : ",result.content)

# #2 making first chatbot: it was having problem of keeping context of previous messages so we used chat_history for solving this problem

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm = llm)

# chat_history = []
# while True:
#     user_input = input("you : ")
#     chat_history.append(user_input)
#     if user_input=="exit":
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(result.content)
#     print("AI : ",result.content)


# this above chat history has a problem: we don't have information about message( sent by whome Me or Ai)
# future conversation will have a problem so store message and who sent it. (best is to mentain the dictionary )

#3 chatbot with messages: refere messages.py

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

chat_history = [
    SystemMessage(content="You are a helpful assistant")
]

while True:
    user_input = input("you : ")
    chat_history.append(HumanMessage(content = user_input)) # ADDING TO CHAT HISTORY
    if user_input=="exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI : ",result.content)