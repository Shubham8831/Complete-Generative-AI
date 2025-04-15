# topic 2/3 (chatbot.py)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

# SYSTEM MESSAGE : 
# sent in start,  Think of this as setting the rules or behavior for the AI.
#It tells the AI how to act or what role to play.
#Example: "You are a helpful assistant that answers like a teacher."
#So, it's like giving the AI some instructions before the conversation starts.

# AI MESSAGE : 
# This is the response from the AI.
# It answers the human message based on the system instructions.
# Example:"The capital of France is Paris."

# HUMAN MESSAGE : 
# This is the actual input or question from the user (you or anyone talking to the AI).
# It’s what the human says.
# Example:"What is the capital of France?"
# This is the prompt or message that AI will respond to.

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about langchian in 100 words")
] # this is like a chat history


result = model.invoke(messages)
messages.append(AIMessage(content = result.content)) # result puted into chat history

print(messages)