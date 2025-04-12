from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# huggin face end point whe you want to HF api [Connects to a model hosted on Hugging Face using their API.]

from dotenv import load_dotenv

load_dotenv()  #HUGGINGFACEHUB_API_TOKEN become available in your Python script.

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", #The model's path on Hugging Face
    task="text-generation", # Type of task the model performs
    max_new_tokens=10
)

model = ChatHuggingFace(llm=llm) #This wraps the llm (TinyLlama model) into a chat interface so you can interact with it using prompts like a chatbot.

#result = model.invoke("what is the capital of India?")  #Used to call the model with a prompt.

# print(result.content)

# result1 = model.invoke("What is cybersecurity?")
# print(result1.content)

result1 = model.invoke([{"role": "user", "content": "What is cybersecurity?"}])
print(result1.content)
