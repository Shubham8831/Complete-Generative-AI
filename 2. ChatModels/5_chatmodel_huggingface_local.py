# Importing necessary classes from langchain_huggingface package
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Create a language model pipeline from Hugging Face using a specific model ID
llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # The name of the model you want to use (from Hugging Face)
    task="text-generation",  # The type of task the model should perform (in this case, generating text)
    
    # Extra settings to control the model's behavior
    pipeline_kwargs=dict(
        temperature = 0.5,  # Controls randomness. Lower = more focused, higher = more creative
        max_new_tokens = 100  # Maximum number of words/tokens the model can generate in response
    )
)

# Wrap the model (llm) into a chat-style interface so we can talk to it like a chatbot
model = ChatHuggingFace(llm = llm)

# Send a question to the model and store its response in a variable
result = model.invoke("what is the capital of India?")  # Asks the model this question

# Print the model's response to the question
print(result.content)

# (Optional) Another example query, currently commented out
# result1 = model.invoke("What is cybersecurity?")  # You can ask another question like this
# print(result1.content)  # And print that answer too
