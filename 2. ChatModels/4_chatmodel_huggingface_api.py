from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()


# Choose a Together-supported model
llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)

# Ask a question
response = llm.invoke([
    HumanMessage(content="Explain quantum computing in simple terms.")
])

print(response.content)
