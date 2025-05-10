from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough
load_dotenv()

model =OllamaLLM(
    model = "gemma3:1b",
    temperature=0.5)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)

def length_fn(text):
    return len(text.split())

joke_chian = RunnableSequence(prompt, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length': RunnableLambda(length_fn)
})

final_chain = RunnableSequence(joke_chian, parallel_chain)
print(final_chain.invoke({'topic': 'human'}))