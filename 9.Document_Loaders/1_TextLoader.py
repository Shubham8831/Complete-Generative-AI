from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

model = OllamaLLM(model = 'gemma3:1b')

parser = StrOutputParser()

loader = TextLoader(file_path='C:\\Users\\shubu\\Desktop\\GenAI\\9.Document_Loaders\\text_file.txt' )
doc = loader.load()

prompt = PromptTemplate(
    template="write a summary of:  {doc}.",
    input_variables=['doc']
)
chain = prompt | model | parser

print(chain.invoke({"doc": doc[0].page_content}))