from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "give me the name, age and city of a fictional person \n{format_instruction}", # format_instruction means we are giving additional instruction with our prompt about what type of output do we want
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}  # partial var coz its not field at run time its filled before runtime.

)

prompt = template.format()
# print(prompt)  
#give me the name, age and city of a fictional person 
#Return a JSON object.

# we can also write it manually in prompt but above method is correct
result = llm.invoke(prompt)
print(result.content )


# we have to parse the response form llm

final_result = parser.parse(result.content)
print(final_result)
print(type(final_result)) # python json data objects ko dictionary ki tarah treat karta h


# we can also write above code with help of chains
chain = template | llm | parser
result = chain.invoke({})
print(result)