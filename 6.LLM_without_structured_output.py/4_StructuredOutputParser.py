from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # its umbrela library core has important reusable componenets.
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7
)

schema = [
    ResponseSchema(name = 'fact_1', description='fact 1 about the topic'),
    ResponseSchema(name = 'fact_2', description='fact 2 about the topic'),
    ResponseSchema(name = 'fact_3', description='fact 3 about the topic'),
] # this is structure who we would like to get output 

parser = StructuredOutputParser.from_response_schemas(schema) # sending schema in parser

template = PromptTemplate(
    template = 'give 3 facts about {topic} \n {format_instruction}', #{format_instruction} adding additional info about how to get output
    input_variables=['topic'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'balck hole'})
# print(prompt)
result = llm.invoke(prompt)
print(result.content)


# #transforming unstructured model output into structured data for easier access and use.
final_result = parser.parse(result.content)
print(final_result)


# we  can also use chains to do it