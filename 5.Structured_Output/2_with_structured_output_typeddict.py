from langchain_openai import ChatOpenAI  # Import ChatOpenAI from langchain_openai package
from dotenv import load_dotenv  # Import function to load environment variables
from typing import TypedDict, Annotated, Optional, Literal  # Import typing tools for structured schema

#TypedDict: A way to define a dictionary with specific key-value types.
#Annotated:Lets you add extra info or metadata to a type (used for tools like LangChain, Pydantic, etc.), LangChain uses this to understand how to describe or validate output fields.
#Optional: A type that can either be the given type or None.
#Literal: A type that only allows specific fixed values. [status: Literal["active", "inactive"]]


load_dotenv()  # Load API keys and environment variables from .env file

model = ChatOpenAI()  # Create a ChatOpenAI model instance (default: gpt-3.5-turbo)

# Define the structure of the expected output using TypedDict
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]  # List of main topics
    summary: Annotated[str, "A brief summary of the review"]  # Summary of the review
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]  # Sentiment label
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]  # List of positive points
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]  # List of negative points
    name: Annotated[Optional[str], "Write the name of the reviewer"]  # Reviewer's name

# Attach the structured schema to the model to expect structured output
structured_model = model.with_structured_output(Review)

# Provide a product review and ask the model to return structured data
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

print(result['name'])  # Print the name of the reviewer from the structured result
