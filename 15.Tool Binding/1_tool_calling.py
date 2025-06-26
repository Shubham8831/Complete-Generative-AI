# Import required libraries
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get your GROQ API key from environment variables
key = os.getenv("GROQ_API_KEY")

# Import Groq Chat model from langchain_groq package
from langchain_groq import ChatGroq

# Import decorator to define tools
from langchain_core.tools import tool

# Import message classes
from langchain_core.messages import HumanMessage, ToolMessage




# ---------------------- Tool Definition -----------------------

# Create a simple tool using @tool decorator
# This tool takes two integers and returns their product
@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a * b

# Check metadata of the tool
print(multiply.name)         # Will print: multiply
print(multiply.description)  # Will print the docstring as description
print(multiply.args)         # Will print argument types like {"a": int, "b": int}




# ---------------------- LLM Initialization -----------------------

# Instantiate the Groq-hosted LLaMA 3 model using your API key
llm = ChatGroq(model="llama3-70b-8192", api_key=key)

# Bind the tool to the LLM
# This tells the model it is allowed to call this tool if needed
llm_with_tools = llm.bind_tools([multiply])





# ---------------------- Step 1: User Sends a Message -----------------------

# Simulate a user message (HumanMessage)
# This acts like a message in a chat history
messages = [HumanMessage(content="can you multiply 3 with 10")]





# ---------------------- Step 2: Let LLM Decide What to Do -----------------------

# Pass the user message to the model
# The model decides whether it should call a tool based on the prompt
ai_response = llm_with_tools.invoke(messages)

# Add the AI’s response (usually an empty message with a tool call) to the conversation history
messages.append(ai_response)




# ---------------------- Step 3: Perform Tool Execution -----------------------

# Check if the AI actually made a tool call
# For this example, we assume it correctly calls the `multiply` tool
tool_call = ai_response.tool_calls[0]  # LLM may generate multiple tool calls — we take the first one

# tool_call.args contains the input parameters as a dictionary: {"a": 3, "b": 10}
# Now we execute the tool function using those arguments
tool_output = multiply.invoke(tool_call["args"])

# Wrap the tool's output (e.g., 30) as a ToolMessage
# This tells the LLM: "Here is the result of the tool you asked for"
tool_result_message = ToolMessage(
    tool_call_id=tool_call["id"],  # This links the result to the specific tool call
    content=str(tool_output)    # Convert the output (int) to string for the message content
)

# Add the tool result message to the message history
messages.append(tool_result_message)



# ---------------------- Step 4: Let LLM Respond Based on Tool Output -----------------------

# Pass the full message history to the LLM again
# Now that it has both the user's question and the tool result, it will generate a final answer
final_response = llm_with_tools.invoke(messages)

# Print the final answer from the LLM
print(final_response.content)
