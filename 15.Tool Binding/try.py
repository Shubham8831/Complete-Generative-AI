import openai

# Set your Together.ai API key
openai.api_key = "5d50fd339e710b9e8eb7cc862c19415c7fb7a64dda60364dafe7800b65c0cb30"
openai.api_base = "https://api.together.xyz/v1"

# Tool: Simple multiplication
def multiplication_tool(a: float, b: float) -> float:
    return a * b

# Agent logic
def agent(prompt: str):
    if "multiply" in prompt.lower():
        # Try to extract numbers from prompt
        import re
        numbers = list(map(float, re.findall(r"\d+(?:\.\d+)?", prompt)))
        if len(numbers) >= 2:
            result = multiplication_tool(numbers[0], numbers[1])
            return f"The result of multiplying {numbers[0]} and {numbers[1]} is {result}."
        else:
            return "Please provide two numbers to multiply."

    # Fallback to LLM for general questions
    response = openai.ChatCompletion.create(
        model="togethercomputer/llama-2-70b-chat",  # or another available model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response['choices'][0]['message']['content']

# Example Usage
if __name__ == "__main__":
    while True:
        user_input = input("Ask something (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        print(agent(user_input))
