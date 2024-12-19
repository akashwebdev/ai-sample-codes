import os

from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import argparse

load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument('--task', default="return list of numbers")
parser.add_argument('--lang', default="python")
args = parser.parse_args()

model = 'llama3-8b-8192'
# Initialize Groq Langchain chat object and conversation
llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"], 
        model_name=model
)

# Define a prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will do {task}",
    input_variables=["language", "task"],
    output_key="code"
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n {code}",
    input_variables=["language", "code"],
    output_key="test"
)

# Create individual runnable components
code_runnable = code_prompt | llm | StrOutputParser()
test_runnable = test_prompt | llm | StrOutputParser()

# Define the sequential chain explicitly
def sequential_chain(inputs):
    # Generate the code
    code_output = code_runnable.invoke(inputs)
    
    # Prepare inputs for the test generation
    test_inputs = {
        "language": inputs["language"],
        "code": code_output
    }
    
    # Generate the test
    test_output = test_runnable.invoke(test_inputs)
    return {
        "code": code_output,
        "test": test_output
    }

inputs = {
    "language": args.lang,
    "task": args.task
}

# Invoke the chain and fetch the result
result = sequential_chain(inputs)

# Print the results
print("Generated Code:")
print(result["code"])
print("\nGenerated Test:")
print(result["test"])
