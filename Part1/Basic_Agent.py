# Basic_Agent.py

# Langchain imports
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')


# Load environment variables (e.g., OpenAI API keys) from a .env file located on the local system
env_path = r'C:\Users\user\Desktop\Coding\.env'
load_dotenv(env_path)

## Case 1: Basic Prompt Template and LLM invocation

# Define a simple prompt template
prompt_template = PromptTemplate(
    input_variables=["name"],
    template="Hello, {name}! How can I help you today?"
)

# Generate the prompt by filling in the variable
formatted_prompt = prompt_template.format(name="David")
print(formatted_prompt)  # Output: Hello, John! How can I help you today?

# Initialize the OpenAI Chat model
chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# Send a message to the model and get the response
response = chat_model.invoke("What is Capital of USA?")
print(response)  # Output: The capital of the United States of America is Washington, D.C
print("=========================")


## Case 2: Using Prompt Template with LLMChain

# Initialize the OpenAI Chat model
llm = ChatOpenAI(model="gpt-3.5-turbo")

learn_template = """
I want you to act as a consultant for a AI training
Return a list of topics and why it is important to learn in given area of AI
The description should be relevant to recent advancement in AI
What are some good topics to learn in {AI_topic}
"""

prompt_template = PromptTemplate(
    input_variables=["AI_topic"],
    template=learn_template,
)

description = "Please suggest me topics to learn in Deep learning"
prompt_template.format(AI_topic=description)

prompt = prompt_template.format(AI_topic=description)
response = llm.invoke(prompt)
print(response)
print("=========================")


## Case 3: Using Agents with tools (e.g., web search duckduckgo)

# Define a simple tool
def my_tool_function(query: str) -> str:
    return f"Tool response: {query}"

# Creating tool from function
my_tool = Tool.from_function(func=my_tool_function, name="simple_tool", description="A simple tool")
ddg_search = DuckDuckGoSearchRun()

# Initialize the LLM and the agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [ddg_search, my_tool]

system_prompt = "You are a helpful assistant."
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)
# Use the agent
response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like today in London?"}]})
print(response)

prompt_template = "Summarize the following content: {content}"
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

summarize_tool = Tool.from_function(
    func=lambda content: llm.invoke(prompt_template.format(content=content)),
    name="Summarizer",
    description="Summarizes a web page"
)

tools = [ddg_search, summarize_tool]

prompt = """Please tell me how to become a good footballer"""
system_prompt = "You are a helpful assistant."
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)
prompt = "Please tell me how to become a good footballer"
response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
print(response)

print("=========================")
