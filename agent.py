import langchain 
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import Tool
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import langgraph
from dotenv import load_dotenv

load_dotenv("secrets.env")

# Access the variables
api_key = os.getenv("API_KEY")

from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
  project_name="hackathon-app", # Default is 'default'
  auto_instrument=True # See 'Trace all calls made to a library' below
)
tracer = tracer_provider.get_tracer(__name__)

class Item:
    def __init__(self, name: str, quantity: int):
        self.name = name
        self.quantity = quantity
    
    def increase_quantity(self, amount: int):
        self.quantity += amount
        return self.quantity
    
    def decrease_quantity(self, amount: int):
        self.quantity -= amount
        return self.quantity

