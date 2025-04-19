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

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = api_key
from phoenix.otel import register

#Initalizing the LLM
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# configure the Phoenix tracer
tracer_provider = register(
  project_name="hackathon-app", # Default is 'default'
  auto_instrument=True # See 'Trace all calls made to a library' below
)
tracer = tracer_provider.get_tracer(__name__)

#Creating a class for the items in the grocery list
class Item:
    def __init__(self, name: str, quantity: int, days_to_expire: int):
        self.name = name
        self.quantity = quantity
        self.days_to_expire = days_to_expire
    
    def increase_quantity(self, amount: int):
        self.quantity += amount
        return self.quantity
    
    def decrease_quantity(self, amount: int):
        self.quantity -= amount
        return self.quantity
    
    def update_days_to_expire(self, days: int):
        self.days_to_expire = days
        return self.days_to_expire
    
    def get_item_info(self):
        return f"Item: {self.name}, Quantity: {self.quantity}, Days to Expire: {self.days_to_expire}"

#Creating a class for the grocery list
class GroceryList:
    def __init__(self):
        self.items = []
    
    def add_item(self, item: Item):
        self.items.append(item)

    def remove_item(self, item: Item):
        self.items.remove(item)

    def get_grocery_list(self):
        return self.items

    def get_item_count(self):
        return len(self.items)
    
    def get_item_by_name(self, name: str):
        for item in self.items:
            if item.name == name:
                return item 
        return None
    
    def get_items_by_days_to_expire(self, days: int):
        return [item for item in self.items if item.days_to_expire <= days]
    
    
