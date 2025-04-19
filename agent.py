import langchain 
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import Tool
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import langgraph
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages
from langgraph.graph import Graph, END
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
import tempfile
import subprocess
import graphviz

load_dotenv("secrets.env")

# Access the variables
api_key = os.getenv("API_KEY")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = api_key

#Initalizing the LLM
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Define the state for our graph
class AgentState(TypedDict):
    messages: List[str]
    brief: str
    budget: int
    priority: str
    analysis: Dict[str, Any]
    rethink: Dict[str, Any]
    iteration: int

# Creating state class for the agent
class State:
    def __init__(self, brief: str, budget: int, priority: str = "balanced"):
        self.messages = []
        self.brief = brief
        self.budget = budget
        self.priority = priority

    #Decrease the budget by the amount of the item
    def decrease_budget(self, amount: int):
        self.budget -= amount
        return self.budget
    
    #Increase the budget by the amount of the item
    def increase_budget(self, amount: int):
        self.budget += amount
        return self.budget

def analyze_brief_tool(state: AgentState) -> AgentState:
    """
    A tool node that analyzes a project brief and breaks down required items into categories.
    """
    # Increment iteration count
    state['iteration'] += 1
    
    prompt = f"""
    You are a helpful assistant that analyzes project briefs and breaks down required items into categories.
    Analyze this project brief and break down the required items into categories.
    Consider the priority ({state['priority']}) and budget (${state['budget']}).
    
    Brief: {state['brief']}
    
    Return ONLY a valid JSON object in this exact format, with no additional text:
    {{
        "categories": [
            {{
                "name": "category_name",
                "items": [
                    {{
                        "name": "item_name",
                        "estimated_cost": cost_in_dollars,
                        "priority": "high/medium/low",
                        "notes": "any relevant notes"
                    }}
                ]
            }}
        ],
        "total_estimated_cost": total_cost,
        "budget_status": "within_budget/over_budget/under_budget"
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean the response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Validate JSON structure
        import json
        analysis = json.loads(response_text)
        
        if not isinstance(analysis, dict):
            raise ValueError("Analysis is not a dictionary")
        if "categories" not in analysis:
            raise ValueError("Missing 'categories' in analysis")
        if "total_estimated_cost" not in analysis:
            raise ValueError("Missing 'total_estimated_cost' in analysis")
        if "budget_status" not in analysis:
            raise ValueError("Missing 'budget_status' in analysis")
        
        # Update state with analysis
        state['analysis'] = analysis
        state['messages'].append(f"Brief analyzed. Total estimated cost: ${analysis['total_estimated_cost']}")
        state['messages'].append(f"Budget status: {analysis['budget_status']}")
        
        return state
        
    except Exception as e:
        state['analysis'] = {
            "error": str(e),
            "categories": [],
            "total_estimated_cost": 0,
            "budget_status": "error"
        }
        state['messages'].append(f"Error analyzing brief: {str(e)}")
        return state

def rethink_analysis(state: AgentState) -> AgentState:
    """
    A tool node that rethinks the analysis to ensure it aligns with priorities and brief.
    """
    prompt = f"""
    You are a helpful assistant that reviews project analyses to ensure they align with priorities and requirements.
    
    Review this analysis for the following project:
    Brief: {state['brief']}
    Priority: {state['priority']}
    Budget: ${state['budget']}
    
    Current Analysis:
    {state['analysis']}
    
    Consider:
    1. Does the budget allocation match the priority? (e.g., if priority is 'performance', are high-performance components prioritized?)
    2. Are there any items that don't align with the brief?
    3. Could the budget be better allocated to match the priorities?
    
    Return ONLY a valid JSON object in this exact format, with no additional text:
    {{
        "alignment_score": score_from_0_to_100,
        "recommendations": [
            {{
                "category": "category_name",
                "item": "item_name",
                "suggestion": "specific suggestion for improvement",
                "impact": "expected impact of the change"
            }}
        ],
        "overall_feedback": "brief summary of how well the analysis aligns with priorities"
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean the response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse the response
        import json
        rethink = json.loads(response_text)
        
        # Update state with rethink results
        state['rethink'] = rethink
        state['messages'].append(f"Analysis reviewed. Alignment score: {rethink['alignment_score']}/100")
        state['messages'].append(f"Overall feedback: {rethink['overall_feedback']}")
        
        return state
        
    except Exception as e:
        state['rethink'] = {
            "error": str(e),
            "alignment_score": 0,
            "recommendations": [],
            "overall_feedback": "Error during review"
        }
        state['messages'].append(f"Error during review: {str(e)}")
        return state

def should_retry_analysis(state: AgentState) -> str:
    """
    Determines if the analysis should be retried based on alignment score and iteration count.
    Returns the next node to execute.
    """
    if state['rethink']['alignment_score'] > 90 or state['iteration'] >= 2:
        return "end"
    return "analyze"

class Agent:
    def __init__(self, brief: str, budget: int, priority: str = "balanced"):
        """
        Initialize an agent with project details.
        """
        if priority not in ["balanced", "performance", "aesthetic"]:
            raise ValueError("Priority must be 'balanced', 'performance', or 'aesthetic'")
            
        self.state = State(brief=brief, budget=budget, priority=priority)
        self.messages = []
        self.llm = llm
        
        # Create the graph
        self.workflow = Graph()
        
        # Add nodes
        self.workflow.add_node("analyze", analyze_brief_tool)
        self.workflow.add_node("rethink", rethink_analysis)
        self.workflow.add_node("end", lambda x: x)  # End node
        
        # Add edges
        self.workflow.add_edge("analyze", "rethink")
        
        # Add conditional edges for rethink
        self.workflow.add_conditional_edges(
            "rethink",
            should_retry_analysis,
            {
                "analyze": "analyze",
                "end": "end"
            }
        )
        
        # Set entry point
        self.workflow.set_entry_point("analyze")
        
        # Set end point
        self.workflow.set_finish_point("end")
        
        # Compile
        self.app = self.workflow.compile()

    def print_graph_structure(self):
        """Print the structure of the workflow graph."""
        # Create a networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_name in self.workflow.nodes:
            G.add_node(node_name)
        
        # Add edges
        for edge in self.workflow.edges:
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1])
        
        # Set the style
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Create positions dynamically
        nodes = list(G.nodes())
        pos = {}
        for i, node in enumerate(nodes):
            pos[node] = (i, 0)
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold',
                arrows=True, arrowsize=20)
        
        # Add edge labels dynamically
        edge_labels = {}
        for edge in G.edges():
            if edge[0] == '__start__':
                edge_labels[edge] = 'start'
            elif edge[0] == 'rethink' and edge[1] == 'analyze':
                edge_labels[edge] = 'retry'
            elif edge[0] == 'rethink' and edge[1] == 'end':
                edge_labels[edge] = 'end'
            else:
                edge_labels[edge] = edge[0]
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        # Show the plot
        plt.show()

    def add_message(self, message: str):
        """Add a message to the agent's message history."""
        self.messages.append(message)
        
    def get_messages(self) -> list:
        """Get all messages in the agent's history."""
        return self.messages
        
    def get_brief(self) -> str:
        """Get the project brief."""
        return self.state.brief
        
    def get_budget(self) -> int:
        """Get the current budget."""
        return self.state.budget
        
    def get_priority(self) -> str:
        """Get the project priority."""
        return self.state.priority
        
    def update_budget(self, amount: int):
        """Update the budget by adding or subtracting an amount."""
        if amount < 0:
            self.state.decrease_budget(abs(amount))
        else:
            self.state.increase_budget(amount)
            
    def set_priority(self, priority: str):
        """Update the project priority."""
        if priority not in ["balanced", "performance", "aesthetic"]:
            raise ValueError("Priority must be 'balanced', 'performance', or 'aesthetic'")
        self.state.priority = priority

    def analyze_brief(self) -> dict:
        """
        Run the analysis workflow and return the results.
        """
        # Prepare initial state
        initial_state = AgentState(
            messages=self.messages,
            brief=self.state.brief,
            budget=self.state.budget,
            priority=self.state.priority,
            analysis={},
            rethink={},
            iteration=0
        )
        
        # Run the workflow
        try:
            print("Starting analysis workflow...")
            final_state = self.app.invoke(initial_state)
            
            if final_state is None:
                print("Warning: Graph execution returned None")
                # Try to get more information about the state
                try:
                    print(f"Current state: {initial_state}")
                    print(f"Graph nodes: {self.workflow.nodes}")
                    print(f"Graph edges: {self.workflow.edges}")
                except Exception as e:
                    print(f"Error getting graph info: {str(e)}")
                raise ValueError("Graph execution failed - final state is None")
                
            # Update agent state
            self.messages = final_state.get('messages', [])
            
            return {
                'analysis': final_state.get('analysis', {}),
                'rethink': final_state.get('rethink', {}),
                'iterations': final_state.get('iteration', 0)
            }
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            self.messages.append(f"Error during analysis: {str(e)}")
            return {
                'analysis': {'error': str(e)},
                'rethink': {'error': str(e)},
                'iterations': 0
            }

if __name__ == "__main__":
    # Create a new agent with your project details
    agent = Agent(
        brief="Build a high-performance Gaming Desktop",
        budget=1000,
        priority="performance"
    )
    
    # Print the graph structure
    agent.print_graph_structure()
    
    # Add some initial messages
    agent.add_message("Starting project initialization")
    agent.add_message("Analyzing requirements")
    
    # Analyze the brief
    results = agent.analyze_brief()
    analysis = results['analysis']
    rethink = results['rethink']
    iterations = results['iterations']
    
    # Print the analysis
    print("\nProject Analysis:")
    if 'error' in analysis:
        print(f"Error during analysis: {analysis['error']}")
    else:
        total_cost = analysis.get('total_estimated_cost', 'N/A')
        budget_status = analysis.get('budget_status', 'N/A')
        print(f"Total Estimated Cost: ${total_cost}")
        print(f"Budget Status: {budget_status}")
        print(f"Number of Iterations: {iterations}")
        
        print("\nCategories and Items:")
        for category in analysis.get('categories', []):
            print(f"\n{category.get('name', 'Unnamed Category')}:")
            for item in category.get('items', []):
                print(f"  - {item.get('name', 'Unnamed Item')} (${item.get('estimated_cost', 'N/A')})")
                print(f"    Priority: {item.get('priority', 'N/A')}")
                print(f"    Notes: {item.get('notes', 'N/A')}")
    
    # Print the rethink results
    print("\nAnalysis Review:")
    if 'error' in rethink:
        print(f"Error during review: {rethink['error']}")
    else:
        print(f"Alignment Score: {rethink.get('alignment_score', 'N/A')}/100")
        print(f"Overall Feedback: {rethink.get('overall_feedback', 'N/A')}")
        
        if rethink.get('recommendations'):
            print("\nRecommendations:")
            for rec in rethink['recommendations']:
                print(f"\nCategory: {rec.get('category', 'N/A')}")
                print(f"Item: {rec.get('item', 'N/A')}")
                print(f"Suggestion: {rec.get('suggestion', 'N/A')}")
                print(f"Expected Impact: {rec.get('impact', 'N/A')}")