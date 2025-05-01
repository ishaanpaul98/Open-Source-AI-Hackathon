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

# Load environment variables from secrets.env file if it exists
if os.path.exists('secrets.env'):
    load_dotenv('secrets.env')

# Try to get API key from environment variables (works for both local .env and GitHub Secrets)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in secrets.env or GitHub Secrets.")

# Set API key in environment for Google services
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
    # Create a new state dictionary with default values
    new_state = {
        'messages': state.get('messages', []),
        'brief': state.get('brief', ''),
        'budget': state.get('budget', 0),
        'priority': state.get('priority', 'balanced'),
        'analysis': {},
        'rethink': {},
        'iteration': state.get('iteration', 0) + 1
    }
    
    prompt = f"""
    You are a helpful assistant that analyzes project briefs and breaks down required items into categories.
    Analyze this project brief and break down the required items into categories.
    Consider the priority ({new_state['priority']}) and budget (${new_state['budget']}).
    
    Brief: {new_state['brief']}
    
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
        new_state['analysis'] = analysis
        new_state['messages'] = new_state['messages'] + [
            f"Brief analyzed. Total estimated cost: ${analysis['total_estimated_cost']}",
            f"Budget status: {analysis['budget_status']}"
        ]
        
        return new_state
        
    except Exception as e:
        error_msg = str(e)
        new_state['analysis'] = {
            "error": error_msg,
            "categories": [],
            "total_estimated_cost": 0,
            "budget_status": "error"
        }
        new_state['messages'] = new_state['messages'] + [f"Error analyzing brief: {error_msg}"]
        return new_state

def get_user_feedback(state: AgentState) -> AgentState:
    """
    A tool node that gets feedback from the user about the analysis.
    """
    # Create a new state dictionary with default values
    new_state = {
        'messages': state.get('messages', []),
        'brief': state.get('brief', ''),
        'budget': state.get('budget', 0),
        'priority': state.get('priority', 'balanced'),
        'analysis': state.get('analysis', {}),
        'rethink': state.get('rethink', {}),
        'iteration': state.get('iteration', 0),
        'user_feedback': None,
        'next_node': 'end'  # Default to end node
    }
    
    # Check if there's feedback in the state
    if 'user_feedback' in state and state['user_feedback']:
        feedback = state['user_feedback']
        new_state['user_feedback'] = feedback
        
        # Update the brief with the feedback
        new_state['brief'] = f"{new_state['brief']}\n\nUser Feedback: {feedback}"
        
        new_state['messages'] = new_state['messages'] + [
            f"User feedback received: {feedback}",
            "Brief updated with user feedback"
        ]
        new_state['next_node'] = 'analyze'  # Go back to analyze with updated brief
    else:
        new_state['messages'] = new_state['messages'] + ["No user feedback provided"]
        new_state['next_node'] = 'end'  # End the workflow
    
    return new_state

def review_and_decide(state: AgentState) -> AgentState:
    """
    A tool node that reviews the analysis and decides whether to retry or end.
    """
    # Create a new state dictionary with default values
    new_state = {
        'messages': state.get('messages', []),
        'brief': state.get('brief', ''),
        'budget': state.get('budget', 0),
        'priority': state.get('priority', 'balanced'),
        'analysis': state.get('analysis', {}),
        'rethink': {},
        'iteration': state.get('iteration', 0),
        'user_feedback': state.get('user_feedback', None)
    }
    
    # Check if analysis exists and is valid
    if not isinstance(new_state['analysis'], dict) or 'error' in new_state['analysis']:
        new_state['rethink'] = {
            "error": "Invalid or missing analysis",
            "alignment_score": 0,
            "recommendations": [],
            "overall_feedback": "Error during review",
            "should_retry": False
        }
        new_state['messages'] = new_state['messages'] + ["Error: Invalid or missing analysis"]
        new_state['next_node'] = 'end'
        return new_state
    
    prompt = f"""
    You are a helpful assistant that reviews project analyses to ensure they align with priorities and requirements.
    
    Review this analysis for the following project:
    Brief: {new_state['brief']}
    Priority: {new_state['priority']}
    Budget: ${new_state['budget']}
    
    Current Analysis:
    {new_state['analysis']}
    
    {f"User Feedback: {new_state['user_feedback']}" if new_state['user_feedback'] else ""}
    
    Consider:
    1. Does the budget allocation match the priority? (e.g., if priority is 'performance', are high-performance components prioritized?)
    2. Are there any items that don't align with the brief?
    3. Could the budget be better allocated to match the priorities?
    4. {f"Address the user's feedback: {new_state['user_feedback']}" if new_state['user_feedback'] else ""}
    
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
        "overall_feedback": "brief summary of how well the analysis aligns with priorities",
        "should_retry": boolean
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
        review = json.loads(response_text)
        
        # Update state with review results
        new_state['rethink'] = review
        new_state['messages'] = new_state['messages'] + [
            f"Analysis reviewed. Alignment score: {review['alignment_score']}/100",
            f"Overall feedback: {review['overall_feedback']}"
        ]
        
        # Decide next node based on review and iteration count
        if review['alignment_score'] <= 70 and new_state['iteration'] < 3:
            print(f"Alignment score is {review['alignment_score']}, retrying analysis")
            new_state['next_node'] = 'analyze'
        else:
            new_state['next_node'] = 'feedback'  # Go to feedback node
        
        return new_state
        
    except Exception as e:
        error_msg = str(e)
        new_state['rethink'] = {
            "error": error_msg,
            "alignment_score": 0,
            "recommendations": [],
            "overall_feedback": "Error during review",
            "should_retry": False
        }
        new_state['messages'] = new_state['messages'] + [f"Error during review: {error_msg}"]
        new_state['next_node'] = 'end'
        return new_state

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
        self.workflow.add_node("review", review_and_decide)
        self.workflow.add_node("feedback", get_user_feedback)
        self.workflow.add_node("end", lambda x: x)  # End node
        
        # Add edges
        self.workflow.add_edge("analyze", "review")
        self.workflow.add_conditional_edges(
            "review",
            lambda x: x.get('next_node', 'end'),
            {
                "analyze": "analyze",
                "feedback": "feedback",
                "end": "end"
            }
        )
        self.workflow.add_conditional_edges(
            "feedback",
            lambda x: x.get('next_node', 'end'),
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

if __name__ == "__main__":
    # Create a new agent with your project details
    agent = Agent(
        brief="I want to create a home office. I want 2 monitors and a comfortable chair",
        budget=1500,
        priority="performance"
    )
    
    # Add some initial messages
    agent.add_message("Starting project initialization")
    agent.add_message("Analyzing requirements")
    
    # Prepare initial state
    initial_state = {
        'messages': agent.messages,
        'brief': agent.state.brief,
        'budget': agent.state.budget,
        'priority': agent.state.priority,
        'analysis': {},
        'rethink': {},
        'iteration': 0
    }
    
    # Run the workflow
    try:
        print("Starting analysis workflow...")
        
        # Ensure initial_state is a dictionary
        if not isinstance(initial_state, dict):
            raise ValueError("Initial state must be a dictionary")
        
        # Run the workflow with proper state handling
        def run_workflow(state):
            try:
                # Ensure state is a dictionary before passing to graph
                if not isinstance(state, dict):
                    state = {}
                
                # Run the workflow and get the result
                result = agent.app.invoke(state)
                
                # Ensure the result is a dictionary
                if not isinstance(result, dict):
                    print(f"Warning: Workflow returned non-dictionary result: {type(result)}")
                    return {
                        'messages': state.get('messages', []),
                        'brief': state.get('brief', ''),
                        'budget': state.get('budget', 0),
                        'priority': state.get('priority', 'balanced'),
                        'analysis': {},
                        'rethink': {},
                        'iteration': state.get('iteration', 0)
                    }
                
                return result
            except Exception as e:
                print(f"Error in workflow execution: {str(e)}")
                return {
                    'messages': state.get('messages', []) + [f"Error in workflow execution: {str(e)}"],
                    'brief': state.get('brief', ''),
                    'budget': state.get('budget', 0),
                    'priority': state.get('priority', 'balanced'),
                    'analysis': {'error': str(e)},
                    'rethink': {'error': str(e)},
                    'iteration': state.get('iteration', 0)
                }
        
        final_state = run_workflow(initial_state)
        
        # Update agent state
        agent.messages = final_state.get('messages', [])
        
        # Get results
        analysis = final_state.get('analysis', {})
        rethink = final_state.get('rethink', {})
        iterations = final_state.get('iteration', 0)
        
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
                    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        agent.messages.append(f"Error during analysis: {str(e)}")
