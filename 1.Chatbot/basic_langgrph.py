# Import required libraries
from langchain_groq import ChatGroq  # Groq's chat model integration
from dotenv import load_dotenv  # For loading environment variables
from typing import Annotated, Literal  # Type hints
from typing_extensions import TypedDict  # For structured typing
from langgraph.graph import StateGraph, START, END  # Graph components
from langgraph.graph.message import add_messages  # Message handling

# Load environment variables from .env file
load_dotenv()

# Initialize Groq chat model with specific parameters
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Use Llama 3 8B model
    temperature=0.1,  # Controls randomness (0=deterministic, 1=creative)
    max_tokens=1000  # Maximum response length
)

# Define the conversation state structure
class State(TypedDict):
    """
    Represents the chatbot's conversation state:
    - messages: Chat history with annotations for LangGraph
    - name: Placeholder for future state tracking (sentiment, etc.)
    """
    messages: Annotated[list, add_messages]  # Annotated message history
    name: Literal['positive', 'negative']  # Can track conversation sentiment

# Initialize the state graph with our State structure
graph_builder = StateGraph(State)

# Define the main chatbot processing node
def chatbot(state: State):
    """Processes incoming messages using the Groq model"""
    return {"messages": [llm.invoke(state["messages"])]}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Set up the conversation flow:
# START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")  # Initial connection
graph_builder.add_edge("chatbot", END)    # Final connection

# Compile the graph into an executable workflow
graph = graph_builder.compile()

# Example usage:
initial_state = {
    "messages": [("user", "Hello")],
    "name": "positive"
}
response = graph.invoke(initial_state)

print(response)

print(response["messages"][1].content)
