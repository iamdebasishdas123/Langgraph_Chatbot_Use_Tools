from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Load environment variables
load_dotenv()

# Initialize Groq chat model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1000
)

# Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize tools with custom configurations
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3))
tavily = TavilySearchResults(max_results=5)

tools = [tavily, wikipedia, arxiv]

# Create agent graph
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build the workflow graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Define edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot")

# Compile the graph
agent = graph_builder.compile()

# Example usage
def run_agent(query: str):
    response = agent.invoke({
        "messages": [("user", query)]
    })
    return response["messages"][-1].content

# Test the agent
if __name__ == "__main__":
    query = "Explain attention mechanism in neural networks with recent research developments"
    result = run_agent(query)
    print(result)
