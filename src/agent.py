import os
import re
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from src.tools import search_pdf
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 1. Initialize Groq (8B is faster and has higher free limits than 70B)
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# 2. The Agent Node
def call_model(state: State):
    system_prompt = SystemMessage(
        content=(
            "You are a research assistant. You have a search tool. "
            "If you need to look at the PDF, you MUST write: SEARCH: [query] "
            "If you have the information, provide a final answer based on the search results."
        )
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# 3. The Manual Tool Node (We link it back to your Gemini-powered search)
def execute_tools(state: State):
    last_message = state["messages"][-1]
    content = last_message.content
    
    match = re.search(r"SEARCH:\s*(.*)", content)
    if match:
        query = match.group(1).strip()
        print(f"--- Heartbeat 1: Starting search for '{query}' ---")
        
        try:
            # We call the tool
            result = search_pdf.invoke({"query": query})
            
            print(f"--- Heartbeat 2: Search complete! Found {len(result)} chars ---")
            
            return {"messages": [HumanMessage(content=f"Search Results: {result}")]}
        except Exception as e:
            print(f"--- Heartbeat ERROR: {e} ---")
            return {"messages": [HumanMessage(content="Search failed.")]}
    
    return {"messages": []}
    
# 4. Routing Logic
def should_continue(state: State):
    last_message = state["messages"][-1]
    if "SEARCH:" in last_message.content:
        return "tools"
    return END

# 5. Build the Graph
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

if __name__ == "__main__":
    print("Agentic RAG is ready (Hybrid Mode).")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit": break
        
        try:
            # We process the loop until we get a final answer
            final_state = app.invoke({"messages": [HumanMessage(content=user_input)]})
            
            # The last message in the conversation is the Assistant's final answer
            print(f"Assistant: {final_state['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {e}")