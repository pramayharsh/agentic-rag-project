import os
import re
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from src.tools import search_pdf, search_web # Ensure both are imported
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

def call_model(state: State):
    system_prompt = SystemMessage(
        content=(
            "You are a 2025 AI Research Assistant. "
            "1. To search PDF: SEARCH_PDF: [query]\n"
            "2. To search Web: SEARCH_WEB: [query]\n"
            "If the question is about current events (like the President), you MUST use SEARCH_WEB.\n"
            "IMPORTANT: Write the search command on its own line. Do not say 'I am searching'. Just write the command."
        )
    )
    # Filter messages to avoid empty inputs
    messages = [m for m in state["messages"] if m.content.strip()]
    response = llm.invoke([system_prompt] + messages)
    return {"messages": [response]}

def execute_tools(state: State):
    last_message = state["messages"][-1]
    content = last_message.content
    
    # NEW: Better Regex that catches "SEARCH_WEB" or "SEARCH_WEB:" or "search_web"
    pdf_match = re.search(r"SEARCH_PDF:?\s*(.*)", content, re.IGNORECASE)
    web_match = re.search(r"SEARCH_WEB:?\s*(.*)", content, re.IGNORECASE)

    if web_match:
        query = web_match.group(1).split('\n')[0].strip().replace('"', '')
        if not query: query = "current president of the United States" # Fallback
        
        print(f"--- Action: Searching Web for '{query}' ---")
        web_result = search_web.invoke({"query": query})
        
        # We print a snippet to the console so you can see it's working
        print(f"--- Web Data Retrieved (first 100 chars): {web_result[:100]} ---")
        
        return {"messages": [HumanMessage(content=f"ACTUAL CURRENT WEB DATA: {web_result}")]}

    if pdf_match:
        query = pdf_match.group(1).split('\n')[0].strip().replace('"', '')
        print(f"--- Action: Searching PDF for '{query}' ---")
        result = search_pdf.invoke({"query": query})
        
        if len(result) < 150:
            print("--- Grade: PDF insufficient. Switching to Web ---")
            web_result = search_web.invoke({"query": query})
            return {"messages": [HumanMessage(content=f"ACTUAL CURRENT WEB DATA: {web_result}")]}
        
        return {"messages": [HumanMessage(content=f"PDF Results: {result}")]}
    
    return {"messages": []}

def should_continue(state: State):
    last_message = state["messages"][-1]
    content = last_message.content.upper()
    if "SEARCH_PDF" in content or "SEARCH_WEB" in content:
        return "tools"
    return END

    
# Build the Graph
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()

if __name__ == "__main__":
    print("Agentic RAG Ready.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit": break
        
        try:
            # ADDED: recursion_limit=5 prevents infinite loops!
            # It will stop after 5 turns even if the agent is confused.
            state = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 5} 
            )
            print(f"Assistant: {state['messages'][-1].content}")
        except Exception as e:
            print(f"Loop ended or Error: {e}")