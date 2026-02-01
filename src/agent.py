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
    # We count how many times a search has happened in the message history
    search_count = sum(1 for m in state["messages"] if "RESULTS:" in m.content)

    system_prompt = SystemMessage(
        content=(
            "You are a strict research assistant. \n"
            f"Current search count: {search_count}\n"
            "INSTRUCTIONS:\n"
            "1. Search the Internal PDF ONCE using SEARCH_PDF: [query].\n"
            "2. If you already have 'INTERNAL PDF RESULTS' in your history, DO NOT search the PDF again.\n"
            "3. If the PDF results are missing the info, search the Web ONCE using SEARCH_WEB: [query].\n"
            "4. If you have any results (PDF or Web), STOP SEARCHING and provide the final answer.\n"
            "5. Never output a SEARCH command as part of a final answer."
        )
    )
    
    messages = state["messages"]
    # Ensure system prompt is always the first message and updated with search_count
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    response = llm.invoke([system_prompt] + filtered_messages)
    return {"messages": [response]}

def execute_tools(state: State):
    last_message = state["messages"][-1]
    content = last_message.content
    
    # Check if this exact query has already been searched to prevent loops
    past_queries = [m.content for m in state["messages"]]

    pdf_match = re.search(r"SEARCH_PDF:?\s*(.*)", content, re.IGNORECASE)
    web_match = re.search(r"SEARCH_WEB:?\s*(.*)", content, re.IGNORECASE)

    if pdf_match:
        query = pdf_match.group(1).split('\n')[0].strip().replace('"', '')
        # SAFETY: Stop nonsense searches
        if len(query) < 2 or "again" in query.lower():
            return {"messages": [HumanMessage(content="SYSTEM: Invalid search query. Please provide a final answer with what you have.")]}
        
        print(f"--- Action: Searching PDF for '{query}' ---")
        result = search_pdf.invoke({"query": query})
        return {"messages": [HumanMessage(content=f"INTERNAL PDF RESULTS: {result}")]}

    if web_match:
        query = web_match.group(1).split('\n')[0].strip().replace('"', '')
        # SAFETY: Stop nonsense searches
        if len(query) < 2 or "again" in query.lower():
             return {"messages": [HumanMessage(content="SYSTEM: Invalid search query. Please provide a final answer.")]}
             
        print(f"--- Action: Searching Web for '{query}' ---")
        web_result = search_web.invoke({"query": query})
        return {"messages": [HumanMessage(content=f"WEB SEARCH RESULTS: {web_result}")]}
    
    return {"messages": []}


def should_continue(state: State):
    last_message = state["messages"][-1]
    content = last_message.content
    
    # Logic to stop the loop:
    # 1. If the model didn't output a search command, stop.
    if "SEARCH_PDF" not in content.upper() and "SEARCH_WEB" not in content.upper():
        return END
    
    # 2. If we already have 2 or more results, force a stop to prevent infinite loops
    search_count = sum(1 for m in state["messages"] if "RESULTS:" in m.content)
    if search_count >= 2:
        return END
        
    return "tools"


    
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