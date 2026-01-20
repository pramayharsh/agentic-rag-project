import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load variables from .env
load_dotenv()

def test_groq():
    try:
        # Initializing with the latest Llama 3.3 model
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        response = llm.invoke("Hello! Are you ready to help me build an Agentic RAG?")
        print("Groq Response:")
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_groq()