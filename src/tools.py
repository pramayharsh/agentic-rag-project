import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool

load_dotenv()

# Initialize embeddings with REST transport
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    transport="rest"
)

@tool
def search_pdf(query: str):
    """Searches the PDF database using FAISS for relevant information."""
    print(f"--- FAISS Search for: {query} ---")
    
    # Load the index (allow_dangerous_deserialization is required for loading local pickle files)
    try:
        vector_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = vector_db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error loading FAISS index: {e}"