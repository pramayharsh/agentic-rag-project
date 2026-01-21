import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool

load_dotenv()

# 1. Setup the same Embeddings we used in Ingestion
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# 2. Connect to the existing ChromaDB
# We specify the persist_directory where we saved it in Step 2
db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

@tool
def search_pdf(query: str):
    """Searches the PDF database for relevant information based on the user's query."""
    # This is the search function
    # k=3 means "Give me the top 3 most relevant chunks"
    docs = db.similarity_search(query, k=3)
    
    # Combine the results into one string for the Agent to read
    results = "\n\n".join([doc.page_content for doc in docs])
    return results