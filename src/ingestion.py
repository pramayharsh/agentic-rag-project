import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest_docs():
    print("Loading PDF...")
    loader = PyPDFLoader("data/sample.pdf")
    documents = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("Generating Embeddings (Cloud)...")
    # Keeping transport="rest" for stability
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        transport="rest"
    )

    print("Creating FAISS Index...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    print("Saving FAISS index locally...")
    vector_db.save_local("faiss_index")
    
    print("Ingestion complete! Index saved to 'faiss_index' folder.")

if __name__ == "__main__":
    ingest_docs()