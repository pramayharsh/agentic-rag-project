import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def ingest_docs():
    # 1. Load the PDF
    pdf_path = "data/sample.pdf" 
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print("Loading document...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split the text into chunks
    print("Splitting document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Initialize Hugging Face Cloud Embeddings
    # This uses your API key and does NOT need local Torch/ONNX
    print("Connecting to Hugging Face Inference API...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # 4. Store in ChromaDB
    print("Storing chunks in ChromaDB (this might take a few seconds)...")
    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("Ingestion complete! Database saved to ./chroma_db")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    ingest_docs()