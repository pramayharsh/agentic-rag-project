# Agentic RAG: Multi-Tool Research Assistant 🤖📚

This project is a sophisticated **Agentic RAG (Retrieval-Augmented Generation)** system. Unlike standard RAG pipelines, this agent uses a reasoning loop to determine if it can answer a question using local documents (PDFs) or if it needs to search the live web.

## 🌟 Features
- **Agentic Orchestration:** Built with `LangGraph` to manage stateful, multi-turn reasoning loops.
- **Self-Correction:** The agent evaluates the relevance of PDF search results and automatically pivots to a Web Search if the local data is insufficient.
- **Hybrid AI Stack:** Uses **Groq (Llama 3.1 8B)** for ultra-fast reasoning and **Google Gemini (text-embedding-004)** for high-accuracy embeddings.
- **Multi-Tool Routing:** 
  - **PDF Search:** Vector search using `FAISS`.
  - **Web Search:** Real-time internet access via `DuckDuckGo`.
- **Circuit Breaker Logic:** Prevents infinite loops and redundant API calls.
- **Modern UI:** A clean, conversational interface built with `Streamlit`.

## 🛠️ Tech Stack
- **LLM:** Groq (Llama 3.1 8B)
- **Embeddings:** Google Gemini
- **Orchestration:** LangGraph
- **Vector Database:** FAISS
- **Web Search:** DuckDuckGo (via `ddgs`)
- **UI Framework:** Streamlit

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-project.git
cd agentic-rag-project
```

### 2. Set Up Environment
Create a `.env` file in the root directory and add your API keys:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Ingest Documents
Place your PDF in the `data/` folder and name it `sample.pdf`. Then run the ingestion script to create the FAISS index:

```bash
python src/ingestion.py
```

### 5. Launch the App
```bash
streamlit run app.py
```

## 🧠 How It Works
The agent follows the ReAct (Reason + Act) pattern:

1. **Thought:** Analyzes the user query.
2. **Action:** Searches the Internal PDF.
3. **Observation:** Evaluates if the result answers the question.
4. **Self-Correction:** If the PDF result is irrelevant, it automatically triggers a Web Search.
5. **Final Answer:** Synthesizes all gathered data into a concise response.

## 📄 License
This project is licensed under the MIT License.