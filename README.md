📝 RAG-Agent: Multi-Document Chatbot with Local EmbeddingsA high-performance Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and have context-aware conversations. 
This project combines the privacy/speed of local embeddings (LangChain) with the reasoning power of Google Gemini.🚀 
FeaturesLocal Processing: PDF text extraction and document chunking are handled locally.
Hybrid Embedding Pipeline: Uses LangChain with local embedding models for fast, cost-effective vector generation.
Vector Persistence: Powered by ChromaDB for efficient similarity search and long-term storage of document vectors.
Strict Agentic Reasoning: Uses a "Strict Prompt" via the Gemini API to ensure the agent only answers based on the provided context (reducing hallucinations).
Session Management: Intelligent tracking of uploaded document IDs to prevent duplicate processing.

🏗️ Technical ArchitectureIngestion Phase: * Extraction: Text is pulled from user-uploaded PDFs/Docs.
Chunking: Text is split into overlapping segments to preserve context.
Vectorization: Local embeddings are generated (via LangChain).
Storage: Vectors are indexed in a local ChromaDB instance.
Retrieval Phase: * User query is converted into a vector.Top-$k$ most relevant chunks are retrieved from ChromaDB.
Generation Phase: * Retrieved chunks + user query + Strict System Prompt are sent to the Gemini API.Agent generates a response anchored strictly to the data.

🛠️ Tech StackLLM: Google Gemini (Generative AI)Framework: 
LangChainVector Database: ChromaDBEmbeddings: Local Embeddings (HuggingFace/Sentence-Transformers)Language: Python 3.10+Backend: FastAPI / Uvicorn (if applicable)

# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and query them for natural language answers using AI.

## Features

- PDF document ingestion and text extraction
- Semantic search and retrieval using vector embeddings
- AI-powered question answering with Google Gemini
- FastAPI backend for RESTful API endpoints
- ChromaDB for efficient vector storage

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Rag_Chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the backend server:
   ```bash
   python backend/main.py
   ```

2. The API will be available at `http://localhost:8000`

3. Upload a PDF document via the `/upload` endpoint

4. Query the documents via the `/query` endpoint with a question

## Tech Stack

- **Backend**: FastAPI, Python
- **Vector Database**: ChromaDB
- **AI Model**: Google Generative AI (Gemini)
- **Text Processing**: LangChain Text Splitters
- **PDF Processing**: PyPDF
