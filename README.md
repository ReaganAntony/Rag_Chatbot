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
