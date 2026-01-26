"""
FastAPI Backend for RAG System

This module provides the REST API endpoints for document ingestion
and retrieval operations.
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingest import extract_pdf_pages, get_page_count
from core.chunker import chunk_document
from core.storage import save_chunks_to_db
from core.retriever import get_context
from core.qa_engine import generate_answer
from models import Document

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="API for PDF document ingestion and retrieval using RAG",
    version="1.0.0"
)

# Configure CORS for Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory for storing uploaded files
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


# Response Models
class UploadResponse(BaseModel):
    """Response model for document upload - returns full Document entity."""
    status: str
    doc_id: str
    filename: str
    uploaded_at: datetime
    tags: List[str]
    indexed_to_kb: bool
    page_count: int
    chunks_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str


class QueryRequest(BaseModel):
    """Request model for querying documents."""
    question: str
    doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str
    question: str
    sources_used: int


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the server is running.
    """
    return HealthResponse(
        status="ok",
        message="RAG System API is running"
    )


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    tags: str = Form(default="", description="Comma-separated tags (optional)"),
    indexed_to_kb: bool = Form(default=True, description="Index to knowledge base")
):
    """
    Upload and ingest a PDF document.

    This endpoint:
    1. Receives a PDF file upload
    2. Saves it to the data directory
    3. Extracts text from PDF pages
    4. Chunks the document text
    5. Stores chunks with embeddings in ChromaDB
    6. Returns the complete Document entity

    Args:
        file: PDF file to upload
        tags: Comma-separated tags (optional)
        indexed_to_kb: Whether to index to knowledge base (default: True)

    Returns:
        UploadResponse with complete Document entity and chunks_count.
    """
    # Validate PDF file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        # Step 1: Generate document metadata
        doc_id = uuid4()
        uploaded_at = datetime.now()
        filename = file.filename
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []

        # Step 2: Save uploaded file to data directory
        file_path = DATA_DIR / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved to: {file_path}")

        # Step 3: Extract text from PDF pages
        print(f"Extracting pages from: {filename}")
        pages_data = extract_pdf_pages(str(file_path))

        if not pages_data:
            # Clean up file if no text extracted
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in the PDF"
            )

        # Step 4: Get page count
        page_count = get_page_count(str(file_path))

        # Step 5: Create Document entity
        document = Document(
            doc_id=doc_id,
            filename=filename,
            uploaded_at=uploaded_at,
            tags=tag_list,
            indexed_to_kb=indexed_to_kb,
            page_count=page_count
        )

        chunks_count = 0

        # Step 6: If indexed_to_kb is True, chunk and store in ChromaDB
        if indexed_to_kb:
            print(f"Chunking {len(pages_data)} pages...")
            chunks = chunk_document(pages_data, doc_id)

            doc_metadata = {
                "doc_id": str(doc_id),
                "filename": filename,
                "tags": tag_list,
                "page_count": page_count
            }

            print(f"Saving {len(chunks)} chunks to database...")
            chunks_count = save_chunks_to_db(chunks, doc_metadata)

        print(f"Successfully ingested document: {filename}")

        return UploadResponse(
            status="success",
            doc_id=str(document.doc_id),
            filename=document.filename,
            uploaded_at=document.uploaded_at,
            tags=document.tags,
            indexed_to_kb=document.indexed_to_kb,
            page_count=document.page_count,
            chunks_count=chunks_count
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        await file.close()


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query documents and get an AI-generated natural language answer.

    This endpoint:
    1. Retrieves relevant context chunks from ChromaDB
    2. Sends the context to Gemini for answer generation
    3. Returns a natural language answer with citations

    Args:
        request: QueryRequest containing question and optional doc_id filter.

    Returns:
        QueryResponse with answer, question, and sources_used count.
    """
    try:
        # Step 1: Retrieve relevant context from vector database
        print(f"Retrieving context for: {request.question}")
        context = get_context(
            query=request.question,
            doc_id=request.doc_id,
            n_results=5
        )

        # Step 2: Check if we have any context
        if not context or context.strip() == "":
            return QueryResponse(
                answer="I don't have any relevant information to answer this question. Please make sure documents have been ingested into the system.",
                question=request.question,
                sources_used=0
            )

        # Count sources (each chunk separated by ---)
        sources_count = context.count("---") + 1 if context else 0

        # Step 3: Generate natural language answer using Gemini
        print("Generating answer with Gemini...")
        answer = generate_answer(
            question=request.question,
            context=context
        )

        print(f"Answer generated successfully")

        return QueryResponse(
            answer=answer,
            question=request.question,
            sources_used=sources_count
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
