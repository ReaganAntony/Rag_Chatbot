from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a PDF document uploaded to the RAG system.
    Stores metadata about the document including its source and indexing status.
    """
    doc_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the document")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the document was uploaded")
    indexed_to_kb: bool = Field(default=True, description="Whether the document is indexed to the knowledge base")
    page_count: int = Field(default=0, description="Total number of pages in the document")


class Chunk(BaseModel):
    """
    Represents a chunk of text extracted from a document.
    Each chunk maintains a reference to its source document and page for traceability.
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: UUID = Field(..., description="Reference to the parent document's ID")
    page_number: int = Field(..., description="Page number from which this chunk was extracted")
    text: str = Field(..., description="The actual text content of the chunk")
