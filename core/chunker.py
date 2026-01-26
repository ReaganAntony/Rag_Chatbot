"""
Document Chunking Module

This module handles splitting document text into smaller chunks
for efficient embedding and retrieval.
"""

import uuid
from typing import List, Dict
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append("..")
from models import Chunk


def chunk_document(pages_data: List[Dict], doc_id: UUID, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
    """
    Split document pages into smaller chunks for embedding.

    Args:
        pages_data: List of dicts with 'text' and 'page_number' from extract_pdf_pages.
        doc_id: UUID of the parent document.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of Chunk objects ready for storage.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages_data:
        page_text = page["text"]
        page_number = page["page_number"]

        # Split the page text into smaller chunks
        text_chunks = text_splitter.split_text(page_text)

        for idx, text in enumerate(text_chunks):
            # Generate a unique chunk_id
            chunk_id = f"{doc_id}_{page_number}_{idx}"

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_number=page_number,
                text=text
            )
            chunks.append(chunk)

    return chunks
