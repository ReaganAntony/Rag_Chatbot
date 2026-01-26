"""
Context Retrieval Module

This module handles retrieving relevant document chunks based on user queries.
It formats the retrieved context with source citations for the LLM.
"""

from typing import Optional

from core.storage import query_collection


def get_context(query: str, doc_id: str = None, n_results: int = 5) -> str:
    """
    Retrieve relevant context chunks for a given query.

    Fetches the top N most relevant chunks from the vector database
    and formats them with source citations for the LLM to reference.

    Args:
        query: The user's question or search query.
        doc_id: Optional document ID to filter results to a specific document.
        n_results: Number of chunks to retrieve (default: 5).

    Returns:
        Formatted string containing relevant chunks with source headers.
        Returns empty string if no relevant chunks are found.
    """
    # Query the vector database for similar chunks
    results = query_collection(
        query_text=query,
        n_results=n_results,
        filter_doc_id=doc_id
    )

    # Check if we got any results
    if not results or not results.get("documents") or not results["documents"][0]:
        return ""

    # Extract results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    # Format context with source citations
    context_parts = []

    for idx, (text, metadata) in enumerate(zip(documents, metadatas)):
        page_number = metadata.get("page_number", "Unknown")
        filename = metadata.get("filename", "Unknown")

        # Create source header
        header = f"[Source: {filename}, Page {page_number}]"

        # Add distance/similarity score if available
        if distances and idx < len(distances):
            similarity = 1 - distances[idx]  # Convert distance to similarity
            header += f" (Relevance: {similarity:.2%})"

        # Combine header and text
        chunk_text = f"{header}\n{text}"
        context_parts.append(chunk_text)

    # Join all chunks with separators
    formatted_context = "\n\n---\n\n".join(context_parts)

    return formatted_context


def get_context_with_metadata(query: str, doc_id: str = None, n_results: int = 5) -> dict:
    """
    Retrieve relevant context chunks with full metadata.

    Similar to get_context but returns structured data instead of formatted string.

    Args:
        query: The user's question or search query.
        doc_id: Optional document ID to filter results to a specific document.
        n_results: Number of chunks to retrieve (default: 5).

    Returns:
        Dictionary containing:
            - context: Formatted context string
            - chunks: List of chunk data with text and metadata
            - total_chunks: Number of chunks retrieved
    """
    results = query_collection(
        query_text=query,
        n_results=n_results,
        filter_doc_id=doc_id
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return {
            "context": "",
            "chunks": [],
            "total_chunks": 0
        }

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for idx, (text, metadata) in enumerate(zip(documents, metadatas)):
        chunk_data = {
            "text": text,
            "page_number": metadata.get("page_number"),
            "filename": metadata.get("filename"),
            "doc_id": metadata.get("doc_id"),
            "relevance": 1 - distances[idx] if distances and idx < len(distances) else None
        }
        chunks.append(chunk_data)

    return {
        "context": get_context(query, doc_id, n_results),
        "chunks": chunks,
        "total_chunks": len(chunks)
    }
