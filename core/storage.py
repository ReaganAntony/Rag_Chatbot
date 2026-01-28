"""
Vector Storage Module

This module handles the storage and retrieval of document chunks in ChromaDB.
It uses local HuggingFace embeddings for offline embedding generation.
"""

import os
from typing import List, Dict

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize ChromaDB persistent client
CHROMA_DB_PATH = "./data/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Collection name for storing document chunks
COLLECTION_NAME = "documents"

# Initialize local HuggingFace embeddings
# Using sentence-transformers/all-MiniLM-L6-v2 for fast, efficient local embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU is available
    encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
)


def get_or_create_collection(collection_name: str = COLLECTION_NAME):
    """
    Get an existing collection or create a new one if it doesn't exist.

    Args:
        collection_name: Name of the collection.

    Returns:
        ChromaDB collection object.
    """
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "RAG document chunks with embeddings"}
    )


def save_chunks_to_db(chunks: List, doc_metadata: Dict) -> int:
    """
    Save document chunks to ChromaDB with their embeddings.

    Args:
        chunks: List of Chunk objects to store.
        doc_metadata: Dictionary containing document metadata (must include 'filename').

    Returns:
        Number of chunks successfully saved.
    """
    if not chunks:
        print("No chunks to save.")
        return 0

    collection = get_or_create_collection()

    # Prepare data for batch insertion
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        metadatas.append({
            "doc_id": str(chunk.doc_id),
            "page_number": chunk.page_number,
            "filename": doc_metadata.get("filename", "unknown")
        })

    # Generate embeddings for all chunks using local HuggingFace model
    print(f"Generating embeddings locally for {len(documents)} chunks...")
    embeddings = embedding_function.embed_documents(documents)

    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully saved {len(chunks)} chunks to ChromaDB.")
    return len(chunks)


def query_collection(query_text: str, n_results: int = 5, filter_doc_id: str = None) -> Dict:
    """
    Query the collection for similar chunks.

    Args:
        query_text: The query string to search for.
        n_results: Number of results to return.
        filter_doc_id: Optional doc_id to filter results to a specific document.

    Returns:
        Dictionary containing query results with documents and metadata.
    """
    collection = get_or_create_collection()

    # Generate embedding for query using local HuggingFace model
    query_embedding = embedding_function.embed_query(query_text)

    # Build where filter if doc_id is provided
    where_filter = None
    if filter_doc_id:
        where_filter = {"doc_id": filter_doc_id}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    return results


def delete_document_chunks(doc_id: str) -> bool:
    """
    Delete all chunks associated with a specific document.

    Args:
        doc_id: The document ID whose chunks should be deleted.

    Returns:
        True if deletion was successful, False otherwise.
    """
    try:
        collection = get_or_create_collection()
        collection.delete(where={"doc_id": doc_id})
        print(f"Deleted chunks for document: {doc_id}")
        return True
    except Exception as e:
        print(f"Error deleting chunks: {e}")
        return False


