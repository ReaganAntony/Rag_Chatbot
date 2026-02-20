"""
Vector Storage Module

This module handles the storage and retrieval of document chunks in ChromaDB.
It uses Google's Generative AI for generating embeddings.
"""

import os
from typing import List, Dict

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB persistent client
CHROMA_DB_PATH = "../Rag_Chatbot/data/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Collection name for storing document chunks
COLLECTION_NAME = "documents"


class GeminiEmbeddingFunction:
    """
    Custom embedding function using Google's Generative AI.
    Uses the embedding-001 model for generating text embeddings.
    """

    def __init__(self, model_name: str = "models/gemini-embedding-001"):
        self.model_name = model_name

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            input_texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for text in input_texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings


# Initialize the embedding function
embedding_function = GeminiEmbeddingFunction()


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

    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = embedding_function(documents)

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

    # Generate embedding for query
    query_embedding = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query_text,
        task_type="retrieval_query"
    )["embedding"]

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
