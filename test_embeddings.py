"""
Test script to verify local HuggingFace embeddings are working correctly.
"""

from langchain_huggingface import HuggingFaceEmbeddings

print("Initializing HuggingFace embeddings...")
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("[OK] Embeddings initialized successfully!")

# Test embedding a single query
test_query = "What is machine learning?"
print(f"\nTesting query embedding: '{test_query}'")
query_embedding = embedding_function.embed_query(test_query)
print(f"[OK] Query embedding generated! Dimension: {len(query_embedding)}")

# Test embedding multiple documents
test_docs = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language."
]
print(f"\nTesting document embeddings for {len(test_docs)} documents...")
doc_embeddings = embedding_function.embed_documents(test_docs)
print(f"[OK] Document embeddings generated! Count: {len(doc_embeddings)}, Dimension: {len(doc_embeddings[0])}")

print("\n[SUCCESS] All tests passed! Local embeddings are working correctly.")
print("[INFO] The system is now using offline, local embeddings via HuggingFace!")
