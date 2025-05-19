# scripts/chroma_utils.py

import os
from loguru import logger
import chromadb


def setup_chromadb_collection(collection_name, persist_directory):
    """
    Initializes ChromaDB client and gets or creates a collection.
    """
    logger.info(f"Setting up ChromaDB collection '{collection_name}' in '{persist_directory}'")
    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    logger.info(f"Collection '{collection_name}' ready.")
    return collection


def store_in_chromadb(collection, text_chunks, embeddings, pdf_file_path):
    """
    Stores text chunks and their embeddings in the ChromaDB collection.
    """
    if not collection or embeddings is None or not text_chunks:
        logger.error("Missing collection, embeddings, or chunks.")
        return False

    logger.info(f"Storing {len(text_chunks)} items in ChromaDB.")

    ids = [f"chunk_{i+1}_from_{os.path.basename(pdf_file_path)}" for i in range(len(text_chunks))]
    metadatas = [{"source": os.path.basename(pdf_file_path), "chunk_index": i} for i in range(len(text_chunks))]

    collection.add(
        embeddings=embeddings.tolist(),
        documents=text_chunks,
        metadatas=metadatas,
        ids=ids
    )

    logger.info("Storage in ChromaDB completed.")
    return True


def get_chromadb_collection(persist_directory, collection_name):
    """
    Connects to an existing ChromaDB collection.
    """
    if not os.path.exists(persist_directory):
        logger.error(f"ChromaDB directory not found at: {persist_directory}")
        return None

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Connected to ChromaDB collection '{collection_name}'")
        return collection
    except Exception as e:
        logger.error(f"Failed to connect to collection '{collection_name}': {e}")
        return None


def query_chromadb(collection, query_text, bge_model, n_results=3):
    """
    Queries the ChromaDB collection using a text query and BGE model for embedding.
    """
    if not collection or not bge_model:
        logger.error("Missing ChromaDB collection or BGE model.")
        return None

    logger.info(f"Querying collection with: '{query_text}'")
    query_embedding_dict = bge_model.encode(
        [query_text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    query_embedding = query_embedding_dict['dense_vecs'].tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas']
    )

    logger.info("Query successful.")
    return results
