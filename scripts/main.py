# main.py

from pdf_utils import download_pdf_and_extract_text
from embedding_utils import load_and_chunk_pdf, initialize_bge_model, generate_embeddings
from chroma_utils import (
    setup_chromadb_collection,
    store_in_chromadb,
    get_chromadb_collection,
    query_chromadb
)
from rag_pipeline import (
    initialize_llm_and_tokenizer,
    retrieve_documents,
    generate_response_with_llm
)

import os

# --- Konfigurace ---
PDF_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"
PDF_FILE_PATH = "data/ai_act.pdf"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
COLLECTION_NAME = "ai_act_collection_bge_m3"
PERSIST_DIRECTORY = "chroma_db_storage_bge_m3"
N_RESULTS = 3


def run_pipeline():
    # --- 1. Stáhnout PDF a extrahovat text ---
    print("\n--- 1. Download & extract PDF ---")
    text = download_pdf_and_extract_text(PDF_URL)
    if not text:
        print("PDF nebylo možné stáhnout nebo extrahovat.")
        return

    # Volitelně uložit PDF pro pozdější použití
    os.makedirs("data", exist_ok=True)
    with open(PDF_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    # --- 2. Rozdělit text na chunky ---
    print("\n--- 2. Chunk text ---")
    chunks = load_and_chunk_pdf(PDF_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print("Chyba při dělení textu.")
        return

    # --- 3. Inicializace modelu a vytvoření embeddingů ---
    print("\n--- 3. Generate embeddings ---")
    bge_model = initialize_bge_model()
    embeddings = generate_embeddings(bge_model, chunks)
    if embeddings is None:
        print("Chyba při generování embeddingů.")
        return

    # --- 4. Uložení do ChromaDB ---
    print("\n--- 4. Store in ChromaDB ---")
    collection = setup_chromadb_collection(COLLECTION_NAME, PERSIST_DIRECTORY)
    success = store_in_chromadb(collection, chunks, embeddings, PDF_FILE_PATH)
    if not success:
        print("Chyba při ukládání do ChromaDB.")
        return

    print("Všechno uloženo vektorově!")


def run_query():
    # --- 1. Inicializuj embedding model a LLM ---
    print("\n--- 1. Init models ---")
    bge_model = initialize_bge_model()
    llm_model, tokenizer = initialize_llm_and_tokenizer()

    # --- 2. Připoj se k ChromaDB kolekci ---
    print("\n--- 2. Load ChromaDB ---")
    collection = get_chromadb_collection(PERSIST_DIRECTORY, COLLECTION_NAME)
    if not collection:
        print("Nelze načíst kolekci.")
        return

    # --- 3. Zeptej se na dotaz ---
    print("\n--- 3. Query ---")
    query = input("Zadej dotaz: ")
    retrieved_docs = retrieve_documents(collection, query, bge_model, N_RESULTS)

    if not retrieved_docs:
        print("Nenalezeny žádné dokumenty.")
        return

    # --- 4. Získat odpověď z LLM ---
    print("\n--- 4. LLM odpověď ---")
    generate_response_with_llm(llm_model, tokenizer, query, retrieved_docs)


if __name__ == "__main__":
    print("Vyber režim:")
    print("1 – Stáhnout a uložit PDF do ChromaDB")
    print("2 – Dotazovat databázi a získat odpověď")

    mode = input("Zadej 1 nebo 2: ").strip()

    if mode == "1":
        run_pipeline()
    elif mode == "2":
        run_query()
    else:
        print("Neplatný výběr.")
