# Load needed libraries
import os
import pdfplumber
from typing import List, Tuple
from loguru import logger
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np

# Settings for chunk size
CHUNK_SIZE = 512        # how many characters in one chunk
CHUNK_OVERLAP = 100     # how much text overlaps between chunks
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

#Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Load a PDF file and extract all text from it.
    pdf_path is just a variable name here.
    when you call this function, you tell it what that path is.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file does not exist: {pdf_path}")
        return ""

    logger.info(f"Reading PDF: {pdf_path}")
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i % 10 == 0:
                    logger.debug(f"Processing page {i + 1}/{len(pdf.pages)}")
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""

    logger.info(f"Total characters extracted: {len(full_text)}")
    return full_text

#Split the text into chunks
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split long text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap

    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

#Load the BGE-M3 model
def initialize_bge_model():
    """
    Load the BGE-M3 embedding model.
    """
    use_fp16 = torch.cuda.is_available()
    logger.info(f"Using {'GPU' if use_fp16 else 'CPU'} for embedding.")
    model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=use_fp16)
    logger.info("Model loaded.")
    return model

#Create embeddings for each chunk
def generate_embeddings(model, chunks: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Create embeddings for all chunks using BGE-M3 model.
    """
    logger.info(f"Creating embeddings for {len(chunks)} chunks...")
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings_dict = model.encode(
            batch,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        all_embeddings.extend(embeddings_dict["dense_vecs"])
        logger.debug(f"Processed batch {i // batch_size + 1}")

    logger.info("All embeddings created.")
    return np.array(all_embeddings)

#Main function that runs all steps
def process_pdf_to_embeddings(pdf_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Main function: from PDF file to chunks + embeddings.
    """
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.warning("No text extracted.")
        return [], np.array([])

    chunks = chunk_text(text)
    model = initialize_bge_model()
    embeddings = generate_embeddings(model, chunks)
    return chunks, embeddings
