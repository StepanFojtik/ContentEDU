# scripts/embedding_utils.py

import os
import pdfplumber
import numpy as np
import torch
from loguru import logger
from FlagEmbedding import BGEM3FlagModel


def load_and_chunk_pdf(file_path, chunk_size=512, chunk_overlap=100):
    """
    Loads text from a PDF file and splits it into overlapping chunks.
    """
    if not os.path.exists(file_path):
        logger.error(f"PDF file not found at {file_path}")
        return None

    logger.info(f"Loading PDF: {file_path}")
    full_text = ""

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i % 10 == 0:
                logger.debug(f"Processing page {i + 1}/{len(pdf.pages)}")
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    logger.info(f"Total characters extracted: {len(full_text)}")

    chunks = []
    start_index = 0
    while start_index < len(full_text):
        end_index = start_index + chunk_size
        chunks.append(full_text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
        if start_index >= len(full_text) and len(chunks[-1]) < chunk_overlap and len(chunks) > 1:
            break

    logger.info(f"Created {len(chunks)} chunks.")
    return chunks


def initialize_bge_model(model_name='BAAI/bge-m3'):
    """
    Initializes the BGE-M3 FlagModel.
    """
    logger.info(f"Initializing BGE model: {model_name}")

    use_fp16_flag = torch.cuda.is_available()
    if use_fp16_flag:
        logger.info("CUDA available. Using GPU with fp16.")
    else:
        logger.info("CUDA not available. Using CPU.")

    model = BGEM3FlagModel(model_name, use_fp16=use_fp16_flag)
    logger.info("BGE model initialized.")
    return model


def generate_embeddings(model, text_chunks, batch_size=32):
    """
    Generates dense embeddings for the given list of text chunks.
    """
    if not text_chunks:
        logger.error("No text chunks to embed.")
        return None
    if not model:
        logger.error("Model not initialized.")
        return None

    logger.info(f"Generating embeddings for {len(text_chunks)} chunks.")
    all_embeddings = []

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        embeddings_dict = model.encode(
            batch,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        dense_embeddings = embeddings_dict['dense_vecs']
        all_embeddings.extend(dense_embeddings)

        if i % 10 == 0:
            logger.debug(f"Embedded batch {i // batch_size + 1}/{(len(text_chunks) + batch_size - 1) // batch_size}")

    logger.info("Embedding generation completed.")
    return np.array(all_embeddings)
