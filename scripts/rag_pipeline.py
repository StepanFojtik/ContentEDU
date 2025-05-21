# scripts/rag_pipeline.py

from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def initialize_llm_and_tokenizer(
    model_name="google/gemma-2b",  # změněno na kompatibilní CPU model
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_8bit=False,
):
    """
    Initializes a HuggingFace Transformer model and tokenizer.
    """
    logger.info(f"Loading LLM model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    logger.info("LLM and tokenizer loaded successfully.")
    return model, tokenizer


def retrieve_documents(collection, query_text, bge_embed_model, n_results=3):
    """
    Embeds the query and retrieves relevant documents from ChromaDB.
    """
    logger.info(f"Retrieving documents for query: '{query_text}'")
    query_embedding_dict = bge_embed_model.encode(
        [query_text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    query_embedding = query_embedding_dict['dense_vecs'].tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'distances']
    )

    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]
    logger.info(f"Retrieved {len(documents)} documents.")
    return documents


def generate_response_with_llm(llm_model, tokenizer, query, context_docs, max_new_tokens=256):
    """
    Generates a response using the LLM given the query and retrieved context.
    """
    logger.info("Generating response from LLM...")
    context_str = "\n\n".join(context_docs)

    prompt = (
        "You are a helpful assistant. Answer the user's question based ONLY on the provided context. "
        "If the context doesn't contain the answer, say that the information is not available.\n\n"
        f"Context:\n---\n{context_str}\n---\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    _ = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=streamer
    )
