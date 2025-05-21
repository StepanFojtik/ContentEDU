# scripts/rag_pipeline.py

from loguru import logger
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer


def initialize_llm_and_tokenizer(model_name="unsloth/gemma-3-4b-it", max_seq_length=65536, load_in_4bit=True, load_in_8bit=False, chat_template="gemma-3"):
    """
    Initializes the Unsloth FastLanguageModel and tokenizer with specified parameters.
    """
    logger.info(f"Loading LLM model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    logger.info("LLM loaded. Applying chat template...")
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
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
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are a helpful assistant. Answer the user's question based ONLY on the provided context. "
                    "If the context doesn't contain the answer, say that the information is not available."
                )
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Based on the following context:\n\n---\n{context_str}\n---\n\nAnswer this question: {query}"
            }]
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    _ = llm_model.generate(
        **tokenizer([input_text], return_tensors="pt").to("cuda"),
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


