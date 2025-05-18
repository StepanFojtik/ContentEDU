
 Commented out IPython magic to ensure Python compatibility.
 %%capture
 import os
 if "COLAB_" not in "".join(os.environ.keys()):
     !pip install unsloth
 else:
     # Do this only in Colab notebooks! Otherwise use pip install unsloth
     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
     !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
     !pip install --no-deps unsloth
!pip install FlagEmbedding pdfplumber chromadb loguru
 import unsloth


import requests
import pdfplumber
import io

def download_pdf_and_extract_text(pdf_url):
    """
    Downloads a PDF from a given URL and extracts text from it.

    Args:
        pdf_url (str): The URL of the PDF file.

    Returns:
        str: The extracted text from the PDF, or None if an error occurs.
    """
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        print("PDF downloaded successfully.")

        if 'application/pdf' not in response.headers.get('Content-Type', '').lower():
            print(f"Error: The URL does not point to a PDF file. Content-Type: {response.headers.get('Content-Type')}")
            return None

        pdf_file = io.BytesIO(response.content)

        extracted_text = ""
        print("Extracting text from PDF...")
        with pdfplumber.open(pdf_file) as pdf:
            # Iterate through each page and extract text
            for i, page in enumerate(pdf.pages):
                if i % 10 == 0:
                    print(f"Processing page {i+1}/{len(pdf.pages)}")
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        if not extracted_text.strip():
            print("Warning: No text could be extracted from the PDF. It might be image-based or protected.")
            return None

        print("Text extraction complete.")
        return extracted_text

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the PDF: {e}")
        return None
    except pdfplumber.exceptions.PDFSyntaxError as e:
        print(f"Error: Invalid or corrupted PDF file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    pdf_url_to_process = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"
    text_content = download_pdf_and_extract_text(pdf_url_to_process)

    if text_content:
        # Print the first 1000 characters of the extracted text as a sample
        print("\n--- Extracted Text (First 1000 characters) ---")
        print(text_content[:1000])

        # Optionally, save to a file
        # with open("extracted_ai_act_text.txt", "w", encoding="utf-8") as f:
        #     f.write(text_content)
        # print("\nFull extracted text saved to 'extracted_ai_act_text.txt'")

import os
import numpy as np
import chromadb
from PyPDF2 import PdfReader
from FlagEmbedding import BGEM3FlagModel
import torch
from loguru import logger

PDF_FILE_PATH = "ai_act.pdf"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
COLLECTION_NAME = "ai_act_collection_bge_m3"
PERSIST_DIRECTORY = "./chroma_db_storage_bge_m3"
BGE_MODEL_NAME = 'BAAI/bge-m3'
BGE_EMBEDDING_DIMENSION = 1024

def load_and_chunk_pdf(file_path, chunk_size, chunk_overlap):
    """
    Loads text from a PDF file and splits it into overlapping chunks.
    """
    if not os.path.exists(file_path):
        logger.error(f"PDF file not found at {file_path}")
        return None

    logger.info(f"Loading PDF: {file_path}")
    # reader = PdfReader(file_path)
    # full_text = ""
    # for page_num, page in enumerate(reader.pages):
    #     full_text += page.extract_text() or "" # handle None returns
    #     if page_num % 10 == 0:

    full_text = ""
    print("Extracting text from PDF...")
    with pdfplumber.open(file_path) as pdf:
        # Iterate through each page and extract text
        for i, page in enumerate(pdf.pages):
            if i % 10 == 0:
                logger.debug(f"Processed page {i + 1}/{len(pdf.pages)}")
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
        if start_index >= len(full_text) and len(chunks[-1]) < chunk_overlap and len(chunks) > 1: # Avoid very small last chunk if it's mostly overlap
            break

    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def initialize_bge_model():
    """
    Initializes the BGE-M3 FlagModel.
    """
    logger.info(f"Initializing BGE model: {BGE_MODEL_NAME}")

    use_fp16_flag = False
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU with fp16.")
        use_fp16_flag = True
    else:
        logger.info("CUDA not available. Using CPU.")

    model = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=use_fp16_flag)
    logger.info("BGE model initialized successfully.")
    return model

def generate_embeddings(model, text_chunks, batch_size=32):
    """
    Generates embeddings for a list of text chunks using the BGE model.
    """
    if not text_chunks:
        logger.error("No text chunks to embed.")
        return None

    if not model:
        logger.error("BGE model not initialized. Cannot generate embeddings.")
        return None

    logger.info(f"Generating embeddings for {len(text_chunks)} chunks...")
    all_embeddings = []

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        embeddings_dict = model.encode(batch, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense_embeddings = embeddings_dict['dense_vecs']
        all_embeddings.extend(dense_embeddings)
        if i % 10 == 0:
            logger.debug(f"Embedded batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")

    logger.info("Embeddings generated successfully.")
    return np.array(all_embeddings)

def setup_chromadb_collection(collection_name, persist_directory, embedding_dimension):
    """
    Initializes ChromaDB client and creates/gets a collection.
    """
    logger.info(f"Setting up ChromaDB collection: {collection_name}")

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # use cosine distance as a distance metric
    )

    logger.info(f"ChromaDB collection '{collection_name}' ready in '{persist_directory}'.")
    return collection

def store_in_chromadb(collection, text_chunks, embeddings, pdf_file_path):
    """
    Stores text chunks and their embeddings in the ChromaDB collection.
    """
    if not collection or embeddings is None or not text_chunks:
        logger.error("Missing collection, embeddings, or chunks. Cannot store in ChromaDB.")
        return False

    logger.info(f"Storing {len(text_chunks)} items in ChromaDB...")

    ids = [f"chunk_{i+1}_from_{os.path.basename(pdf_file_path)}" for i in range(len(text_chunks))]
    metadatas = [{"source": os.path.basename(pdf_file_path), "chunk_index": i} for i in range(len(text_chunks))]
    embeddings_list = embeddings.tolist()

    collection.add(
        embeddings=embeddings_list,
        documents=text_chunks,
        metadatas=metadatas,
        ids=ids
    )

    logger.info("Data stored in ChromaDB successfully.")
    return True

def query_chromadb(collection, query_text, bge_model, n_results=3):
    """
    Queries the ChromaDB collection using a text query and BGE model for embedding.
    """
    if not collection or not bge_model:
        logger.error("ChromaDB collection or BGE model not available for querying.")
        return None

    logger.info(f"Querying collection with: '{query_text}'")

    # Generate embedding for the query text
    query_embedding_dict = bge_model.encode([query_text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    query_embedding = query_embedding_dict['dense_vecs'].tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas']
    )

    logger.info("Query successful.")
    return results

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting PDF processing and vector database creation...")

    # 1. Load and Chunk PDF
    chunks = load_and_chunk_pdf(PDF_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        logger.error("Exiting due to PDF loading/chunking error.")
        exit()

    # 2. Initialize BGE-M3 Model
    bge_model_instance = initialize_bge_model()
    if not bge_model_instance:
        logger.error("Exiting due to BGE model initialization error.")
        exit()

    # 3. Generate Embeddings
    chunk_embeddings = generate_embeddings(bge_model_instance, chunks)
    if chunk_embeddings is None:
        logger.error("Exiting due to embedding generation error.")
        exit()

    # 4. Setup ChromaDB
    chroma_collection = setup_chromadb_collection(COLLECTION_NAME, PERSIST_DIRECTORY, BGE_EMBEDDING_DIMENSION)
    if not chroma_collection:
        logger.error("Exiting due to ChromaDB setup error.")
        exit()

    # 5. Store in ChromaDB
    success = store_in_chromadb(chroma_collection, chunks, chunk_embeddings, PDF_FILE_PATH)
    if not success:
        logger.error("Exiting due to error storing data in ChromaDB.")
        exit()

    logger.success("Process completed successfully!")
    logger.info(f"Vector database created and populated in: {PERSIST_DIRECTORY}/{COLLECTION_NAME}")

    # 6. Example Query
    if chroma_collection and bge_model_instance:
        print("--- Example Query ---")
        sample_query = "What are the obligations for AI providers?"
        query_results = query_chromadb(chroma_collection, sample_query, bge_model_instance, n_results=3)
        if query_results:
            for i in range(len(query_results['ids'][0])):
                print(f"Result {i+1}:")
                print(f"  ID: {query_results['ids'][0][i]}")
                print(f"  Distance: {query_results['distances'][0][i]:.4f}")
                print(f"  Metadata: {query_results['metadatas'][0][i]}")
                print(f"  Document: {query_results['documents'][0][i][:200]}...")

import os
import numpy as np
import chromadb
from FlagEmbedding import BGEM3FlagModel
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from loguru import logger
from transformers import TextStreamer


# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db_storage_bge_m3"
COLLECTION_NAME = "ai_act_collection_bge_m3"
BGE_MODEL_NAME = 'BAAI/bge-m3'
UNSLOTH_MODEL_NAME = "unsloth/gemma-3-4b-it"
MAX_SEQ_LENGTH = 65536
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
CHAT_TEMPLATE = "gemma-3"
N_RETRIEVED_DOCS = 3
MAX_NEW_TOKENS_LLM = 256

def initialize_bge_model():
    """
    Initializes the BGE-M3 FlagModel for query embedding.
    """
    logger.info(f"Initializing BGE embedding model: {BGE_MODEL_NAME}")
    model = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=True)
    logger.info("BGE embedding model initialized successfully.")
    return model

def initialize_llm_and_tokenizer():
    """
    Initializes the Unsloth FastLanguageModel and its tokenizer.
    """
    logger.info(f"Initializing LLM model: {UNSLOTH_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=UNSLOTH_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    logger.info("LLM model loaded.")
    logger.info(f"Applying chat template: {CHAT_TEMPLATE}")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )
    logger.info("Chat template applied.")
    return model, tokenizer

def get_chromadb_collection():
    """
    Connects to the existing ChromaDB collection.
    """
    logger.info(f"Connecting to ChromaDB collection: {COLLECTION_NAME} in {PERSIST_DIRECTORY}")
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.error(f"ChromaDB persist directory not found at {PERSIST_DIRECTORY}")
        logger.error("Please ensure you have created the database.")
        return None

    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}'.")
        logger.info(f"Collection has {collection.count()} items.")
        return collection
    except Exception as e:
        logger.error(f"Collection '{COLLECTION_NAME}' not found or other ChromaDB error: {e}")
        return None

def retrieve_documents(collection, query_text, bge_embed_model, n_results):
    """
    Embeds the query and retrieves relevant documents from ChromaDB.
    """
    if not collection or not bge_embed_model:
        logger.error("ChromaDB collection or BGE model not available for retrieval.")
        return []
    logger.info(f"Embedding query: '{query_text}'")
    query_embedding_dict = bge_embed_model.encode(
        [query_text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    query_embedding = query_embedding_dict['dense_vecs'].tolist()
    logger.info(f"Querying ChromaDB for {n_results} relevant documents...")
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'distances']
    )

    retrieved_docs = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]
    logger.info(f"Retrieved {len(retrieved_docs)} documents.")
    for i, doc_text in enumerate(retrieved_docs):
        logger.debug(f"Doc {i+1} (Distance: {distances[i]:.4f}): {doc_text[:100]}...")
    return retrieved_docs

def generate_response_with_llm(llm_model, tokenizer, query, context_docs):
    """
    Generates a response using the Unsloth LLM, given the query and retrieved context.
    """
    logger.info("Constructing prompt for LLM...")
    context_str = "\n\n".join(context_docs)
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain the answer, clearly state that the information is not found in the provided context."
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

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    logger.info("Generating response from LLM...")

    _ = llm_model.generate(
        **tokenizer([inputs], return_tensors = "pt").to("cuda"),
        max_new_tokens = 256,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

    return

# --- Main RAG Application ---
if __name__ == "__main__":
    logger.info("--- Starting RAG Application ---")

    # 1. Initialize BGE Embedding Model
    bge_embed_model = initialize_bge_model()
    if not bge_embed_model:
        logger.error("Exiting: BGE model initialization failed.")
        exit()

    # 2. Initialize Unsloth LLM and Tokenizer
    llm_model, llm_tokenizer = initialize_llm_and_tokenizer()
    if not llm_model or not llm_tokenizer:
        logger.error("Exiting: LLM initialization failed.")
        exit()

    # 3. Connect to ChromaDB Collection
    chroma_collection = get_chromadb_collection()
    if not chroma_collection:
        logger.error("Exiting: ChromaDB connection failed.")
        exit()

    user_query = "What are the risk categories?" # input("\nYour Question: ")

    # 4. Retrieve relevant documents
    retrieved_documents = retrieve_documents(
        chroma_collection,
        user_query,
        bge_embed_model,
        N_RETRIEVED_DOCS
    )

    print("\nLLM Answer:")
    if not retrieved_documents:
        logger.warning("No relevant documents found in the database for your query.")
        # fallback
        print("I could not find relevant information in the provided PDF content to answer your question.")
    else:
        generate_response_with_llm(
            llm_model,
            llm_tokenizer,
            user_query,
            retrieved_documents
        )

"""***
# Fine tuninig

We now add LoRA adapters so we only need to update a small amount of parameters!
"""

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 42,
)

"""<a name="Data"></a>
### Data Prep
We now use the `Gemma-3` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-3 renders multi turn conversations like below:

```
<bos><start_of_turn>user
Hello!<end_of_turn>
<start_of_turn>model
Hey there!<end_of_turn>
```

We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

"""We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!"""

from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

"""Let's see how row 100 looks like!"""

dataset[100]

"""We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`"""

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }
pass
dataset = dataset.map(apply_chat_template, batched = True)

"""Let's see how the chat template did! Notice `Gemma-3` default adds a `<bos>`!"""

dataset[100]["text"]

"""<a name="Train"></a>
### Train the model
Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
"""

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!"""

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

"""Let's verify masking the instruction part is done! Let's print the 100th row again:"""

tokenizer.decode(trainer.train_dataset[100]["input_ids"])

"""Now let's print the masked out example - you should see only the answer is present:"""

tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`"""

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

"""<a name="Inference"></a>
### Inference
Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)
outputs = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)

""" You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!"""

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

model.save_pretrained("gemma-3")  # Local saving
tokenizer.save_pretrained("gemma-3")
# model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving

"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-3?",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

"""### Saving to float16 for VLLM

We also support saving to `float16` directly for deployment! We save it in the folder `gemma-3-finetune`. Set `if False` to `if True` to let it run!
"""

if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3-finetune", tokenizer)

"""If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!"""

if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3-finetune", tokenizer,
        token = "hf_..."
    )

"""### GGUF / llama.cpp Conversion
To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!
"""

if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-3-finetune",
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

"""Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!"""

if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3-finetune",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "HF_ACCOUNT/gemma-finetune-gguf",
        token = "hf_...",
    )

"""Now, use the `gemma-3-finetune.gguf` file or `gemma-3-finetune-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)

And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other links:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>

"""