import os
from typing import List
import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables -> our OPENAI_API_KEY
load_dotenv()

# === Settings ===
API_KEY = os.getenv("OPENAI_API_KEY")  # our OpenAI key
VECTOR_DB_PATH = "vectorstore"         # folder to save the vector database
CHUNK_SIZE = 1000                      # how many characters in one chunk
CHUNK_OVERLAP = 100                    # how much text overlaps between chunks

# 1. Read and extract all text from a PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# 2. Break the text into smaller overlapping pieces (chunks)
def split_into_chunks(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents([text])

# 3. Create embeddings (vector representations) and save them to Chroma vector DB
def embed_and_store_chunks(docs: List[Document]):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_PATH)
    vectorstore.persist()

# 4. Search the vector DB using the syllabus to find the most relevant parts
def retrieve_relevant_context(syllabus_text: str, k: int = 5) -> str:
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(syllabus_text)
    return "\n".join([doc.page_content for doc in docs])

# 5. Load the course methodology file (how we want to structure the course)
def load_methodology(path: str = "course_methodics.txt") -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

# 6. Build a prompt to send to the AI model (LLM)
def generate_prompt(context: str, methodology: str, course_name: str) -> str:
    template = """
    Course Design Methodology:
    {methodology}

    Context from Teaching Materials:
    {context}

    Course Name: {course_name}

    ➔ Based on the above, generate a structured course suitable for Moodle or LMS.
    """
    prompt = PromptTemplate.from_template(template)
    return prompt.format(methodology=methodology, context=context, course_name=course_name)

# 7. Send the final prompt to the AI model and get the course content
def get_course_content(final_prompt: str) -> str:
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"))
    return chain.run({"input": final_prompt})

# 8. Main pipeline: take multiple teaching material PDFs, extract, embed, and generate content
def run_full_course_pipeline(material_paths: List[str], syllabus_text: str, course_name: str) -> str:
    # Combine the text from all teaching material PDFs
    full_text = "\n\n".join([extract_text_from_pdf(path) for path in material_paths])

    # Break into chunks and embed them
    chunks = split_into_chunks(full_text)
    embed_and_store_chunks(chunks)

    # Find the relevant content and generate the course
    context = retrieve_relevant_context(syllabus_text)
    methodology = load_methodology()
    prompt = generate_prompt(context, methodology, course_name)
    return get_course_content(prompt)

# 9. Regenerate the course based on user feedback
def regenerate_with_feedback(
    material_paths: List[str],
    syllabus_text: str,
    course_name: str,
    original_output: str,
    feedback: str
) -> str:
    context = retrieve_relevant_context(syllabus_text)
    methodology = load_methodology()

    # Add feedback to the prompt so the AI knows what to change
    prompt = f"""
You are a course designer AI assistant.

Below is the course creation methodology:
{methodology}

Context retrieved from the teaching materials:
{context}

Original course content generated earlier:
{original_output}

User feedback on what they want changed:
{feedback}

Please regenerate the course content based on the same methodology and context, while incorporating the user's feedback.
"""
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    return llm.predict(prompt)
