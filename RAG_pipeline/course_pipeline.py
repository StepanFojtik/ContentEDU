# LangGraph pipeline: ContentEDU Generator
import os
from typing import List
import pdfplumber
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import re

# Load environment variables -> our OPENAI_API_KEY
load_dotenv()

# === Settings ===
API_KEY = os.getenv("OPENAI_API_KEY")            # OpenAI key
VECTOR_DB_PATH = "vectorstore"                   # Directory for vector DB
CHUNK_SIZE = 1000                                 # Character length per chunk
CHUNK_OVERLAP = 100                               # Overlap between chunks

# === PDF Processing and Embedding ===

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

# === LangGraph Nodes ===
# These nodes define the flow of course content generation

# === Prompt implementation for course structure ===
# Loads a text prompt file from the prompts directory
def load_prompt(filename: str) -> str:
    path = os.path.join("prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# Generates course structure using external prompt file
def generate_structure(syllabus_text: str) -> dict:
    prompt_template = load_prompt("course_structure_prompt.txt")
    system_prompt = load_prompt("system_prompt.txt")

    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview").with_config({"system_message": system_prompt})
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.run({"syllabus_text": syllabus_text})
    return {"text": output}  # You can parse this to structured format if needed

# === Prompt implementation for announcements and introduction ===
#Generates the Announcements and Introduction sections from a syllabus, including quiz formatting instructions
def generate_announcements_and_intro(syllabus_text: str, context: str, structure_text: str = "") -> str:
    intro_prompt = load_prompt("announcements_and_intro_prompt.txt")
    quiz_format = load_prompt("quiz_format_prompt.txt")

    full_prompt = intro_prompt + "\n\n" + quiz_format

    prompt = PromptTemplate.from_template(full_prompt)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run({"syllabus_text": syllabus_text, "context": context, "structure_text": structure_text})

# === Prompt implementation for content modules ===
# Generates a content module for the course with embedded quiz formatting rules.
def generate_module(idx: int, topic: str, context: str) -> str:
    module_template = load_prompt("content_module_prompt.txt")
    quiz_format = load_prompt("quiz_format_prompt.txt")

    combined_prompt = module_template + "\n\n" + quiz_format

    filled_prompt = combined_prompt.replace("[CONTENT_TOPIC_HERE]", topic).replace("Module X", f"Module {idx}")
    prompt = PromptTemplate.from_template(filled_prompt)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo-1106")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": context})

# === Prompt implementation for final quiz and conclusion ===
# Generates the Final Quiz and Conclusion sections using the syllabus and course modules, including quiz formatting instructions
def generate_final_parts(syllabus_text: str, structure: dict, modules: list, context: str) -> str:
    prompt_template = load_prompt("conclusion_and_final_quiz_prompt.txt")
    quiz_format = load_prompt("quiz_format_prompt.txt")

    modules_text = "\n\n".join(modules)
    full_prompt = prompt_template + "\n\n" + quiz_format

    prompt = PromptTemplate.from_template(full_prompt)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "modules_text": modules_text,
        "syllabus_text": syllabus_text,
        "context": context
    })


# === 1. State ===
# The state dictionary that carries data between graph steps
class CourseState(dict):
    course_name: str                  # Name of the course
    syllabus_text: str                # Full syllabus text
    structure: dict                    # Structured outline of the course (JSON or text)
    context: str
    modules: list                     # List of module contents
    announcements_intro: str          # Combined content of Announcements and Introduction
    final_parts: str                  # Combined content of Final Quiz and Conclusion
    full_draft: str                   # Assembled full draft of the course
    qa_result: str                    # Output of the QA agent

# === 2. Nodes ===

def node_retrieve_context(state):
    context = retrieve_relevant_context(state["syllabus_text"])
    return {**state, "context": context}
# Each node is a step in the course generation pipeline

def node_generate_structure(state):
    # Generates the structure of the course based on syllabus
    structure = generate_structure(state["syllabus_text"])
    return {**state, "structure": structure}

def node_announcements_intro(state):
    # Generates Announcements and Introduction sections
    content = generate_announcements_and_intro(state["syllabus_text"], state["context"])
    return {**state, "announcements_intro": content}

def node_generate_modules(state):
    structure_text = state["structure"]["text"]

    # Najde řádky jako: Module 1 – Název modulu
    topics = re.findall(r"Module \d+\s*[-–]\s*(.+)", structure_text)



    modules = []
    for idx, topic in enumerate(topics, 1):
        module_content = generate_module(idx, topic, state["context"])
        modules.append(module_content)

    return {**state, "modules": modules}

def node_final_parts(state):
    # Generates Final Quiz and Conclusion sections
    content = generate_final_parts(
        syllabus_text=state["syllabus_text"],
        structure=state["structure"],
        modules=state["modules"],
        context=state["context"]
    )
    return {**state, "final_parts": content}

def node_qa_check(state):
    # QA check is currently disabled
    full_draft = "\n\n".join([
        state["announcements_intro"],
        *state["modules"],
        state["final_parts"]
    ])
    # qa_result = run_qa_agent(full_draft)
    qa_result = "(QA agent is currently disabled)"
    return {**state, "full_draft": full_draft, "qa_result": qa_result}

# === 3. LangGraph Definition ===
# Builds the graph structure and defines transitions between nodes

graph = StateGraph(CourseState)
graph.add_node("retrieve_context", node_retrieve_context)
graph.add_node("generate_structure", node_generate_structure)          # First: structure
graph.add_node("gen_announcements_intro", node_announcements_intro)      # Then: intro sections
graph.add_node("generate_modules", node_generate_modules)            # Then: main modules
graph.add_node("gen_final_parts", node_final_parts)                      # Then: quiz + conclusion
graph.add_node("qa_check", node_qa_check)                            # Final: QA check

# Define the graph flow
graph.set_entry_point("retrieve_context")
graph.add_edge("retrieve_context", "generate_structure")
graph.add_edge("generate_structure", "gen_announcements_intro")
graph.add_edge("gen_announcements_intro", "generate_modules")
graph.add_edge("generate_modules", "gen_final_parts")
graph.add_edge("gen_final_parts", "qa_check")
graph.set_finish_point("qa_check")

# Compile the graph with in-memory state tracking
course_graph = graph.compile(checkpointer=MemorySaver())

# === 4. Execution Example ===
# This is how you'd invoke the graph with sample input
# example_state = {
#     "course_name": "Intro to AI",
#     "syllabus_text": syllabus_extracted_text
# }
# result = course_graph.invoke(example_state)
# print(result["qa_result"])

