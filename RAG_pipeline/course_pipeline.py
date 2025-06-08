# ContentEDU – LangGraph pipeline for course generation
# This script defines the backend logic for generating structured digital courses
# from teaching materials and syllabi using LLMs and LangGraph

# Standard library imports
import os
from typing import List
from typing import Dict
import re

# PDF processing
import pdfplumber

# Environment variable loading
from dotenv import load_dotenv

# LangGraph components
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# LangChain core components
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

# LangChain integrations
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# Utilities
from html import escape

# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# === Global configuration settings ===
API_KEY = os.getenv("OPENAI_API_KEY")            # API key for accessing OpenAI services
VECTOR_DB_PATH = "vectorstore"                   # Directory path for storing the Chroma vector database
CHUNK_SIZE = 1000                                 # Character length of each text chunk for embedding
CHUNK_OVERLAP = 100                               # Number of overlapping characters between consecutive chunks

# === PDF Processing and Embedding ===

# 1. Read and extract all text from a PDF file using pdfplumber
# Returns a single string containing the text from all pages
def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# 2. Split the extracted text into overlapping chunks for embedding
# Each chunk is a LangChain Document, suitable for vectorization
def split_into_chunks(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents([text])

# 3. Generate vector embeddings for all chunks and store them in a persistent Chroma vector database
def embed_and_store_chunks(docs: List[Document]):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_PATH)
    vectorstore.persist()

# 4. Retrieve top-k relevant text chunks from the vector DB based on a similarity search using the syllabus
def retrieve_relevant_context(syllabus_text: str, k: int = 5) -> str:
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(syllabus_text)
    return "\n".join([doc.page_content for doc in docs])

# Format selected syllabus fields into a structured text block
# This text is used as input for prompt templates
def build_syllabus_text(info: dict) -> str:
    return f"""
Course title in English: {info.get("course_name", "").strip()}
Name of lecturer(s): {info.get("lecturers", "").strip()}
Aims of the course: {info.get("aims", "").strip()}
Learning outcomes and competences: {info.get("learning_outcomes", "").strip()}
Course contents: {info.get("course_contents", "").strip()}
Assessment methods and criteria: {info.get("grading_method", "").strip()}
""".strip()

# === Prompt-based generation functions ===
# These functions use prompt templates and LLMs to generate different parts of the course

# Load a text prompt file from the prompts directory
def load_prompt(filename: str) -> str:
    path = os.path.join("prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# Generate the course structure using the syllabus and extracted context
def generate_structure(syllabus_text: str, context: str = "") -> dict:
    prompt_template = load_prompt("course_structure_prompt.txt")
    system_prompt = load_prompt("system_prompt.txt")

    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview").with_config({"system_message": system_prompt})
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.run({
        "syllabus_text": syllabus_text,
        "context": context  
    })
    return {"text": output}

# Generate the Announcements and Introduction sections using syllabus, structure, and relevant context
def generate_announcements_and_intro(syllabus_text: str, context: str, structure_text: str = "") -> str:
    intro_prompt = load_prompt("announcements_and_intro_prompt.txt")
    quiz_format = load_prompt("quiz_format_prompt.txt")

    full_prompt = intro_prompt + "\n\n" + quiz_format

    prompt = PromptTemplate.from_template(full_prompt)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run({
        "syllabus_text": syllabus_text, 
        "context": context, 
        "structure_text": structure_text
    })

# Generate one content module including learning content and a self-check quiz
def generate_module(idx: int, topic: str, context: str) -> str:
    module_template = load_prompt("content_module_prompt.txt")
    quiz_format = load_prompt("quiz_format_prompt.txt")

    combined_prompt = module_template + "\n\n" + quiz_format
    filled_prompt = combined_prompt.replace("[CONTENT_TOPIC_HERE]", topic).replace("Module X", f"Module {idx}")
    
    prompt = PromptTemplate.from_template(filled_prompt)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo-1106")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run({"context": context})

# Generate the Final Quiz and Conclusion sections using syllabus and modules
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

# === LangGraph state ===
# Shared data structure passed between nodes during course generation

class CourseState(dict):
    course_name: str                  # name of the course (English)
    syllabus_text: str               # formatted syllabus content used as input for LLM prompts
    structure: dict                  # generated course structure (titles + descriptions)
    context: str                     # relevant context retrieved from PDF materials via vector search
    modules: list                    # list of generated content modules (as strings)
    announcements_intro: str        # generated content for Announcements and Introduction sections
    final_parts: str                # generated Final Quiz and Conclusion sections
    full_draft: str                 # combined full output of the entire course              

# === LangGraph nodes ===
# These functions represent individual steps in the LangGraph pipeline.
# Each node receives the current state, performs a task, and returns the updated state.

# Retrieve relevant context from PDF materials using vector similarity search
def node_retrieve_context(state):
    context = retrieve_relevant_context(state["syllabus_text"])
    return {**state, "context": context}

# Generate the overall course structure using the syllabus and relevant context
def node_generate_structure(state):
    structure = generate_structure(
        syllabus_text=state["syllabus_text"],
        context=state["context"]
    )
    return {**state, "structure": structure}

# Generate the Announcements and Introduction sections
def node_announcements_intro(state):
    content = generate_announcements_and_intro(
        syllabus_text=state["syllabus_text"],
        context=state["context"],
        structure_text=state["structure"]["text"]
    )
    return {**state, "announcements_intro": content}

# Generate all course modules based on the structure and context
def node_generate_modules(state):
    structure_text = state["structure"]["text"]
    topics = re.findall(r"Module \d+\s*[-–]\s*(.+)", structure_text)

    modules = []
    for idx, topic in enumerate(topics, 1):
        module_content = generate_module(idx, topic, state["context"])
        modules.append(module_content)

    return {**state, "modules": modules}

 # Generate the Final Quiz and Conclusion sections
def node_final_parts(state):
    content = generate_final_parts(
        syllabus_text=state["syllabus_text"],
        structure=state["structure"],
        modules=state["modules"],
        context=state["context"]
    )
    return {**state, "final_parts": content}

# Assemble all generated parts into a complete course draft
def node_assemble_draft(state):
    full_draft = "\n\n".join([
        state["announcements_intro"],
        *state["modules"],
        state["final_parts"]
    ])
    return {**state, "full_draft": full_draft}

# === LangGraph definition ===
# Defines the full pipeline structure and transitions between steps

# Initialize the state graph with the CourseState data structure
graph = StateGraph(CourseState)

# Register individual nodes (functions) that make up the pipeline
graph.add_node("retrieve_context", node_retrieve_context)
graph.add_node("generate_structure", node_generate_structure)          
graph.add_node("gen_announcements_intro", node_announcements_intro)      
graph.add_node("generate_modules", node_generate_modules)           
graph.add_node("gen_final_parts", node_final_parts)                     
graph.add_node("assemble_draft", node_assemble_draft)                           

# Define the execution order of nodes from start to finish
graph.set_entry_point("retrieve_context")
graph.add_edge("retrieve_context", "generate_structure")
graph.add_edge("generate_structure", "gen_announcements_intro")
graph.add_edge("gen_announcements_intro", "generate_modules")
graph.add_edge("generate_modules", "gen_final_parts")
graph.add_edge("gen_final_parts", "assemble_draft")
graph.set_finish_point("assemble_draft")

# Compile the graph with in-memory checkpointing to retain state between steps
course_graph = graph.compile(checkpointer=MemorySaver())