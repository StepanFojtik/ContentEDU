# ContentEDU – LangGraph pipeline for course generation
# This script defines the backend logic for generating structured digital courses
# from teaching materials and syllabi using LLMs and LangGraph
import os
from typing import List
from typing import Dict
import re

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


# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# === Global configuration settings ===
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

# Build syllabus text from parsed data
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

# Load a text prompt from the prompts directory
def load_prompt(filename: str) -> str:
    path = os.path.join("prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# Generate the course structure based on the syllabus
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

# Generate the Announcements and Introduction sections based on the syllabus and context
def generate_announcements_and_intro(syllabus_text: str, context: str, structure_text: str = "") -> str:
    intro_prompt = load_prompt("announcements_and_intro_prompt.txt")
    quiz_format = load_prompt("quiz_format_prompt.txt")

    full_prompt = intro_prompt + "\n\n" + quiz_format

    prompt = PromptTemplate.from_template(full_prompt)
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run({"syllabus_text": syllabus_text, "context": context, "structure_text": structure_text})

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

# Generate the Final Quiz and Conclusion sections using previous modules and syllabus
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
# Shared data structure passed between graph steps during course generation

class CourseState(dict):
    course_name: str                  
    syllabus_text: str                
    structure: dict                   
    context: str
    modules: list                     
    announcements_intro: str         
    final_parts: str                  
    full_draft: str                   
    qa_result: str                    

# === LangGraph nodes ===
# These functions represent individual steps in the LangGraph pipeline.
# Each node takes the current state, performs a task, and returns an updated state.

# Retrieve relevant content from uploaded materials using vector search
def node_retrieve_context(state):
    context = retrieve_relevant_context(state["syllabus_text"])
    return {**state, "context": context}

# Generate the overall structure of the course from the syllabus
def node_generate_structure(state):
    structure = generate_structure(
        syllabus_text=state["syllabus_text"],
        context=state["context"]
    )
    return {**state, "structure": structure}

# Generate Announcements and Introduction sections
def node_announcements_intro(state):
    content = generate_announcements_and_intro(state["syllabus_text"], state["context"])
    return {**state, "announcements_intro": content}

# Generate all content modules based on the structure and context
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

# Assemble all parts into a final draft (QA agent is currently disabled)
def node_qa_check(state):
    full_draft = "\n\n".join([
        state["announcements_intro"],
        *state["modules"],
        state["final_parts"]
    ])
    qa_result = "(QA agent is currently disabled)"
    return {**state, "full_draft": full_draft, "qa_result": qa_result}

# === LangGraph definition ===
# Defines the full pipeline structure and transitions between steps

graph = StateGraph(CourseState)

# Register individual nodes (functions) that make up the pipeline
graph.add_node("retrieve_context", node_retrieve_context)
graph.add_node("generate_structure", node_generate_structure)          
graph.add_node("gen_announcements_intro", node_announcements_intro)      
graph.add_node("generate_modules", node_generate_modules)           
graph.add_node("gen_final_parts", node_final_parts)                     
graph.add_node("qa_check", node_qa_check)                           

# Define execution flow from entry to final node
graph.set_entry_point("retrieve_context")
graph.add_edge("retrieve_context", "generate_structure")
graph.add_edge("generate_structure", "gen_announcements_intro")
graph.add_edge("gen_announcements_intro", "generate_modules")
graph.add_edge("generate_modules", "gen_final_parts")
graph.add_edge("gen_final_parts", "qa_check")
graph.set_finish_point("qa_check")

# Compile the graph with in-memory checkpointing
course_graph = graph.compile(checkpointer=MemorySaver())


