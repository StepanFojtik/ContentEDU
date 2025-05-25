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
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode


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
{methodology}

---

**Course Name**: {course_name}

**Relevant Extracted Teaching Content**:
{context}

---

Please generate a complete, structured Moodle course based strictly on the methodology and provided content above. Avoid using external knowledge or adding invented facts.
"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(
        methodology=methodology.strip(),
        context=context.strip(),
        course_name=course_name.strip()
    )


# 7. Send the final prompt to the AI model and get the course content
def get_course_content(final_prompt: str) -> str:
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"))
    return chain.run({"input": final_prompt})


#8. tools and agent 
@tool
def check_course_quality(course_draft: str) -> str:
    """Reviews the course draft for structure, coherence, and Moodle compliance."""
    response = ChatOpenAI(model="gpt-4", temperature=0).invoke([
        HumanMessage(content=f"Please review the following course draft for structure, coherence, and Moodle compliance:\n\n{course_draft}")
    ])
    return response.content

def call_model(state: MessagesState):
    model = ChatOpenAI(model="gpt-4", temperature=0).bind_tools([check_course_quality])
    return {"messages": [model.invoke(state["messages"])]}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    return "qa_tool" if last.tool_calls else END

def build_qa_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("qa_tool", ToolNode([check_course_quality]))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("qa_tool", "agent")
    return graph.compile(checkpointer=MemorySaver())

qa_app = build_qa_graph()

def run_qa_agent(course_draft: str) -> str:
    result = qa_app.invoke({"messages": [HumanMessage(content=course_draft)]})
    return result["messages"][-1].content

# === Full Pipeline with QA Integration ===

def run_full_course_pipeline(material_paths: List[str], syllabus_text: str, course_name: str) -> str:
    full_text = "\n\n".join([extract_text_from_pdf(path) for path in material_paths])
    chunks = split_into_chunks(full_text)
    embed_and_store_chunks(chunks)

    context = retrieve_relevant_context(syllabus_text)
    methodology = load_methodology()
    prompt = generate_prompt(context, methodology, course_name)
    course_draft = get_course_content(prompt)

    return run_qa_agent(course_draft)

def regenerate_with_feedback(
    material_paths: List[str],
    syllabus_text: str,
    course_name: str,
    original_output: str,
    feedback: str
) -> str:
    context = retrieve_relevant_context(syllabus_text)
    methodology = load_methodology()
    prompt = f"""
{methodology.strip()}

---

**Course Name**: {course_name.strip()}

**Relevant Extracted Teaching Content**:
{context.strip()}

**Previously Generated Output**:
{original_output.strip()}

**User Feedback for Regeneration**:
{feedback.strip()}

---

Please regenerate the course content strictly based on the methodology and the extracted context above, while incorporating the user's feedback. Avoid using external facts or assumptions.
"""
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-1106-preview")
    regenerated_draft = llm.predict(prompt)
    return run_qa_agent(regenerated_draft)
