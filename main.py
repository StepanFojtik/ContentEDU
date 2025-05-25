# project_structure/
# ├── documents/
# │   └── methodology.txt (tvůj metodologický soubor)
# ├── main.py
# └── requirements.txt

# ------------------ main.py -------------------
# main.py
import os
import streamlit as st
import pdfplumber
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from docx import Document as WordDocument
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_PATH = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

st.set_page_config(page_title="Course Generator", layout="wide")

# ================== FUNKCE ==================
def extract_text_from_pdf(pdf_file) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def split_into_chunks(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents([text])

def embed_and_store_chunks(docs: List[Document]):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_PATH)
    vectorstore.persist()

def retrieve_relevant_context(query: str, k: int = 5) -> str:
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])

def load_methodology(name: str) -> str:
    path = f"methodology-{name.lower()}.txt"
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def generate_prompt(context: str, methodology: str, title: str, extra: str = "") -> str:
    return f"""
{methodology}

---

**Course Name**: {title}
{f"**Additional Info**: {extra}\n" if extra else ""}
**Relevant Teaching Content**:
{context}

---
Generate the response strictly based on the methodology and provided content.
"""

def call_gpt(prompt: str, model: str = "gpt-4") -> str:
    llm = ChatOpenAI(model=model, openai_api_key=API_KEY)
    return llm.predict(prompt)

def generate_course_structure(course_title: str, syllabus: str, extra: str = "") -> Dict:
    context = retrieve_relevant_context(syllabus)
    methodology = load_methodology("structure")
    prompt = generate_prompt(context, methodology, course_title, extra)
    response = call_gpt(prompt, model="gpt-4")
    return {"structure": response, "context": context}

def generate_full_course(structure: str, course_title: str) -> str:
    blocks = ["announcements", "introduction", "conclusion", "quiz"]
    content = []

    for part in blocks:
        context = retrieve_relevant_context(course_title)
        methodology = load_methodology(part)
        prompt = generate_prompt(context, methodology, course_title)
        section = call_gpt(prompt, model="gpt-3.5-turbo")
        content.append(section)

    modules = [line for line in structure.split("\n") if line.lower().startswith("module")]
    for module_line in modules:
        module_name = module_line.strip()
        context = retrieve_relevant_context(module_name)
        methodology = load_methodology("modules")
        prompt = generate_prompt(context, methodology, module_name)
        module_content = call_gpt(prompt, model="gpt-3.5-turbo")
        content.append(f"\n## {module_name}\n{module_content}")

    return "\n\n".join(content)

def export_to_word(text: str) -> bytes:
    doc = WordDocument()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    from io import BytesIO
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()

def export_to_html(text: str) -> str:
    return "<html><body>" + "<br>".join(text.split("\n")) + "</body></html>"

# ================== UI ==================
st.title("📚 Moodle Course Generator")
st.markdown("Generuj kurzy na základě PDF a metodiky.")

if "structure" not in st.session_state:
    st.session_state.structure = None
if "context" not in st.session_state:
    st.session_state.context = None
if "course" not in st.session_state:
    st.session_state.course = None

uploaded_pdfs = st.file_uploader("Nahrát PDF materiály", type="pdf", accept_multiple_files=True)
syllabus_file = st.file_uploader("Nahrát PDF sylabus", type="pdf")
course_title = st.text_input("Název kurzu")
extra_info = st.text_area("Doplňující info (nepovinné)")

if st.button("🔍 Vygeneruj strukturu kurzu") and uploaded_pdfs and syllabus_file and course_title:
    full_text = "\n\n".join([extract_text_from_pdf(pdf) for pdf in uploaded_pdfs])
    chunks = split_into_chunks(full_text)
    embed_and_store_chunks(chunks)

    syllabus_text = extract_text_from_pdf(syllabus_file)
    result = generate_course_structure(course_title, syllabus_text, extra_info)
    st.session_state.structure = result["structure"]
    st.session_state.context = result["context"]

if st.session_state.structure:
    st.subheader("📑 Návrh struktury kurzu")
    edited = st.text_area("Návrh", st.session_state.structure, height=300)
    if st.button("✅ Potvrdit a vytvořit obsah"):
        st.session_state.course = generate_full_course(edited, course_title)

if st.session_state.course:
    st.subheader("📘 Generovaný kurz")
    st.markdown(st.session_state.course)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Stáhnout jako Word", export_to_word(st.session_state.course), file_name="course.docx")
    with col2:
        st.download_button("⬇️ Stáhnout jako HTML", export_to_html(st.session_state.course), file_name="course.html", mime="text/html")

    st.subheader("💬 Připomínky")
    feedback = st.text_area("Co se ti nelíbí?")
    if st.button("♻️ Přegenerovat podle připomínek") and feedback:
        prompt = f"{st.session_state.context}\n\nUživatelská připomínka: {feedback}\n\nPůvodní návrh:\n{st.session_state.course}"
        st.session_state.course = call_gpt(prompt, model="gpt-3.5-turbo")
