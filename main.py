# project_structure/
# ├── .env
# ├── methodology-announcements.txt
# ├── methodology-conclusion.txt
# ├── methodology-introduction.txt
# ├── methodology-modules.txt
# ├── methodology-quiz.txt
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
import re

# ===== Načtení proměnných z .env souboru =====
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# ===== Konfigurace =====
VECTOR_DB_PATH = "vectorstore"  # cesta k vektorové databázi
CHUNK_SIZE = 1000               # velikost chunku při rozdělení textu
CHUNK_OVERLAP = 100             # překrytí chunků

# ===== Nastavení stránky ve Streamlitu =====
st.set_page_config(page_title="Course Generator", layout="wide")

# ================== FUNKCE ==================

# Extrakce textu z PDF souboru pomocí pdfplumber
def extract_text_from_pdf(pdf_file) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# Rozdělení textu na menší překrývající se chunky
def split_into_chunks(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents([text])

# Uložení chunků do vektorové databáze (Chroma + OpenAI embeddings)
def embed_and_store_chunks(docs: List[Document]):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_PATH)
    vectorstore.persist()

# Načtení relevantního kontextu z vektorové databáze podle dotazu
def retrieve_relevant_context(query: str, k: int = 5) -> str:
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])

# Načtení textového souboru s metodologií podle typu kapitoly
def load_methodology(name: str) -> str:
    path = f"methodology-{name.lower()}.txt"
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

# Sestavení promptu pro LLM na základě kontextu, metodologie a názvu kapitoly
def generate_prompt(context: str, methodology: str, title: str = "", extra: str = "") -> str:
    return f"""
{methodology}

---
{f"Additional Info: {extra}\n" if extra else ""}Relevant Teaching Content:
{context}

---
Generate the response strictly based on the methodology and provided content.
"""

# Odeslání promptu do OpenAI GPT modelu a získání odpovědi
def call_gpt(prompt: str, model: str = "gpt-4") -> str:
    llm = ChatOpenAI(model=model, openai_api_key=API_KEY)
    return llm.predict(prompt)

# Vygenerování návrhu struktury kurzu na základě sylabu a metodologie
def generate_course_structure(course_title: str, syllabus: str, extra: str = "") -> Dict:
    context = retrieve_relevant_context(syllabus)
    methodology = load_methodology("structure")
    prompt = generate_prompt(context, methodology, course_title, extra)
    response = call_gpt(prompt, model="gpt-4")
    return {"structure": response, "context": context}

# Určení typu kapitoly na základě názvu
def detect_chapter_type(title: str) -> str:
    title_lower = title.lower()
    if "announcement" in title_lower:
        return "announcements"
    elif "introduction" in title_lower:
        return "introduction"
    elif "conclusion" in title_lower:
        return "conclusion"
    elif "quiz" in title_lower:
        return "quiz"
    elif "module" in title_lower:
        return "modules"
    else:
        return "modules"  # fallback pro neznámé typy

# Extrakce jednotlivých kapitol ze struktury do seznamu slovníků (název + typ)
def extract_chapters(structure: str) -> List[Dict[str, str]]:
    lines = structure.strip().split("\n")
    chapters = []
    for line in lines:
        clean_line = line.strip("-• ")
        if len(clean_line) > 0:
            chapters.append({
                "title": clean_line,
                "type": detect_chapter_type(clean_line)
            })
    return chapters

# Generování obsahu celého kurzu podle potvrzené struktury a příslušné metodologie
def generate_full_course(structure: str, course_title: str) -> str:
    chapters = extract_chapters(structure)
    results = []

    for chapter in chapters:
        title = chapter["title"]
        chapter_type = chapter["type"]
        context = retrieve_relevant_context(title)
        methodology = load_methodology(chapter_type)
        prompt = generate_prompt(context, methodology, title=title)
        content = call_gpt(prompt, model="gpt-3.5-turbo")
        results.append(f"## {title}\n{content}")

    return "\n\n".join(results)

# Export vygenerovaného textu do Word dokumentu
def export_to_word(text: str) -> bytes:
    doc = WordDocument()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    from io import BytesIO
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()

# Export vygenerovaného textu do HTML formátu
def export_to_html(text: str) -> str:
    return "<html><body>" + "<br>".join(text.split("\n")) + "</body></html>"

# ================== UI ==================
st.title("📚 Moodle Course Generator")
st.markdown("Generuj kurzy na základě PDF a metodiky.")

# Inicializace proměnných stavu (session)
if "structure" not in st.session_state:
    st.session_state.structure = None
if "context" not in st.session_state:
    st.session_state.context = None
if "course" not in st.session_state:
    st.session_state.course = None

# Nahrávání souborů od uživatele
uploaded_pdfs = st.file_uploader("Nahrát PDF materiály", type="pdf", accept_multiple_files=True)
syllabus_file = st.file_uploader("Nahrát PDF sylabus", type="pdf")
course_title = st.text_input("Název kurzu")
extra_info = st.text_area("Doplňující info (nepovinné)")

# Vygenerování návrhu struktury kurzu
if st.button("🔍 Vygeneruj strukturu kurzu") and uploaded_pdfs and syllabus_file and course_title:
    full_text = "\n\n".join([extract_text_from_pdf(pdf) for pdf in uploaded_pdfs])
    chunks = split_into_chunks(full_text)
    embed_and_store_chunks(chunks)

    syllabus_text = extract_text_from_pdf(syllabus_file)
    result = generate_course_structure(course_title, syllabus_text, extra_info)
    st.session_state.structure = result["structure"]
    st.session_state.context = result["context"]

# Uživatel potvrdí nebo upraví návrh struktury
if st.session_state.structure:
    st.subheader("📑 Návrh struktury kurzu")
    edited = st.text_area("Návrh", st.session_state.structure, height=300)
    if st.button("✅ Potvrdit a vytvořit obsah"):
        st.session_state.course = generate_full_course(edited, course_title)

# Zobrazení vygenerovaného kurzu a exportní funkce
if st.session_state.course:
    st.subheader("📘 Generovaný kurz")
    st.markdown(st.session_state.course)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Stáhnout jako Word", export_to_word(st.session_state.course), file_name="course.docx")
    with col2:
        st.download_button("⬇️ Stáhnout jako HTML", export_to_html(st.session_state.course), file_name="course.html", mime="text/html")

    # Uživatel může zadat připomínku k úpravě výstupu
    st.subheader("💬 Připomínky")
    feedback = st.text_area("Co se ti nelíbí?")
    if st.button("♻️ Přegenerovat podle připomínek") and feedback:
        prompt = f"{st.session_state.context}\n\nUživatelská připomínka: {feedback}\n\nPůvodní návrh:\n{st.session_state.course}"
        st.session_state.course = call_gpt(prompt, model="gpt-3.5-turbo")
