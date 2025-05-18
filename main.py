# project_structure/
# ├── documents/
# │   └── methodology.txt (tvůj metodologický soubor)
# ├── main.py
# └── requirements.txt

# ------------------ main.py -------------------
import os
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

# Globální nastavení
VECTOR_DB_PATH = "vectorstore"
API_KEY = ""  # <-- Sem vložíš své API klíče

# Načti metodologii
with open("documents/methodology.txt", "r", encoding="utf-8") as file:
    methodology_text = file.read()

# Funkce pro načítání dokumentů
def load_documents(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        documents.extend(loader.load())
    return documents

# Funkce pro vytvoření a uložení vektorové databáze
def create_and_store_vector_db(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

# Funkce pro načtení relevantního kontextu z vektorové DB
def retrieve_context(query):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    return context

# Funkce generující prompt
def generate_prompt(context, methodology, course_info):
    template = """
    Metodologie:
    {methodology}

    Kontext:
    {context}

    Informace o kurzu:
    {course_info}

    Na základě výše uvedených informací vytvoř strukturovaný obsah online kurzu do Moodle.
    """
    prompt = PromptTemplate(input_variables=["methodology", "context", "course_info"], template=template)
    return prompt.format(methodology=methodology, context=context, course_info=course_info)

# Získání obsahu kurzu z LLM
def get_course_content(prompt):
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    return chain.run({})

# Export do Word souboru
def export_to_word(content, filename="generated_course.docx"):
    from docx import Document

    doc = Document()
    doc.add_heading("Vygenerovaný kurz", 0)
    doc.add_paragraph(content)
    doc.save(filename)
    return filename

# Gradio frontend
def process(pdf_files, syllabus_file, course_title, additional_info):
    pdf_paths = [pdf.name for pdf in pdf_files]
    syllabus_path = syllabus_file.name

    # Zpracování dokumentů
    documents = load_documents(pdf_paths + [syllabus_path])
    create_and_store_vector_db(documents)

    # Vytvoření promptu
    context = retrieve_context(course_title)
    course_info = f"Název kurzu: {course_title}\nDalší informace: {additional_info or 'N/A'}"
    prompt = generate_prompt(context, methodology_text, course_info)

    # Získání odpovědi od LLM
    course_content = get_course_content(prompt)

    return course_content

# GUI aplikace
iface = gr.Interface(
    fn=process,
    inputs=[
        gr.File(label="Nahraj PDF dokumenty", file_count="multiple"),
        gr.File(label="Nahraj sylabus (PDF)"),
        gr.Textbox(label="Název kurzu"),
        gr.Textbox(label="Další informace (nepovinné)")
    ],
    outputs=gr.TextArea(label="Vygenerovaný obsah kurzu"),
    title="Generátor Moodle kurzů",
    allow_flagging="never"
)

iface.launch()
