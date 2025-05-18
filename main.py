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
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from docx import Document

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
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_DB_PATH)
    vectorstore.persist()

# Funkce pro načtení relevantního kontextu z vektorové DB
def retrieve_context(query):
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
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
    doc = Document()
    doc.add_heading("Vygenerovaný kurz", 0)
    doc.add_paragraph(content)
    doc.save(filename)
    return filename

# Gradio frontend
conversation_history = []

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

    conversation_history.append({"dotaz": prompt, "odpoved": course_content})

    return course_content, conversation_history

def refine_course(feedback):
    if not conversation_history:
        return "Nejdříve vytvoř kurz.", conversation_history

    last_response = conversation_history[-1]['odpoved']
    new_prompt = f"Uprav předchozí obsah kurzu na základě této připomínky uživatele:\n\n{feedback}\n\nPůvodní obsah:\n{last_response}"
    refined_content = get_course_content(new_prompt)

    conversation_history.append({"dotaz": feedback, "odpoved": refined_content})

    return refined_content, conversation_history

iface = gr.Blocks()
with iface:
    gr.Markdown("# Generátor Moodle kurzů")

    with gr.Row():
        pdf_files = gr.File(label="Nahraj PDF dokumenty", file_count="multiple")
        syllabus_file = gr.File(label="Nahraj sylabus (PDF)")

    course_title = gr.Textbox(label="Název kurzu")
    additional_info = gr.Textbox(label="Další informace (nepovinné)")

    generate_btn = gr.Button("Generovat kurz")
    course_output = gr.TextArea(label="Vygenerovaný obsah kurzu")

    export_btn = gr.Button("Exportovat do Wordu")
    export_output = gr.File(label="Stáhnout kurz")

    feedback = gr.Textbox(label="Připomínky nebo úpravy kurzu")
    refine_btn = gr.Button("Upravit kurz dle připomínky")

    conversation_display = gr.JSON(label="Historie konverzace")

    generate_btn.click(process, [pdf_files, syllabus_file, course_title, additional_info], [course_output, conversation_display])
    export_btn.click(lambda content: export_to_word(content), inputs=course_output, outputs=export_output)
    refine_btn.click(refine_course, feedback, [course_output, conversation_display])

iface.launch()
