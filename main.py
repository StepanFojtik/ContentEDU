import os 
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from docx import Document

VECTOR_DB_PATH = "vectorstore"
API_KEY = "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"
conversation_history = []

def load_documents(paths):
    docs = []
    for path in paths:
        loader = PyMuPDFLoader(path)
        docs.extend(loader.load())
    return docs

def create_vector_db(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=VECTOR_DB_PATH)
    vectorstore.persist()

def get_retriever():
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":12})

def generate_course_outline(course_title, methodology, additional_info):
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo")
    outline_prompt = f"""
    Název kurzu: {course_title}
    Metodologie: {methodology}
    Specifikace: {additional_info or "Bez dalších informací."}

    Vytvoř pouze osnovu kurzu – jasnou hierarchii modulů, sekcí a lekcí.
    Nepiš zatím žádný obsah lekcí, jen názvy a strukturu.
    """
    response = llm.predict(outline_prompt)
    return response

def generate_lesson_contents(outline, retriever):
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo")
    course_with_content = ""
    for line in outline.split("\n"):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.lower().startswith("název"):
            course_with_content += line + "\n"
            continue
        
        relevant_docs = retriever.get_relevant_documents(stripped_line)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        content_prompt = f"""
        Na základě následující osnovy: "{stripped_line}"
        a relevantního kontextu z dokumentů:
        {context}

        Vytvoř obsah této kapitoly kurzu – včetně teorie, definic, příkladů a praktických aplikací.
        """
        lesson_content = llm.predict(content_prompt)
        course_with_content += f"{line}\n{lesson_content}\n\n"
    return course_with_content

def export_to_word(content, filename="generated_course.docx"):
    doc = Document()
    doc.add_heading("Vygenerovaný kurz", 0)
    doc.add_paragraph(content)
    doc.save(filename)
    return filename

def export_to_html(content, filename="generated_course.html"):
    html_content = f"""<!DOCTYPE html>
<html lang='cs'>
<head>
    <meta charset='UTF-8'>
    <title>Vygenerovaný kurz</title>
</head>
<body>
    <h1>Vygenerovaný kurz</h1>
    <pre>{content}</pre>
</body>
</html>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    return filename

def process(pdf_files, syllabus_file, course_title, additional_info):
    docs = load_documents([f.name for f in pdf_files] + [syllabus_file.name, "documents/methodology.txt"])
    create_vector_db(docs)
    retriever = get_retriever()

    with open("documents/methodology.txt", "r", encoding="utf-8") as file:
        methodology = file.read()

    course_outline = generate_course_outline(course_title, methodology, additional_info)
    full_course = generate_lesson_contents(course_outline, retriever)

    conversation_history.append({"dotaz": course_outline, "odpoved": full_course})
    return full_course, course_outline, conversation_history

def refine_course(feedback):
    if not conversation_history:
        return "Nejdříve vytvoř kurz.", conversation_history

    last_response = conversation_history[-1]['odpoved']
    new_prompt = f"Uprav předchozí obsah kurzu na základě připomínky uživatele:\n\n{feedback}\n\nPůvodní obsah:\n{last_response}"
    retriever = get_retriever()
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo")
    refined_content = llm.predict(new_prompt)

    conversation_history.append({"dotaz": feedback, "odpoved": refined_content})
    return refined_content, conversation_history

# Gradio UI
with gr.Blocks() as iface:
    gr.Markdown("# Generátor Moodle kurzů")

    with gr.Row():
        pdf_files = gr.File(label="Nahraj PDF dokumenty", file_count="multiple")
        syllabus_file = gr.File(label="Nahraj sylabus (PDF)")

    course_title = gr.Textbox(label="Název kurzu")
    additional_info = gr.Textbox(label="Další informace (nepovinné)")

    generate_btn = gr.Button("Generovat kurz")
    course_output = gr.TextArea(label="Vygenerovaný obsah kurzu")
    context_output = gr.TextArea(label="Struktura kurzu (osnova)")

    export_btn = gr.Button("Exportovat do Wordu")
    export_output = gr.File(label="Stáhnout kurz (DOCX)")

    export_html_btn = gr.Button("Exportovat do HTML")
    export_html_output = gr.File(label="Stáhnout kurz (HTML)")

    feedback = gr.Textbox(label="Připomínky nebo úpravy kurzu")
    refine_btn = gr.Button("Upravit kurz dle připomínky")

    conversation_display = gr.JSON(label="Historie konverzace")

    generate_btn.click(process, [pdf_files, syllabus_file, course_title, additional_info], [course_output, context_output, conversation_display])
    export_btn.click(lambda content: export_to_word(content), inputs=course_output, outputs=export_output)
    export_html_btn.click(lambda content: export_to_html(content), inputs=course_output, outputs=export_html_output)
    refine_btn.click(refine_course, feedback, [course_output, conversation_display])

iface.launch()
