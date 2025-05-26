import streamlit as st
import tempfile
import os
from course_pipeline import (
    extract_text_from_pdf,
    retrieve_relevant_context,
    get_course_content,
    load_methodology,
    split_into_chunks,
    embed_and_store_chunks
)

st.set_page_config(page_title="AI Moodle Course Generator", layout="wide")
st.title("📘 Moodle Course Generator with Feedback Loop")

# === Session State Setup ===
if "step" not in st.session_state:
    st.session_state.step = 1
if "structure_output" not in st.session_state:
    st.session_state.structure_output = ""
if "modules_output" not in st.session_state:
    st.session_state.modules_output = ""
if "conclusion_output" not in st.session_state:
    st.session_state.conclusion_output = ""
if "intro_output" not in st.session_state:
    st.session_state.intro_output = ""

# === Step 1: Input ===
st.header("Step 1: Upload Files & Course Info")
with st.form("course_input_form"):
    uploaded_materials = st.file_uploader("Upload course PDF materials", type="pdf", accept_multiple_files=True)
    syllabus_file = st.file_uploader("Upload course syllabus (PDF)", type="pdf")
    course_name = st.text_input("Enter course name")
    additional_note = st.text_area("Optional: Add extra course info")
    submitted = st.form_submit_button("Generate Course Structure")

if submitted and uploaded_materials and syllabus_file and course_name:
    with tempfile.TemporaryDirectory() as tmpdir:
        material_paths = []
        full_text = ""

        for file in uploaded_materials:
            path = os.path.join(tmpdir, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            material_paths.append(path)
            full_text += extract_text_from_pdf(path) + "\n\n"

        syllabus_path = os.path.join(tmpdir, syllabus_file.name)
        with open(syllabus_path, "wb") as f:
            f.write(syllabus_file.read())
        syllabus_text = extract_text_from_pdf(syllabus_path)

        # 🔥 Vložíme embeddingy do ChromaDB
        chunks = split_into_chunks(full_text)
        embed_and_store_chunks(chunks)

        context = retrieve_relevant_context(syllabus_text)
        methodology = load_methodology("methodology-structure.txt")
        system_prompt = load_methodology("system_prompt.txt")

        prompt = f"{system_prompt}\n\n{methodology}\n\nCourse Name: {course_name}\n\nRelevant Context:\n{context}\n\nAdditional Info:\n{additional_note}"

        st.session_state.structure_output = get_course_content(prompt)
        st.session_state.material_paths = material_paths
        st.session_state.syllabus_text = syllabus_text
        st.session_state.course_name = course_name
        st.session_state.step = 2

# === Step 2: Review Structure ===
if st.session_state.step == 2:
    st.header("Step 2: Review Course Structure")
    st.text_area("Generated Structure", st.session_state.structure_output, height=300)

    feedback = st.text_area("Suggest changes to structure")
    if st.button("Regenerate Structure") and feedback:
        context = retrieve_relevant_context(st.session_state.syllabus_text)
        methodology = load_methodology("methodology-structure.txt")
        system_prompt = load_methodology("system_prompt.txt")

        new_prompt = f"{system_prompt}\n\n{methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nPreviously Generated Output:\n{st.session_state.structure_output}\n\nUser Feedback:\n{feedback}"

        st.session_state.structure_output = get_course_content(new_prompt)
    if st.button("Accept Structure"):
        st.session_state.step = 3

# === Step 3: Generate Modules ===
if st.session_state.step == 3:
    st.header("Step 3: Generate Module Content")
    module_methodology = load_methodology("methodology-modules.txt")
    system_prompt = load_methodology("system_prompt.txt")
    context = retrieve_relevant_context(st.session_state.syllabus_text)

    modules_prompt = f"{system_prompt}\n\n{module_methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nCourse Structure:\n{st.session_state.structure_output}"

    st.session_state.modules_output = get_course_content(modules_prompt)
    st.session_state.step = 4
    st.rerun()

# === Step 4: Modules Review ===
if st.session_state.step == 4:
    st.header("Step 4: Review Modules")
    st.text_area("Generated Modules", st.session_state.modules_output, height=400)
    feedback = st.text_area("Suggest changes to modules")
    if st.button("Regenerate Modules") and feedback:
        module_methodology = load_methodology("methodology-modules.txt")
        system_prompt = load_methodology("system_prompt.txt")
        context = retrieve_relevant_context(st.session_state.syllabus_text)

        regen_prompt = f"{system_prompt}\n\n{module_methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nStructure:\n{st.session_state.structure_output}\n\nGenerated Modules:\n{st.session_state.modules_output}\n\nUser Feedback:\n{feedback}"

        st.session_state.modules_output = get_course_content(regen_prompt)
    if st.button("Accept Modules"):
        st.session_state.step = 5

# === Step 5: Conclusion ===
if st.session_state.step == 5:
    st.header("Step 5: Generate Conclusion")
    conc_methodology = load_methodology("methodology-conclusionAndFinQuiz.txt")
    system_prompt = load_methodology("system_prompt.txt")
    context = retrieve_relevant_context(st.session_state.syllabus_text)

    prompt = f"{system_prompt}\n\n{conc_methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nModules:\n{st.session_state.modules_output}\n\nQuiz:\n{st.session_state.quiz_output}"

    st.session_state.conclusion_output = get_course_content(prompt)
    st.session_state.step = 6
    st.rerun()

# === Step 6: Generate Introduction ===
if st.session_state.step == 6:
    st.header("Step 6: Generate Announcement & Introduction")
    intro_methodology = load_methodology("methodology-AnnouncmentAndIntroduction.txt")
    system_prompt = load_methodology("system_prompt.txt")
    context = retrieve_relevant_context(st.session_state.syllabus_text)

    prompt = f"{system_prompt}\n\n{intro_methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nModules:\n{st.session_state.modules_output}\n\nQuiz:\n{st.session_state.quiz_output}\n\nConclusion:\n{st.session_state.conclusion_output}"

    st.session_state.intro_output = get_course_content(prompt)
    st.session_state.step = 7
    st.rerun()

# === Step 7: Final Output ===
if st.session_state.step == 7:
    st.header("✅ Final Moodle Course")
    full_course = f"{st.session_state.intro_output}\n\n{st.session_state.modules_output}\n\n{st.session_state.conclusion_output}"
    st.text_area("Full Course", full_course, height=600)
    st.download_button("Download as .txt", full_course, file_name="moodle_course.txt")
    st.success("Course generation complete.")
