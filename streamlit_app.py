import streamlit as st
import tempfile
import os
import re
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
    st.session_state.modules_output = []
if "conclusion_output" not in st.session_state:
    st.session_state.conclusion_output = ""
if "intro_output" not in st.session_state:
    st.session_state.intro_output = ""
if "modules_headers" not in st.session_state:
    st.session_state.modules_headers = []
if "current_module_index" not in st.session_state:
    st.session_state.current_module_index = 0

# === Step 1: Upload ===
with st.form("course_input_form"):
    uploaded_materials = st.file_uploader("Upload course PDF materials", type="pdf", accept_multiple_files=True)
    syllabus_file = st.file_uploader("Upload course syllabus (PDF)", type="pdf")
    course_name = st.text_input("Course name")
    additional_note = st.text_area("Optional: Add extra course info")
    submitted = st.form_submit_button("Generate Structure")

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
    st.subheader("Review Generated Structure")
    st.text_area("Generated Structure", st.session_state.structure_output, height=300)

    feedback = st.text_area("Suggest changes to structure")
    if st.button("Regenerate Structure") and feedback:
        context = retrieve_relevant_context(st.session_state.syllabus_text)
        methodology = load_methodology("methodology-structure.txt")
        system_prompt = load_methodology("system_prompt.txt")

        new_prompt = f"{system_prompt}\n\n{methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nPreviously Generated Output:\n{st.session_state.structure_output}\n\nUser Feedback:\n{feedback}"

        st.session_state.structure_output = get_course_content(new_prompt)
    if st.button("Accept Structure"):
        # === Extract Module Sections ===
        module_pattern = r"(?<=\n)\d+\. \*\*Module (\d+) – (.*?)\*\*\n\s*- (.*?)\n"
        matches = re.findall(module_pattern, st.session_state.structure_output, re.DOTALL)

        st.session_state.modules_headers = [f"Module {m[0]} – {m[1]}" for m in matches]
        st.session_state.modules_list = [f"{m[0]}. **Module {m[0]} – {m[1]}**\n- {m[2]}" for m in matches]
        st.session_state.step = 3

# === Step 3: Generate Modules One-by-One ===
if st.session_state.step == 3:
    current_index = st.session_state.current_module_index
    headers = st.session_state.modules_headers

    if current_index < len(headers):
        st.subheader(f"Generate Content for {headers[current_index]}")
        module_methodology = load_methodology("methodology-modules.txt")
        system_prompt = load_methodology("system_prompt.txt")
        context = retrieve_relevant_context(st.session_state.syllabus_text)
        quiz_context = load_methodology("methodology-quiz.txt")

        prompt = f"{system_prompt}\n\n{quiz_context}\n\n{module_methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nModule Header:\n{headers[current_index]}"

        module_content = get_course_content(prompt)
        st.session_state.modules_output.append(module_content)

        st.text_area(f"Generated content for {headers[current_index]}", module_content, height=300)
        feedback = st.text_area("Suggest changes to this module")
        if st.button("Regenerate Module") and feedback:
            prompt += f"\n\nUser Feedback:\n{feedback}"
            updated = get_course_content(prompt)
            st.session_state.modules_output[-1] = updated
            st.rerun()
        if st.button("Accept Module"):
            st.session_state.current_module_index += 1
            st.rerun()
    else:
        st.session_state.step = 4
        st.rerun()

# === Step 4: Conclusion ===
if st.session_state.step == 4:
    st.subheader("Generate and Review Conclusion")
    conc_methodology = load_methodology("methodology-conclusionAndFinQuiz.txt")
    system_prompt = load_methodology("system_prompt.txt")
    context = retrieve_relevant_context(st.session_state.syllabus_text)
    quiz_context = load_methodology("methodology-quiz.txt")

    prompt = f"{system_prompt}\n\n{quiz_context}\n\n{conc_methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nModules:\n{chr(10).join(st.session_state.modules_output)}"

    st.session_state.conclusion_output = get_course_content(prompt)
    st.text_area("Generated Conclusion", st.session_state.conclusion_output, height=300)
    feedback = st.text_area("Suggest changes to conclusion")
    if st.button("Regenerate Conclusion") and feedback:
        prompt += f"\n\nUser Feedback:\n{feedback}"
        st.session_state.conclusion_output = get_course_content(prompt)
        st.rerun()
    if st.button("Accept Conclusion"):
        st.session_state.step = 5
        st.rerun()

# === Step 5: Generate Introduction ===
if st.session_state.step == 5:
    st.subheader("Generate Announcement & Introduction")
    intro_methodology = load_methodology("methodology-AnnouncmentAndIntroduction.txt")
    system_prompt = load_methodology("system_prompt.txt")
    context = retrieve_relevant_context(st.session_state.syllabus_text)

    prompt = f"{system_prompt}\n\n{intro_methodology}\n\nCourse Name: {st.session_state.course_name}\n\nRelevant Context:\n{context}\n\nModules:\n{chr(10).join(st.session_state.modules_output)}\n\nConclusion:\n{st.session_state.conclusion_output}"

    st.session_state.intro_output = get_course_content(prompt)
    st.text_area("Generated Announcement & Introduction", st.session_state.intro_output, height=300)
    feedback = st.text_area("Suggest changes to intro")
    if st.button("Regenerate Intro") and feedback:
        prompt += f"\n\nUser Feedback:\n{feedback}"
        st.session_state.intro_output = get_course_content(prompt)
        st.rerun()
    if st.button("Accept Intro"):
        st.session_state.step = 6
        st.rerun()

# === Step 6: Final Output ===
if st.session_state.step == 6:
    st.subheader("✅ Final Moodle Course")
    full_course = f"{st.session_state.intro_output}\n\n" + "\n\n".join(st.session_state.modules_output) + f"\n\n{st.session_state.conclusion_output}"
    st.text_area("Full Course", full_course, height=600)
    st.download_button("Download as .txt", full_course, file_name="moodle_course.txt")
    st.success("Course generation complete.")
