import os
import sys
import streamlit as st
import pdfplumber
import re


# Cesta k backend pipeline
# === Přidání cesty k RAG_pipeline ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG_pipeline")))

from course_pipeline import (
    extract_text_from_pdf,
    embed_and_store_chunks,
    split_into_chunks,
    generate_structure,
    generate_announcements_and_intro,
    retrieve_relevant_context,
)

# === UI title ===
st.title("📘 Welcome to ContentEDU")

# === Step state initialization ===
# 'step' = highest unlocked step so far
# 'current_step' = currently viewed step (e.g. from sidebar navigation)
if "step" not in st.session_state:
    st.session_state.step = 1
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

# === Sidebar navigation ===
st.sidebar.title("📋 Steps")

# Mapping step numbers to labels
steps_labels = {
    1: "Step 1 – Upload files",
    2: "Step 2 – Generate structure",
    3: "Step 3 – Review structure",
    4: "Step 4 – Generate intro",
    5: "Step 5 – Review intro",
    6: "Step 6 – Modules",
    7: "Step 7 – Final part",
    8: "Step 8 – Review & export"
}

# Only show steps the user has already unlocked
available_steps = [i for i in range(1, st.session_state.step + 1)]

# Sidebar lets user navigate back to earlier steps
selected_step = st.sidebar.radio("Go to step:", available_steps, format_func=lambda x: steps_labels[x])

# Set current_step, but prevent jumping forward to unavailable steps
if selected_step <= st.session_state.step:
    st.session_state.current_step = selected_step
else:
    st.session_state.current_step = st.session_state.step

# === Krok 1: Upload ===
if st.session_state.current_step == 1:
    st.header("📥 Step 1: Upload files")

    uploaded_materials = st.file_uploader("Upload teaching materials (multiple PDFs)", type="pdf", accept_multiple_files=True)
    uploaded_syllabus = st.file_uploader("Upload syllabus (PDF)", type="pdf")
    course_name = st.text_input("Course name")

    if st.button("✅ Continue"):
        if uploaded_materials and uploaded_syllabus and course_name:
            os.makedirs("data/courses", exist_ok=True)

            material_paths = []
            for file in uploaded_materials:
                path = os.path.join("data", "courses", file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                material_paths.append(path)

            syllabus_path = os.path.join("data", "courses", uploaded_syllabus.name)
            with open(syllabus_path, "wb") as f:
                f.write(uploaded_syllabus.getbuffer())

            # Extract syllabus text
            syllabus_text = extract_text_from_pdf(syllabus_path)

            # Embed and store materials (real implementation should do real chunking + embedding)
            for path in material_paths:
                text = extract_text_from_pdf(path)
                # TODO: Replace with real chunking + embedding
                embed_and_store_chunks(split_into_chunks(text))

            # Save to session
            st.session_state.course_name = course_name
            st.session_state.material_paths = material_paths
            st.session_state.syllabus_text = syllabus_text
            st.session_state.module_topics = []  # seznam názvů modulů (extrahovaných později)
            st.session_state.modules = []        # hotové vygenerované moduly
            st.session_state.current_module_idx = 0  # začínáme od prvního modulu


            st.session_state.step = 2
            st.session_state.current_step = 2  # 👈 přepne zobrazení na Step 2
            st.rerun()
        else:
            st.warning("Please upload all required files and enter course name.")

# === Krok 2: Generate structure ===
elif st.session_state.current_step == 2:
    st.header("📑 Step 2: Generate course structure")

    if st.button("🚀 Generate Structure"):
        with st.spinner("Generating course structure..."):
            structure = generate_structure(st.session_state.syllabus_text)
            st.session_state.structure = structure
            st.session_state.step = 3
            st.session_state.current_step = 3  # 👈 nutné pro zobrazení dalšího kroku
            st.rerun() 


# === Krok 3: Show structure and ask for approval ===
elif st.session_state.current_step == 3:
    st.header("📋 Step 3: Review course structure")

    st.markdown("### Suggested structure:")
    st.markdown(st.session_state.structure["text"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Approve structure"):
            with st.spinner("🔍 Retrieving relevant context from materials..."):
                # 1. Získání relevantního kontextu
                context = retrieve_relevant_context(st.session_state.syllabus_text)
                st.session_state.context = context

                # 2. Extrakce témat modulů ze struktury
                structure_text = st.session_state.structure["text"]
                topics = re.findall(r"Module \d+\s*[-–]\s*(.+)", structure_text)

                # 3. Uložení do session pro další kroky
                st.session_state.module_topics = topics
                st.session_state.modules = []
                st.session_state.current_module_idx = 0

            st.session_state.step = 4
            st.session_state.current_step = 4  # 👈 přepne zobrazení na Step 2
            st.rerun()

    with col2:
        if st.button("✏️ I want to change it"):
            st.warning("Modification not implemented yet – this is where feedback would go.")

# === Krok 4: Generate announcements + intro ===
elif st.session_state.current_step == 4:
    st.header("📣 Step 4: Generate announcements + introduction")

    if st.button("✍️ Generate Announcements + Intro"):
        with st.spinner("Generating..."):
            intro = generate_announcements_and_intro(
                syllabus_text=st.session_state.syllabus_text,
                context=st.session_state.context
            )
            st.session_state.announcements_intro = intro
            st.session_state.step = 5
            st.session_state.current_step = 5  # 👈 přepne zobrazení na Step 2
            st.rerun()

# === Krok 5: Review announcements + intro ===
elif st.session_state.current_step == 5:
    st.header("🧾 Step 5: Review Announcements + Introduction")
    st.markdown(st.session_state.announcements_intro)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Announcements + Intro"):
            st.session_state.step = 6
            st.session_state.current_step = 6  # 👈 zobrazí další krok
            st.rerun()  # 👈 provede okamžitý přechod
    with col2:
        if st.button("🔁 Regenerate"):
            st.session_state.step = 4
            st.session_state.current_step = 4
            st.rerun()  # zpět na generování

# === Krok 6: ... (příprava na moduly) ===
elif st.session_state.current_step == 6:
    st.header("📦 Step 6: Modules – coming next...")
    st.info("Tady bude generování modulů po jednom – připraveno v dalším kroku.")

    #kontrola, jestli module_topics v session existuji
    if "module_topics" not in st.session_state or not st.session_state.module_topics:
        st.error("❌ Module topics not loaded. Please restart from Step 1.")
        st.stop()
    
    
    topics = st.session_state.module_topics
    idx = st.session_state.current_module_idx

    if idx < len(topics):
        topic = topics[idx]
        st.subheader(f"📘 Module {idx + 1} – {topic}")

        if "current_module_content" not in st.session_state:
            from course_pipeline import generate_module
            with st.spinner("Generating module content..."):
                content = generate_module(idx + 1, topic, st.session_state.context)
                st.session_state.current_module_content = content

        st.markdown(st.session_state.current_module_content)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve this module"):
                st.session_state.modules.append(st.session_state.current_module_content)
                st.session_state.current_module_idx += 1
                del st.session_state.current_module_content  # připravit další modul
                if st.session_state.current_module_idx >= len(topics):
                    st.session_state.step = 7  # pokračuj na závěrečné části
                st.rerun()
        with col2:
            if st.button("🔁 Regenerate this module"):
                del st.session_state.current_module_content
                st.rerun()

    else:
        st.success("✅ All modules approved. Moving to final part.")
        st.session_state.step = 7
        st.session_state.current_step = 7  # 👈 zajistí přechod na další krok
        st.rerun()
# === Krok 7: Generování závěrečné části (Final Quiz + Conclusion) ===
elif st.session_state.current_step == 7:
    st.header("🧾 Step 7: Generate Final Quiz and Conclusion")

    if st.button("🧠 Generate Final Part"):
        from course_pipeline import generate_final_parts

        with st.spinner("Generating..."):
            final = generate_final_parts(
                syllabus_text=st.session_state.syllabus_text,
                structure=st.session_state.structure,
                modules=st.session_state.modules,
                context=st.session_state.context
            )
            st.session_state.final_parts = final
            st.session_state.step = 8
            st.session_state.current_step = 8  # 👈 přepne na další krok
            st.rerun()
# === Krok 8: Náhled závěrečné části a upload/export ===
elif st.session_state.current_step == 8:
    st.header("🏁 Step 8: Review Final Part")
    st.markdown(st.session_state.final_parts)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Approve Final Part"):
            st.success("🎉 Course generation complete!")

    with col2:
        if st.button("🚀 Upload to Moodle"):
            from upload_to_moodle_api import upload_to_moodle
            with st.spinner("Uploading course to Moodle..."):
                result = upload_to_moodle(
                    course_name=st.session_state.course_name,
                    modules=st.session_state.modules,
                    intro=st.session_state.announcements_intro,
                    final_parts=st.session_state.final_parts
                )
                st.success(f"✅ Course successfully uploaded to Moodle! Course ID: {result['id']}")

    with col3:
        if st.button("⬇️ Export to HTML"):
            # Poskládání HTML obsahu
            html_content = "<html><head><meta charset='UTF-8'><title>Course Export</title></head><body>"
            html_content += "<h1>Announcements + Introduction</h1>"
            html_content += f"<div>{st.session_state.announcements_intro}</div>"

            html_content += "<h1>Modules</h1>"
            for i, module in enumerate(st.session_state.modules):
                html_content += f"<h2>Module {i + 1}</h2><div>{module}</div>"

            html_content += "<h1>Final Part</h1>"
            html_content += f"<div>{st.session_state.final_parts}</div>"
            html_content += "</body></html>"

            st.download_button(
                label="Download HTML file",
                data=html_content,
                file_name="course_export.html",
                mime="text/html"
            )



