import os
import sys
import streamlit as st
import pdfplumber
import re
from html import escape

# Add the RAG_pipeline folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG_pipeline")))

# Import backend functions and parsers
from syllabus_parser import parse_syllabus
from course_pipeline import (
    extract_text_from_pdf,
    embed_and_store_chunks,
    split_into_chunks,
    generate_structure,
    generate_announcements_and_intro,
    retrieve_relevant_context,
    generate_final_parts,
    generate_module,
    build_syllabus_text,
    markdown_to_html
)

# === UI title ===
st.title("📘 Welcome to ContentEDU")

# === Initialize Streamlit session state ===
# Ensures all required session variables are defined before starting the app

def initialize_session_state():
    defaults = {
        'step': 1,
        'current_step': 1,
        'course_name': '',
        'material_paths': [],
        'syllabus_text': '',
        'module_topics': [],
        'modules': [],
        'current_module_idx': 0,
        'structure': None,
        'context': '',
        'announcements_intro': '',
        'current_module_content': None,
        'final_parts': ''
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state before anything else
initialize_session_state()

# === Step 1: Upload syllabus and teaching materials ===
# This step allows the user to upload course PDFs (syllabus + materials)
# The syllabus is parsed, materials are embedded, and everything is stored in session

if st.session_state.current_step == 1:
    st.header("📥 Step 1: Upload files")

    # Upload multiple teaching materials (PDFs) and one syllabus
    uploaded_materials = st.file_uploader("Upload teaching materials (multiple PDFs)", type="pdf", accept_multiple_files=True)
    uploaded_syllabus = st.file_uploader("Upload syllabus (.rtf only)", type="rtf")
    course_name = st.text_input("Course name", value=st.session_state.course_name)

    # Continue only if all inputs are filled
    if st.button("✅ Continue", key="step1_continue"):
        if uploaded_materials and uploaded_syllabus and course_name:
            os.makedirs("data/courses", exist_ok=True)

            # Save uploaded materials to disk and collect paths
            material_paths = []
            for file in uploaded_materials:
                path = os.path.join("data", "courses", file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                material_paths.append(path)

            # Save uploaded syllabus to disk
            syllabus_path = os.path.join("data", "courses", uploaded_syllabus.name)
            with open(syllabus_path, "wb") as f:
                f.write(uploaded_syllabus.getbuffer())

            # Parse syllabus and convert to formatted text
            syllabus_info = parse_syllabus(syllabus_path)
            syllabus_text = build_syllabus_text(syllabus_info)

            # Embed and store all uploaded materials
            for path in material_paths:
                text = extract_text_from_pdf(path)
                embed_and_store_chunks(split_into_chunks(text))

            # Save all inputs to session state
            st.session_state.course_name = course_name
            st.session_state.material_paths = material_paths
            st.session_state.syllabus_text = syllabus_text
            st.session_state.module_topics = []
            st.session_state.modules = []
            st.session_state.current_module_idx = 0

            # Proceed to the next step
            st.session_state.step = 2
            st.session_state.current_step = 2
            st.rerun()
        else:
            st.warning("Please upload all required files and enter course name.")

# === Step 2: Generate digital course structure ===
# Uses the parsed syllabus text to generate a proposed course structure
# The structure includes section/module titles and short descriptions

elif st.session_state.current_step == 2:
    st.header("📑 Step 2: Generate digital course structure")

    if st.button("🚀 Generate Structure", key="step2_generate"):
        with st.spinner("Generating digital course structure..."):
            # Call LLM to generate structure from syllabus
            structure = generate_structure(
                syllabus_text=st.session_state.syllabus_text,
                context=st.session_state.context
            )

            # Store result and proceed to the next step
            st.session_state.structure = structure
            st.session_state.step = 3
            st.session_state.current_step = 3
            st.rerun()

# === Step 3: Review and approve generated course structure ===
# Shows the AI-generated structure and lets the user either approve or request changes
# If approved, the app extracts module topics and retrieves relevant content from uploaded materials

elif st.session_state.current_step == 3:
    st.header("📋 Step 3: Review digital course structure")

    if st.session_state.structure:
        # Display suggested structure to the user
        st.markdown("### Suggested structure:")
        st.markdown(st.session_state.structure["text"])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Approve structure", key="step3_approve"):
                with st.spinner("🔍 Retrieving relevant context from materials..."):
                    
                    # Retrieve relevant information from embedded PDFs
                    context = retrieve_relevant_context(st.session_state.syllabus_text)
                    st.session_state.context = context

                    # Extract module topics from the structure (e.g., "Module 1 – Title")
                    structure_text = st.session_state.structure["text"]
                    topics = re.findall(r"Module \d+\s*[-–]\s*(.+)", structure_text)

                    # Save to session for next steps
                    st.session_state.module_topics = topics
                    st.session_state.modules = []
                    st.session_state.current_module_idx = 0

                # Advance to the next step
                st.session_state.step = 4
                st.session_state.current_step = 4
                st.rerun()

        with col2:
            if st.button("✏️ I want to change it", key="step3_change"):
                st.warning("Modification not implemented yet – this is where feedback would go.")

# === Step 4: Generate announcements and introduction ===
# Uses the syllabus and retrieved context to generate two initial course sections:
# "Announcements" (empty with placeholder note) and "Introduction" (course details + quiz)

elif st.session_state.current_step == 4:
    st.header("📣 Step 4: Generate announcements + introduction")

    if st.button("✍️ Generate Announcements + Intro", key="step4_generate"):
        with st.spinner("Generating..."):
            # Generate the first two sections of the course using prompt + context
            intro = generate_announcements_and_intro(
                syllabus_text=st.session_state.syllabus_text,
                context=st.session_state.context,
                structure_text=st.session_state.structure["text"]
            )

            # Save result and move to next step
            st.session_state.announcements_intro = intro
            st.session_state.step = 5
            st.session_state.current_step = 5
            st.rerun()

# === Step 5: Review and approve introduction section ===
# Displays the generated Announcements and Introduction so the user can approve or regenerate them

elif st.session_state.current_step == 5:
    st.header("🧾 Step 5: Review Announcements + Introduction")
    
    if st.session_state.announcements_intro:
        # Show the generated content
        st.markdown(st.session_state.announcements_intro)

        col1, col2 = st.columns(2)

        with col1:
            # Approve and proceed to module generation
            if st.button("✅ Approve Announcements + Intro", key="step5_approve"):
                st.session_state.step = 6
                st.session_state.current_step = 6
                st.rerun()
        with col2:
            # Go back and regenerate announcements + intro
            if st.button("🔁 Regenerate", key="step5_regenerate"):
                st.session_state.step = 4
                st.session_state.current_step = 4
                st.rerun()

# === Step 6: Generate and approve course modules ===
# Displays and generates each module one by one using the previously extracted topics

elif st.session_state.current_step == 6:
    st.header("📦 Step 6: Generate Modules")

    # Safety check – module topics must exist
    if not st.session_state.module_topics:
        st.error("❌ Module topics not loaded. Please restart from Step 1.")
        st.stop()
    
    topics = st.session_state.module_topics
    idx = st.session_state.current_module_idx

    if idx < len(topics):
        topic = topics[idx]
        st.subheader(f"📘 Module {idx + 1} – {topic}")

        # Generate current module content (if not already generated)
        if st.session_state.current_module_content is None:
            with st.spinner("Generating module content..."):
                content = generate_module(idx + 1, topic, st.session_state.context)
                st.session_state.current_module_content = content
        
        # Display generated content
        st.markdown(st.session_state.current_module_content)

        col1, col2 = st.columns(2)

        with col1:
            # Approve current module and move to the next one
            if st.button("✅ Approve this module", key=f"step6_approve_{idx}"):
                st.session_state.modules.append(st.session_state.current_module_content)
                st.session_state.current_module_idx += 1
                st.session_state.current_module_content = None  # Reset for next module
                
                # If this was the last module, proceed to final part
                if st.session_state.current_module_idx >= len(topics):
                    st.session_state.step = 7
                    st.session_state.current_step = 7
                st.rerun()
                
        with col2:
            # Regenerate current module
            if st.button("🔁 Regenerate this module", key=f"step6_regenerate_{idx}"):
                st.session_state.current_module_content = None
                st.rerun()

    else:
        # All modules have been approved
        st.success("✅ All modules approved. Moving to final part.")
        st.session_state.step = 7
        st.session_state.current_step = 7
        st.rerun()

# === Step 7: Generate final quiz and conclusion ===
# Uses the syllabus, course structure, and all approved modules to generate:
# (1) a comprehensive final quiz and (2) a concluding message for the course

elif st.session_state.current_step == 7:
    st.header("🧾 Step 7: Generate Final Quiz and Conclusion")

    if st.button("🧠 Generate Final Part", key="step7_generate"):

        with st.spinner("Generating..."):
            # Generate the final part using all previously collected inputs
            final = generate_final_parts(
                syllabus_text=st.session_state.syllabus_text,
                structure=st.session_state.structure,
                modules=st.session_state.modules,
                context=st.session_state.context
            )

            # Save and move to the final review/export step
            st.session_state.final_parts = final
            st.session_state.step = 8
            st.session_state.current_step = 8
            st.rerun()

# === Step 8: Review final part and export course ===
# Displays the final quiz and conclusion. User can approve it or export it as a downloadable HTML file

elif st.session_state.current_step == 8:
    st.header("🏁 Step 8: Review Final Part")
    
    if st.session_state.final_parts:
        # Display the generated final part
        st.markdown(st.session_state.final_parts)

        col1, col2, col3 = st.columns(3)

        with col1:
            # Final confirmation (no action, just visual feedback)
            if st.button("✅ Approve Final Part", key="step8_approve"):
                st.success("🎉 Course generation complete!")
        
        with col2:
            # .txt dowload
            if st.button("⬇️ Export to .txt", key="step8_export"):
                full_txt_course = (
                    f"{st.session_state.announcements_intro}\n\n"
                    + "\n\n".join(st.session_state.modules)
                    + f"\n\n{st.session_state.final_parts}"
                )
                st.download_button(
                    label="Download as .txt",
                    data=full_txt_course,
                    file_name="moodle_course.txt",
                    mime="text"
                )

        with col3:
            # HTML download
            if st.button("⬇️ Export to HTML", key="step9_export"):
                full_course = (
                    f"{st.session_state.announcements_intro}\n\n"
                    + "\n\n".join(st.session_state.modules)
                    + f"\n\n{st.session_state.final_parts}"
                )

                html_course = f"""
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>{escape(st.session_state.course_name)}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                        h1, h2, h3 {{ color: #003366; }}
                        ul {{ margin-left: 20px; }}
                        hr {{ margin: 30px 0; }}
                    </style>
                </head>
                <body>
                    <h1>{escape(st.session_state.course_name)}</h1>
                    {markdown_to_html(st.session_state.announcements_intro)}<hr>
                    {''.join(
                        f"<h2>Module {i+1}</h2>{markdown_to_html(mod)}<hr>"
                        for i, mod in enumerate(st.session_state.modules)
                    )}
                    <h2>Conclusion</h2>
                    {markdown_to_html(st.session_state.final_parts)}
                </body>
                </html>
                """

                st.download_button(
                    label="Download as HTML",
                    data=html_course,
                    file_name="moodle_course.html",
                    mime="text/html"
                )

# === Sidebar: step navigation ===
# Sidebar allows users to navigate between available steps in the app
# Step switching works independently of main button logic (placed at the end intentionally)

st.sidebar.title("📋 Steps")

# Mapping internal step numbers to readable labels
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

# Only display steps that have already been unlocked by the user
available_steps = [i for i in range(1, st.session_state.step + 1)]

# Render the sidebar radio button for navigation
selected_step = st.sidebar.radio(
    "Go to step:", 
    available_steps, 
    format_func=lambda x: steps_labels[x],
    index=available_steps.index(st.session_state.current_step) if st.session_state.current_step in available_steps else 0
)

# Update visible step only if user selects a different one
if selected_step != st.session_state.current_step:
    st.session_state.current_step = selected_step
    st.rerun()