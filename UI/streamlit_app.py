# === Imports ===
# Standard libraries
import os
import sys
import re
from html import escape

# Third-party libraries
import streamlit as st
import pdfplumber

# === Add backend folder to import path ===
# Allows importing modules from the RAG_pipeline folder (one level up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG_pipeline")))

# === Import backend functions and helpers ===
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
    build_syllabus_text
)

# === UI title ===
# Displayed as the main title at the top of the Streamlit app
st.title("📘 Welcome to ContentEDU")

# === Initialize Streamlit session state ===
# Ensures all required session variables are defined before using the app
# Stores persistent values between interactions across app steps

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

# === Helper to reset downstream steps ===
# Clears all session data for steps after the given step number `n`
# This is used when the user decides to go back and regenerate earlier outputs

def reset_from_step(n):
    if n <= 4:
        st.session_state.announcements_intro = ''
    if n <= 5:
        st.session_state.modules = []
        st.session_state.current_module_idx = 0
    if n <= 7:
        st.session_state.final_parts = ''
    
    # Reset navigation and rerun app from the selected step
    st.session_state.step = n
    st.session_state.current_step = n
    st.rerun()
    
# === Utility: Convert markdown-like text to HTML for export ===
# Used in Step 8 to format the final course content into valid HTML
def markdown_to_html(md_text):
    html = md_text
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    html = re.sub(r'^- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'((<li>.*?</li>\s*)+)', r'<ul>\1</ul>', html, flags=re.DOTALL)
    html = html.replace("\n", "<br>")
    return html

# === Ensure session state is initialized before running the UI ===
initialize_session_state()

# === Step 1: Upload syllabus and teaching materials ===
# This step allows the user to upload course PDFs (syllabus + materials)
# The syllabus is parsed, materials are embedded, and everything is saved to session state

if st.session_state.current_step == 1:
    st.header("📥 Step 1: Upload files")

    # If files are already uploaded, offer to restart the process
    if st.session_state.material_paths and st.session_state.syllabus_text:
        st.warning("Files already uploaded. Restarting will erase all progress.")
        if st.button("🔁 Restart course creation", key="step1_restart"):
            reset_from_step(1)
    
    else:
        # File upload widgets
        uploaded_materials = st.file_uploader("Upload teaching materials (PDFs)", type="pdf", accept_multiple_files=True)
        uploaded_syllabus = st.file_uploader("Upload syllabus (.rtf only)", type="rtf")
        course_name = st.text_input("Course name", value=st.session_state.course_name)

        # When user clicks continue, check all inputs
        if st.button("✅ Continue", key="step1_continue"):
            if uploaded_materials and uploaded_syllabus and course_name:
                # Save uploaded files to local directory
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

                # Parse syllabus and extract structured text
                syllabus_info = parse_syllabus(syllabus_path)
                syllabus_text = build_syllabus_text(syllabus_info)

                # Process each PDF: extract, split, embed, and store chunks
                for path in material_paths:
                    text = extract_text_from_pdf(path)
                    embed_and_store_chunks(split_into_chunks(text))

                 # Save everything to session state
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
# Uses the parsed syllabus and relevant context to generate a proposed course structure
# The structure includes section/module titles and short descriptions

elif st.session_state.current_step == 2:
    st.header("📑 Step 2: Generate digital course structure")

    # If structure already exists, offer to continue or regenerate
    if st.session_state.structure:
        if st.session_state.step > 2:
            st.info("🛠️ Structure already generated. You can regenerate it to modify.")

        col1, col2 = st.columns(2)

        # Continue to next step
        with col1:
            if st.button("✅ Continue", key="step2_continue"):
                st.session_state.step = max(st.session_state.step, 3)
                st.session_state.current_step = 3
                st.rerun()
        
        # Regenerate structure from scratch
        with col2:
            if st.button("🔁 Regenerate", key="step2_regenerate"):
                st.session_state.structure = None
                reset_from_step(2)

    # If structure is not yet generated
    else:
        if st.button("🚀 Generate Structure", key="step2_generate"):
            with st.spinner("Generating digital course structure..."):
                structure = generate_structure(
                    syllabus_text=st.session_state.syllabus_text,
                    context=st.session_state.context
                )
                st.session_state.structure = structure
                st.session_state.step = max(st.session_state.step, 3)
                st.session_state.current_step = 3
                st.rerun()

# === Step 3: Edit and approve generated course structure ===
# Displays the AI-generated structure and allows the user to edit it manually

elif st.session_state.current_step == 3:
    st.header("📋 Step 3: Review digital course structure")

    # If structure is available
    if st.session_state.structure:
        if st.session_state.step > 3:
            st.info("ℹ️ Structure already generated. You can edit it or regenerate it.")

        # Allow user to edit the structure text directly
        st.markdown("### Edit the generated structure:")
        st.session_state.structure["text"] = st.text_area(
            "Course Structure",
            value=st.session_state.structure["text"],
            height=400,
            help="Edit the structure as needed"
        )

        col1, col2 = st.columns(2)
        
        # Continue to next step: retrieve context and extract topics
        with col1:
            if st.button("✅ Continue", key="step3_continue"):
                with st.spinner("🔍 Retrieving relevant context from materials..."):
                    # Retrieve context again (e.g. if structure was edited)
                    context = retrieve_relevant_context(st.session_state.syllabus_text)
                    st.session_state.context = context

                    # Extract module topics from the edited structure
                    topics = re.findall(r"Module \d+\s*[-–]\s*(.+)", st.session_state.structure["text"])
                    st.session_state.module_topics = topics
                    st.session_state.modules = []
                    st.session_state.current_module_idx = 0

                st.session_state.step = max(st.session_state.step, 4)
                st.session_state.current_step = 4
                st.rerun()

        # Option to regenerate structure from Step 2
        with col2:
            if st.button("🔁 Regenerate", key="step3_regenerate"):
                reset_from_step(2)

    # Fallback: no structure exists
    else:
        st.warning("❗ No structure found. Please generate it in Step 2.")

# === Step 4: Generate announcements and introduction ===
# Uses the syllabus and relevant context to generate the first two course sections

elif st.session_state.current_step == 4:
    st.header("📣 Step 4: Generate announcements + Introduction")

    # If content already exists, offer to continue or regenerate
    if st.session_state.announcements_intro:
        if st.session_state.step > 4:
            st.info("ℹ️ Announcements + Introduction already generated. You can regenerate them below.")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Continue", key="step4_continue"):
                st.session_state.step = max(st.session_state.step, 5)
                st.session_state.current_step = 5
                st.rerun()
        
        with col2:
            if st.button("🔁 Regenerate", key="step4_regenerate"):
                reset_from_step(4)

    # If content not yet generated, offer generation button
    else:
        if st.button("✍️ Generate Announcements + Introduction", key="step4_generate"):
            with st.spinner("Generating..."):
                intro = generate_announcements_and_intro(
                    syllabus_text=st.session_state.syllabus_text,
                    context=st.session_state.context,
                    structure_text=st.session_state.structure["text"]
                )
                st.session_state.announcements_intro = intro
                st.session_state.step = max(st.session_state.step, 5)
                st.session_state.current_step = 5
                st.rerun()

# === Step 5: Edit and approve introduction section ===
# Displays the generated "Announcements" and "Introduction" sections for manual editing and approval

elif st.session_state.current_step == 5:
    st.header("🧾 Step 5: Review announcements + Introduction")

    # If content is already generated
    if st.session_state.announcements_intro:
        if st.session_state.step > 5:
            st.info("ℹ️ Announcements + Introduction already generated.")

        # Allow user to edit the generated content directly
        st.markdown("### Edit the generated content:")
        st.session_state.announcements_intro = st.text_area(
            "Announcements + Introduction",
            value=st.session_state.announcements_intro,
            height=500,
            help="You can edit the content as needed"
        )

        col1, col2 = st.columns(2)
        
        # Continue to the next step
        with col1:
            if st.button("✅ Continue", key="step5_continue"):
                st.session_state.step = max(st.session_state.step, 6)
                st.session_state.current_step = 6
                st.rerun()
        
        # Regenerate previous step (Step 4)
        with col2:
            if st.button("🔁 Regenerate", key="step5_regenerate"):
                reset_from_step(4)

    # If introduction is missing (e.g. skipped previous step)
    else:
        st.warning("❗ No introduction found. Please generate it in Step 4.")

# === Step 6: Generate and edit course modules ===
# Generates one module at a time based on topics extracted from course structure
# Allows user to review, edit, and regenerate each module individually

elif st.session_state.current_step == 6:
    st.header("📦 Step 6: Modules")

    # If no topics are loaded, something went wrong earlier
    if not st.session_state.module_topics:
        st.error("❌ Module topics not loaded. Please restart from Step 1.")
        st.stop()

    topics = st.session_state.module_topics
    idx = st.session_state.current_module_idx

    # === Show previously generated modules (even if incomplete)
    if st.session_state.modules:
        st.markdown("### 🗂️ Previously Generated Modules")
        for i, module in enumerate(st.session_state.modules):
            st.session_state.modules[i] = st.text_area(
                f"✏️ Module {i+1}: {topics[i]}",
                value=module,
                height=300,
                key=f"module_review_{i}",
                help="You can still edit this module"
            )

    # === Generate and display the next module, if any are left
    if idx < len(topics):
        topic = topics[idx]
        st.markdown("---")
        st.subheader(f"📘 Module {idx + 1}: {topic}")

        # If not already generated, call backend function
        if st.session_state.current_module_content is None:
            with st.spinner("Generating module content..."):
                st.session_state.current_module_content = generate_module(
                    idx + 1, topic, st.session_state.context
                )

        # Allow user to edit the generated content
        st.markdown("### Edit the module content:")
        st.session_state.current_module_content = st.text_area(
            f"Module {idx + 1} Content",
            value=st.session_state.current_module_content,
            height=500,
            key=f"module_edit_{idx}"
        )

        col1, col2 = st.columns(2)
        
        # Save module and move to the next one
        with col1:
            if st.button("✅ Continue to next module", key=f"step6_continue_{idx}"):
                st.session_state.modules.append(st.session_state.current_module_content)
                st.session_state.current_module_idx += 1
                st.session_state.current_module_content = None
                st.rerun()

        # Regenerate the current module
        with col2:
            if st.button("🔁 Regenerate this module", key=f"step6_regenerate_{idx}"):
                st.session_state.current_module_content = None
                st.rerun()

    # === All modules complete – offer to continue or regenerate all
    elif idx >= len(topics):
        st.markdown("---")
        st.success("✅ All modules completed.")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Continue", key="step6_to_step7"):
                st.session_state.step = max(st.session_state.step, 7)
                st.session_state.current_step = 7
                st.rerun()

        with col2:
            if st.button("🔁 Regenerate All Modules", key="step6_regen_all"):
                st.session_state.modules = []
                st.session_state.final_parts = ''
                st.session_state.current_module_idx = 0
                st.session_state.current_module_content = None
                st.session_state.step = 6
                st.session_state.current_step = 6
                st.rerun()

# === Step 7: Generate final quiz and conclusion ===
# Uses syllabus, course structure, and all generated modules to create:
# (1) a comprehensive final quiz, and (2) a pedagogical conclusion section

elif st.session_state.current_step == 7:
    st.header("🧾 Step 7: Generate Final Quiz and Conclusion")

    # If content already exists, offer to continue or regenerate
    if st.session_state.final_parts:
        st.info("🧠 Final part already generated. You can regenerate or continue.")

        col1, col2 = st.columns(2)
        
        # Proceed to the next step
        with col1:
            if st.button("✅ Continue", key="step7_continue"):
                st.session_state.step = max(st.session_state.step, 8)
                st.session_state.current_step = 8
                st.rerun()
        
        # Clear final part and regenerate
        with col2:
            if st.button("🔁 Regenerate", key="step7_regenerate"):
                st.session_state.final_parts = ''
                reset_from_step(7)
    
    # If final part has not yet been generated
    else:
        if st.button("🧠 Generate Final Part", key="step7_generate"):
            with st.spinner("Generating..."):
                final = generate_final_parts(
                    syllabus_text=st.session_state.syllabus_text,
                    structure=st.session_state.structure,
                    modules=st.session_state.modules,
                    context=st.session_state.context
                )
                st.session_state.final_parts = final
                st.session_state.step = max(st.session_state.step, 8)
                st.session_state.current_step = 8
                st.rerun()

# === Step 8: Edit final part and export course ===
# Displays the final quiz and conclusion for editing and allows downloading the full course

elif st.session_state.current_step == 8:
    st.header("🏁 Step 8: Edit Final Part and Export")
    
    if st.session_state.final_parts:
        st.markdown("### Edit the final quiz and conclusion:")
        
        # Allow user to directly edit the generated content
        st.session_state.final_parts = st.text_area(
            "Final Quiz and Conclusion",
            value=st.session_state.final_parts,
            height=500,
            help="Edit the final quiz and conclusion as needed"
        )

        # Action buttons: regenerate or mark as complete
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔁 Regenerate Final Part", key="step8_regenerate"):
                st.session_state.step = 7
                st.session_state.current_step = 7
                st.rerun()

        with col2:
            if st.button("✅ Course Complete!", key="step8_complete"):
                st.success("🎉 Course generation complete!")

        # Export options
        st.markdown("---")
        st.markdown("### ⬇️ Export your course:")

        col1, col2 = st.columns(2)

        # Export as plain text (.txt)
        with col1:
            if st.button("⬇️ Export to .txt", key="step8_export_txt"):
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

        # Export as formatted HTML (.html)
        with col2:
            if st.button("⬇️ Export to HTML", key="step8_export_html"):
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
# Displays a sidebar for navigating between available steps in the app

st.sidebar.title("📋 Steps")

# Mapping internal step numbers to readable labels
steps_labels = {
    1: "Step 1 – Upload files",
    2: "Step 2 – Generate structure", 
    3: "Step 3 – Review structure",
    4: "Step 4 – Generate intro",
    5: "Step 5 – Review intro",
    6: "Step 6 – Modules",
    7: "Step 7 – Generate final part",
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