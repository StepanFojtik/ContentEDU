import os
import sys
import streamlit as st
import pdfplumber

# Add RAG_pipeline to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG_pipeline")))

from course_pipeline import run_full_course_pipeline, regenerate_with_feedback

# === UI ===
st.title("📘 Welcome to ContentEDU")
st.markdown("Upload the teaching materials (multiple PDFs) and one course syllabus. Then generate your structured course content.")

# Upload multiple teaching materials
uploaded_materials = st.file_uploader("📄 Upload teaching materials (used by the teacher)", type="pdf", accept_multiple_files=True)

# Upload one syllabus file
uploaded_syllabus = st.file_uploader("📑 Upload the course syllabus (with objectives, topics from INSIS)", type="pdf")

# Course name
course_name = st.text_input("📝 Course name", placeholder="e.g. Introduction to Artificial Intelligence")

# Generate button
generate = st.button("🚀 Generate Course")

# === Logic ===
if generate:
    if uploaded_materials and uploaded_syllabus and course_name:
        os.makedirs("data/courses", exist_ok=True)

        # Save all teaching materials
        material_paths = []
        for file in uploaded_materials:
            path = os.path.join("data", "courses", file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            material_paths.append(path)

        # Save syllabus
        syllabus_path = os.path.join("data", "courses", uploaded_syllabus.name)
        with open(syllabus_path, "wb") as f:
            f.write(uploaded_syllabus.getbuffer())

        st.success("✅ All files saved successfully.")

        # Extract text from syllabus
        def extract_text(path):
            with pdfplumber.open(path) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])

        syllabus_text = extract_text(syllabus_path)

        with st.spinner("🔍 Generating course content using GPT-4-mini..."):
            try:
                result = run_full_course_pipeline(
                    material_paths=material_paths,        # now a list of paths
                    syllabus_text=syllabus_text,
                    course_name=course_name
                )
                st.success("✅ Course content successfully generated!")

                st.subheader("📚 Generated Course Content")
                st.markdown(result)

                # Save session state
                st.session_state.update({
                    "original_content": result,
                    "material_paths": material_paths,
                    "syllabus_text": syllabus_text,
                    "course_name": course_name,
                })

                # Feedback section
                st.subheader("💬 Not satisfied? Request improvements")
                with st.form("feedback_form"):
                    feedback = st.text_area("What would you like to change or improve?", placeholder="e.g. Add more interactivity...")
                    regenerate = st.form_submit_button("🔁 Regenerate with feedback")

                if regenerate and feedback.strip():
                    with st.spinner("✏️ Regenerating..."):
                        try:
                            new_result = regenerate_with_feedback(
                                material_paths=st.session_state["material_paths"],
                                syllabus_text=st.session_state["syllabus_text"],
                                course_name=st.session_state["course_name"],
                                original_output=st.session_state["original_content"],
                                feedback=feedback.strip()
                            )
                            st.subheader("🆕 Regenerated Course Content")
                            st.markdown(new_result)
                            st.session_state["original_content"] = new_result
                        except Exception as e:
                            st.error(f"❌ Failed to regenerate: {e}")
                elif regenerate and not feedback.strip():
                    st.warning("✋ Please enter feedback before regenerating.")

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
    else:
        st.warning("📥 Please upload all teaching materials, one syllabus, and enter a course name.")
