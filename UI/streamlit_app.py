import os
import sys
import streamlit as st
import pdfplumber

# 👇 Add RAG_pipeline to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG_pipeline")))

from course_pipeline import run_full_course_pipeline, regenerate_with_feedback

# === UI ===
st.title("📘 Welcome to ContentEDU")
st.markdown("Upload the teaching materials and course syllabus as PDFs. We'll generate structured course content using our methodology.")

# Upload PDFs
uploaded_material = st.file_uploader("📄 Upload teaching materials (used by the teacher)", type="pdf")
uploaded_syllabus = st.file_uploader("📑 Upload course syllabus (with objectives, structure, topics from INSIS)", type="pdf")

# Course name
course_name = st.text_input("📝 Course name", placeholder="e.g. Introduction to Artificial Intelligence")

# === Main logic ===
if uploaded_material and uploaded_syllabus and course_name:
    # Save PDFs locally
    os.makedirs("data/courses", exist_ok=True)
    material_path = os.path.join("data", "courses", uploaded_material.name)
    syllabus_path = os.path.join("data", "courses", uploaded_syllabus.name)

    with open(material_path, "wb") as f:
        f.write(uploaded_material.getbuffer())
    with open(syllabus_path, "wb") as f:
        f.write(uploaded_syllabus.getbuffer())

    st.success("✅ Both files saved successfully.")

    # Run embedding + generation pipeline
    with st.spinner("🔍 Processing both documents and generating course content using GPT-4-mini..."):
        try:
            result = run_full_course_pipeline(
                material_path=material_path,
                syllabus_path=syllabus_path,
                course_name=course_name
            )
            st.success("✅ Course content successfully generated!")

            st.subheader("📚 Generated Course Content")
            st.markdown(result)

            # Save for regeneration
            st.session_state.update({
                "original_content": result,
                "material_path": material_path,
                "syllabus_path": syllabus_path,
                "course_name": course_name,
            })

            # Feedback form
            st.subheader("💬 Not satisfied? Request improvements")
            with st.form("feedback_form"):
                feedback = st.text_area("What would you like to change or improve?", placeholder="e.g. Add more interactivity, remove quiz, expand intro...")
                regenerate = st.form_submit_button("🔁 Regenerate with feedback")

            if regenerate and feedback.strip():
                with st.spinner("✏️ Regenerating course content based on your feedback..."):
                    try:
                        new_result = regenerate_with_feedback(
                            material_path=st.session_state["material_path"],
                            syllabus_path=st.session_state["syllabus_path"],
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
    st.info("📥 Please upload both course materials and syllabus, and provide a course name.")
