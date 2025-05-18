import sys
import os
import streamlit as st

# 👇 Add ContentEDU (root) folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.embedder import process_pdf_to_embeddings


# Title of the app
st.title("📘 Welcome to ContentEDU")

st.markdown("Upload a course PDF, give it a name and goal, and we'll prepare the content for generation!")

# File upload
uploaded_file = st.file_uploader("📄 Upload course PDF", type="pdf")

# Course name input
course_name = st.text_input("📝 Course name", placeholder="e.g. Introduction to Artificial Intelligence")

# Course goal input
course_goal = st.text_area("🎯 What is the goal of this course?", placeholder="e.g. Teach students the basics of AI and its applications.")

# When all inputs are filled
if uploaded_file and course_name and course_goal:
    # Save uploaded PDF to local folder
    os.makedirs("data/courses", exist_ok=True)
    pdf_path = os.path.join("data", "courses", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File saved as: `{pdf_path}`")

    # Process the PDF
    with st.spinner("🔍 Extracting text and generating embeddings..."):
        chunks, embeddings = process_pdf_to_embeddings(pdf_path)

    # Show result
    if chunks and len(embeddings) > 0:
        st.subheader("📄 First Chunk Preview")
        st.code(chunks[0][:500], language="markdown")

        st.subheader("📐 First Embedding Vector (first 5 values)")
        st.write(embeddings[0][:5])

        st.success(f"✅ Processed {len(chunks)} chunks.")
        st.info(f"Embedding dimension: {len(embeddings[0])}")

        # Save values in session for next step
        st.session_state["pdf_path"] = pdf_path
        st.session_state["course_name"] = course_name
        st.session_state["course_goal"] = course_goal
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings

    else:
        st.error("❌ No text could be extracted from the PDF.")

else:
    st.info("📥 Please upload a PDF and fill in the course name and goal.")
