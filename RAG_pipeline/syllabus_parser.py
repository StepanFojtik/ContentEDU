import pdfplumber
import re
from typing import Dict

def parse_syllabus(pdf_path: str) -> Dict[str, str]:
    # Open the syllabus PDF and extract full text
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])

    # Extract instructor(s)
    instructor_match = re.search(r"Name of lecturer\(s\):\s*(.+?)(?=\n\S|$)", text, re.DOTALL)
    instructor = instructor_match.group(1).strip() if instructor_match else ""

    # Extract learning outcomes (whole block until next section)
    learning_outcomes_match = re.search(
        r"Learning outcomes and competences:\s*((?:[\s\S]*?))(?=Assessment methods and criteria:|Recommended reading:|\n\S)",
    text
    )
    if learning_outcomes_match:
        learning_outcomes = learning_outcomes_match.group(1).strip()
    else:
        learning_outcomes = ""


    # Extract grading method / assessment methods
    grading_match = re.search(
        r"Assessment methods and criteria:\s*(.+?)(?=Recommended reading:|\Z)", 
        text, re.DOTALL
    )
    if grading_match:
        raw = grading_match.group(1)
        lines = [line.strip() for line in raw.strip().splitlines() if "%" in line]
        grading_method = "\n".join(lines)
    else:
        grading_method = ""


    # Extract course contents (as fallback "structure proxy" if needed)
    subject_content_match = re.search(
        r"Course contents:\s*(.+?)(?=\n\S|$)", text, re.DOTALL
    )
    subject_content = subject_content_match.group(1).strip() if subject_content_match else ""

    return {
        "instructor": instructor,
        "learning_outcomes": learning_outcomes,
        "grading_method": grading_method,
        "subject_content": subject_content
    }
