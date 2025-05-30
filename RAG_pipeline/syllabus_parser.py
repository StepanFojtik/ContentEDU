import pdfplumber
import re
from typing import Dict

def parse_syllabus(pdf_path: str) -> Dict[str, str]:
    # Open the syllabus PDF and extract full text
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])

    # Extract course name from the syllabus
    course_name_match = re.search(r"Název (?:česky|v jazyce výuky):\s*(.+)", text)
    course_name = course_name_match.group(1).strip() if course_name_match else ""

    # Extract instructor(s) – includes multiline values until another section starts
    instructor_match = re.findall(r"Vyučující:\s*((?:.+\n?)+?)(?:Omezení pro zápis|Doporučené doplňky kurzu|$)", text)
    instructor = instructor_match[0].strip() if instructor_match else ""

    # Extract learning outcomes (bullet-point style under a specific section)
    learning_outcomes_match = re.search(r"Po úspěšném absolvování.*?\n((?:- .+\n)+)", text)
    if learning_outcomes_match:
        lines = learning_outcomes_match.group(1).strip().splitlines()
        learning_outcomes = "\n".join(lines)
    else:
        learning_outcomes = ""

    # Extract grading method section 
    grading_match = re.search(
        r"Způsoby a kritéria hodnocení:.*?(Vypracování.*?Celkem\s+100\s*%)",
        text, re.DOTALL
    )
    grading_method = grading_match.group(1).strip() if grading_match else ""

    # Extract subject content (between section headers)
    subject_content_match = re.search(r"Obsah předmětu:(.*?)Způsob studia", text, re.DOTALL)
    subject_content = subject_content_match.group(1).strip() if subject_content_match else ""

    # Return extracted fields as a dictionary
    return {
        "course_name": course_name,
        "instructor": instructor,
        "learning_outcomes": learning_outcomes,
        "grading_method": grading_method,
        "subject_content": subject_content
    }
