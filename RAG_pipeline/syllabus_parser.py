import pdfplumber
import re
from typing import Dict

def parse_syllabus(pdf_path: str) -> Dict[str, str]:
    with pdfplumber.open(pdf_path) as pdf:
        raw_text = "\n".join([page.extract_text() or "" for page in pdf.pages])

    # Rekonstrukce typických nadpisů (vložíme zpět mezery a dvojtečky)
    text = raw_text \
        .replace("CoursetitleinEnglish", "Course title in English:") \
        .replace("CoursetitleinCzech", "Course title in Czech:") \
        .replace("Nameoflecturer(s)", "Name of lecturer(s):") \
        .replace("Aimsofthecourse", "Aims of the course:") \
        .replace("Learningoutcomesandcompetences", "Learning outcomes and competences:") \
        .replace("Coursecontents", "Course contents:") \
        .replace("Assessmentmethodsandcriteria", "Assessment methods and criteria:")

    # Helper funkce pro extrakci mezi dvěma sekcemi
    def extract_between(start: str, end: str) -> str:
        pattern = rf"{start}\s*(.*?)(?=\n\s*{end})"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    # Přímé extrakce
    course_title = re.search(r"Course title in English:\s*(.*)", text)
    lecturers = re.search(r"Name of lecturer\(s\):\s*(.*)", text)

    # Blokové extrakce
    aims = extract_between("Aims of the course:", "Learning outcomes")
    learning_outcomes = extract_between("Learning outcomes(?: and competences)?:", "Course contents")
    course_contents = extract_between("Course contents:", "Learning activities")
    if not course_contents:
        course_contents = extract_between("Course contents:", "Assessment methods and criteria")
    grading_method = extract_between("Assessment methods and criteria:", "Assessment:")

    return {
        "course_name": course_title.group(1).strip() if course_title else "",
        "lecturers": lecturers.group(1).strip() if lecturers else "",
        "aims": aims,
        "learning_outcomes": learning_outcomes,
        "course_contents": course_contents,
        "grading_method": grading_method
    }
