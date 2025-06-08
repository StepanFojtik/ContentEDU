# === Syllabus parser ===
# Extracts key fields from an .rtf syllabus file and returns them as a dictionary

from striprtf.striprtf import rtf_to_text
from typing import Dict

# Parse selected fields from an RTF syllabus and return them as a dictionary
def parse_syllabus(file_path: str) -> Dict[str, str]:
    # Load RTF file and convert to plain text
    with open(file_path, "r", encoding="utf-8") as f:
        raw_rtf = f.read()
    text = rtf_to_text(raw_rtf)

    # Extract text after a given label (single-line field)
    def extract_after(label: str) -> str:
        try:
            return text.split(label)[1].split("\n")[0].strip()
        except IndexError:
            return ""

    # Extract text between two labels (multi-line block) 
    def extract_between(start_label: str, end_label: str) -> str:
        try:
            return text.split(start_label)[1].split(end_label)[0].strip()
        except IndexError:
            return ""

    # Clean unwanted characters from multiline fields
    def clean_multiline(txt: str) -> str:
        return txt.replace("|", "").strip()

    # Extract specific syllabus fields using the helper functions
    course_name = extract_after("Course title in English:")
    lecturers = extract_after("Name of lecturer(s):")
    aims = extract_between("Aims of the course:", "Learning outcomes and competences:")
    learning_outcomes = extract_between("Learning outcomes and competences:", "Course contents:")
    course_contents = extract_between("Course contents:", "Learning activities, teaching methods and workload (hours):")
    grading_method = extract_between("Assessment methods and criteria:", "Assessment:")

    # Return all cleaned fields as a dictionary
    return {
        "course_name": clean_multiline(course_name),
        "lecturers": clean_multiline(lecturers),
        "aims": clean_multiline(aims),
        "learning_outcomes": clean_multiline(learning_outcomes),
        "course_contents": clean_multiline(course_contents),
        "grading_method": clean_multiline(grading_method)
    }
