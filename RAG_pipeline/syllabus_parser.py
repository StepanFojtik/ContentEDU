from striprtf.striprtf import rtf_to_text
from typing import Dict

def parse_syllabus(file_path: str) -> Dict[str, str]:
    # Load and clean text
    with open(file_path, "r", encoding="utf-8") as f:
        raw_rtf = f.read()
    text = rtf_to_text(raw_rtf)

    def extract_after(label: str) -> str:
        try:
            return text.split(label)[1].split("\n")[0].strip()
        except IndexError:
            return ""

    def extract_between(start_label: str, end_label: str) -> str:
        try:
            return text.split(start_label)[1].split(end_label)[0].strip()
        except IndexError:
            return ""

    def clean_multiline(txt: str) -> str:
        return txt.replace("|", "").strip()

    # Extract fields
    course_name = extract_after("Course title in English:")
    lecturers = extract_after("Name of lecturer(s):")
    aims = extract_between("Aims of the course:", "Learning outcomes and competences:")
    learning_outcomes = extract_between("Learning outcomes and competences:", "Course contents:")
    course_contents = extract_between("Course contents:", "Learning activities, teaching methods and workload (hours):")
    grading_method = extract_between("Assessment methods and criteria:", "Assessment:")

    return {
        "course_name": clean_multiline(course_name),
        "lecturers": clean_multiline(lecturers),
        "aims": clean_multiline(aims),
        "learning_outcomes": clean_multiline(learning_outcomes),
        "course_contents": clean_multiline(course_contents),
        "grading_method": clean_multiline(grading_method)
    }
