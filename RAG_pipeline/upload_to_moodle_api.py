import os
import requests
from dotenv import load_dotenv

load_dotenv()

MOODLE_TOKEN = os.getenv("MOODLE_TOKEN")
MOODLE_URL = os.getenv("MOODLE_URL")

def call_moodle_api(function: str, **params):
    params.update({
        "wstoken": MOODLE_TOKEN,
        "moodlewsrestformat": "json",
        "wsfunction": function
    })
    response = requests.post(MOODLE_URL, data=params)
    response.raise_for_status()
    return response.json()

def upload_to_moodle(course_name, modules, intro, final_parts):
    course_data = {
        "fullname": course_name,
        "shortname": course_name.replace(" ", "_"),
        "categoryid": 1,  # Změň podle potřeby
        "summary": "This course was generated using ContentEDU.",
        "format": "topics",
        "numsections": len(modules) + 4  # intro + modules + quiz + conclusion
    }

    course_resp = call_moodle_api("core_course_create_courses", courses=[course_data])
    
    if isinstance(course_resp, list) and len(course_resp) > 0 and "id" in course_resp[0]:
        course_id = course_resp[0]["id"]
    else:
        raise ValueError(f"Unexpected Moodle response: {course_resp}")
    
    return {"id": course_id}

    st.write("Raw Moodle response:", course_resp)
