# ContentEDU/storage/utils.py
import uuid
from datetime import datetime

def generate_user_id():
    return "u_" + uuid.uuid4().hex[:8]

def generate_course_id():
    return "course_" + uuid.uuid4().hex[:8]

def current_timestamp():
    return datetime.now().isoformat()