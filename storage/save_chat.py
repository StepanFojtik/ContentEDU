# ContentEDU/storage/save_chat.py
import csv
from datetime import datetime

def save_chat_entry(csv_path, user_id, course_id, course_name, goal, file_path, prompt, response, exported=False):
    timestamp = datetime.now().isoformat()

    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            user_id,
            course_id,
            course_name,
            goal,
            file_path,
            prompt,
            response,
            timestamp,
            str(exported)
        ])
