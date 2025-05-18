# ContentEDU/storage/load_chat.py
import csv

def load_conversation(csv_path, course_id):
    history = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['course_id'] == course_id:
                history.append(row)
    return history
