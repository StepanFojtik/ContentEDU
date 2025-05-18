# ContentEDU/test_storage.py
from storage.save_chat import save_chat_entry
from storage.load_chat import load_conversation
from storage.utils import generate_user_id, generate_course_id

csv_path = "data/conversations.csv"
user_id = generate_user_id()
course_id = generate_course_id()

save_chat_entry(
    csv_path=csv_path,
    user_id=user_id,
    course_id=course_id,
    course_name="AI Basics",
    goal="Teach foundations of AI",
    file_path="data/courses/ai_course.pdf",
    prompt="Create week 1",
    response="Week 1: Introduction to AI..."
)

history = load_conversation(csv_path, course_id)
for entry in history:
    print(entry['timestamp'], "|", entry['prompt'], "→", entry['response'])
