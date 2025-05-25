# ContentEDU
School project for 'Data Project' subject

This application enables the generation of Moodle course content using GPT-4 and GPT-3.5 based on uploaded PDF syllabi and teaching materials.

## 💡 Features
- Upload PDF teaching materials and a syllabus
- Create a vector database (ChromaDB) from the documents
- Generate course structure using GPT-4
- Allow feedback on course structure
- Generate full chapter content based on methodology using GPT-3.5
- Export the generated course to Word or HTML
- Iterate course based on user feedback

## 🛠️ Requirements
- Python 3.9+
- OpenAI API key (stored in a `.env` file as `OPENAI_API_KEY`)

## 📁 Project Structure
```
project/
├── main.py
├── .env
├── requirements.txt
├── methodology-structure.txt
├── methodology-modules.txt
├── methodology-introduction.txt
├── methodology-announcements.txt
├── methodology-conclusion.txt
├── methodology-quiz.txt
```

## ▶️ Run the App
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Create a `.env` file and insert your OpenAI API key:
```
OPENAI_API_KEY=your_api_key
```
3. Launch the Streamlit app:
```
streamlit run main.py
```

## 🧠 Methodology
Each stage of generation (structure, modules, introduction, etc.) uses a separate `.txt` file with instructional guidance. These can be customized to tailor the course output.

## 📤 Export
The generated course can be downloaded as:
- `.docx` Word document
- `.html` web page

## 📞 Contact
For any questions, please contact the developer or project maintainer.
