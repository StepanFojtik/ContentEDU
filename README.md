# ContentEDU

ContentEDU is a tool for generating structured digital courses based on a syllabus and teaching materials.  
It uses OpenAI models, LangGraph, and vector search to assist in creating Moodle-compatible content.  
The generation process is guided by FIS methodology templates to ensure alignment with academic standards.

## Features

- Step-by-step course generation via Streamlit UI
- Automatic parsing of `.rtf` syllabi
- Chunking and embedding of PDF materials
- AI-generated course structure, modules, quizzes, and conclusion
- Export to `.txt` and `.html` formats

## Getting Started

### Prerequisites

- Python 3.10 or higher
- OpenAI API key stored in a `.env` file:

```
OPENAI_API_KEY=your-key-here
```

### Installation

Install all required packages:

```
pip install -r requirements.txt
```

### Launch the App

Run the Streamlit frontend:

```
streamlit run streamlit_app.py
```

## Project Structure

```
.
├── streamlit_app.py          main frontend (Streamlit)
├── RAG_pipeline/
│   ├── course_pipeline.py    backend logic (LangGraph + LangChain)
│   └── syllabus_parser.py    syllabus (.rtf) processing
├── prompts/                  LLM prompt templates
├── data/                     uploaded input files
├── .env                      contains your API key (excluded from Git)
├── requirements.txt
└── README.md
```

## Notes

- Prompt templates can be modified in the `prompts/` directory
- This tool was developed as part of a university data project
- All data and keys stay local – nothing is sent externally
