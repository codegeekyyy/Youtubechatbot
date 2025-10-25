# Youtubechatbot
YouTube Video Chatbot — a Streamlit app that fetches YouTube transcripts, splits text with RecursiveCharacterTextSplitter, creates Ollama embeddings, indexes chunks in FAISS, and answers user queries via LangChain. Local-first LLM support (Ollama) enables fast, private retrieval-augmented conversation.


🧠 Project Overview

The chatbot extracts a video transcript using the youtube-transcript-api, splits it into smaller chunks, and embeds each chunk using Ollama Embeddings.
These embeddings are stored in a FAISS vector database, allowing efficient retrieval of the most relevant transcript sections when a user asks a question.

The Ollama LLM (e.g., gemma3:4b) is then used to generate an accurate, context-aware response based on the retrieved transcript snippets.

A simple Streamlit-based frontend lets users:

Enter a YouTube video URL

View transcript chunks

Ask questions related to the video content

Get detailed, AI-generated answers derived directly from the transcript

⚙️ Tech Stack

Python 3.10+

LangChain (core logic)

LangChain Community & Ollama

FAISS (vector storage)

YouTube Transcript API (video transcript extraction)

Streamlit (frontend UI)

Ollama Models (nomic-embed-text, gemma3:4b)

🧩 Features

✅ Extracts transcripts from YouTube videos
✅ Splits and embeds text into FAISS vector storage
✅ Uses Ollama models for embeddings and generation
✅ Implements RAG for contextual Q&A
✅ Simple and interactive Streamlit interface

🗂️ Folder Structure
YouTubeChatbotusingLangChain/
│
├── app.py                     # Streamlit frontend
├── yt.py                      # Backend RAG logic
├── .gitignore                 # Ignore unnecessary files
├── requirements.txt           # Project dependencies
├── venv/                      # Virtual environment (ignored)
└── README.md                  # Project documentation

🧰 Installation and Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/YouTubeChatbotusingLangChain.git
cd YouTubeChatbotusingLangChain

2️⃣ Create a Virtual Environment
python -m venv venv

3️⃣ Activate the Virtual Environment

Windows:

venv\Scripts\activate


macOS/Linux:

source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt


Example dependencies:

youtube-transcript-api
langchain
langchain-community
langchain-ollama
faiss-cpu
streamlit
python-dotenv

🚀 Run the Application
Backend (Data Processing)
python yt.py

Frontend (Streamlit UI)
streamlit run app.py

💡 Example Usage

Launch the Streamlit app.

Enter a YouTube video URL (e.g., https://www.youtube.com/watch?v=Gfr50f6ZBvo).

Wait for the transcript to load and process.

Ask any question about the video — e.g.,
“What is DeepMind?”
The bot will provide an accurate, transcript-based answer.

📚 How It Works

Transcript Extraction: Fetches captions using the YouTube Transcript API.

Chunking: Splits text using RecursiveCharacterTextSplitter.

Embedding: Converts text chunks into embeddings via OllamaEmbeddings.

Vector Storage: Stores vectors using FAISS for fast similarity search.

Retrieval: Retrieves top-k chunks relevant to the question.

Generation: Feeds context + question to Ollama LLM for final answer.

🧑‍💻 Author

Harshdeep singh
B.Tech Student, Pranveer Singh Institute of Technology
Focusing on Artificial Intelligence and Machine Learning

📜 License

This project is licensed under the MIT License — feel free to modify and use it for your learning or projects.

🌟 Acknowledgments

LangChain

Ollama

FAISS

Streamlit

YouTube Transcript API
