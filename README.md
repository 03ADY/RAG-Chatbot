# 🤖 Enterprise RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that leverages **LangChain**, **Gemini Pro (Google Generative AI)**, and **ChromaDB** for context-aware, document-grounded question answering.

Built with **FastAPI** and styled with **TailwindCSS**, this full-stack application includes features like live chat, document re-indexing, markdown support, and Docker deployment.

---

## 🧩 Key Features

- 🔍 **RAG architecture** using LangChain + Gemini Pro
- 📚 Supports multi-format document ingestion: `.txt`, `.pdf`, `.docx`, `.csv`
- 🧠 In-memory **conversational context** with session support
- 🧾 Realtime **source attribution** and content snippets
- ♻️ **Re-indexing API** and UI button to reload documents
- ⚙️ **Health check endpoint** for CI/CD & monitoring
- 📄 Clean, mobile-responsive **TailwindCSS frontend**
- 📦 **Dockerized backend** for deployment ease
- ✅ Fully typed and validated API with **Pydantic models**
- 🔐 `.env` based secure configuration for **Google API Key**

---

## 📁 Project Structure

```bash
rag_chatbot_project/
├── backend/
│   ├── main.py              # FastAPI app with LangChain logic
│   ├── requirements.txt     # Dependencies
│   └── .env                 # Your GOOGLE_API_KEY
├── frontend/
│   └── index.html           # TailwindCSS-powered chat UI
├── data/                    # Store your knowledge base documents here
└── Dockerfile               # Containerization for backend
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag_chatbot_project
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```
GOOGLE_API_KEY=your_gemini_api_key
```

### 4. Run FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

---

### 5. Launch Frontend

Simply open `frontend/index.html` in your browser or use Live Server (VSCode).

---

## 🐳 Docker Deployment

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 rag-chatbot
```

---

## 🔌 API Overview

| Endpoint        | Method | Description                        |
|-----------------|--------|------------------------------------|
| `/chat`         | POST   | Send user message, get AI response |
| `/reindex_kb`   | POST   | Reload all knowledge base files    |
| `/health`       | GET    | Backend service status             |

---

## 🛠 Tech Stack

- **LangChain** — Retrieval chain, memory, text splitting
- **Gemini Pro** — Google’s GenAI LLM & Embedding API
- **ChromaDB** — Local vector database
- **FastAPI** — Backend REST framework
- **TailwindCSS** — UI styling
- **Docker** — Containerization
- **Pydantic** — Data validation

---

## 📸 Screenshots (Optional)

> Add screenshots or screen recordings of your app here to showcase the UI/UX.

---

## 📜 License

This project is open for educational and professional portfolio use. Feel free to fork and build upon it with attribution.

---

## 👨‍💻 Author

**Aayush Yadav**  
🔗 [Portfolio](https://aayush-yadav-portfolio.vercel.app) •  
🐙 [GitHub](https://github.com/03ADY)

---

_“Augment your AI with real-world knowledge. Build smarter assistants today.”_
