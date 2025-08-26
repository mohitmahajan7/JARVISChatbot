# JARVISChatbot
Project Jarvis: AI-powered assistant using Ollama, RAG, and FAISS VectorDB for intelligent task automation, document management, and project support."

# Project Jarvis

**Project Jarvis** is an AI-powered assistant designed to automate tasks, manage documents, and support projects with intelligence and speed.  
It leverages **Ollama**, **RAG (Retrieval-Augmented Generation)**, and **FAISS** for efficient semantic search, contextual awareness, and knowledge retrieval.

---

## 🚀 Features
- **Task Automation** – Streamline repetitive tasks with AI-driven workflows.  
- **Document Intelligence** – Parse, chunk, and query large documents seamlessly.  
- **Semantic Search with Faiss** – Fast vector-based similarity search across data.  
- **RAG-Powered Insights** – Accurate, context-aware responses using retrieval-augmented generation.  
- **Ollama Integration** – Run large language models locally without relying on cloud services.  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- [Ollama](https://ollama.ai) – Local LLM runner  
- [LangChain](https://www.langchain.com/) – Orchestration framework  
- [Faiss](https://github.com/facebookresearch/faiss) – Vector similarity search  
- [SentenceTransformers](https://www.sbert.net/) – Embeddings generation  
- [Streamlit](https://streamlit.io) – Interactive UI for experiments  
- [PyMuPDF](https://pymupdf.readthedocs.io) – PDF processing  

---

## 📂 Project Structure
Project-Jarvis/
│── data/ # PDFs and other documents
│── embeddings/ # Stored FAISS indexes & embeddings
│── src/
│ ├── chunking.py # Document chunking and preprocessing
│ ├── embeddings.py # Embedding generation
│ ├── rag_pipeline.py # RAG query pipeline
│ ├── ui.py # Streamlit-based frontend
│── requirements.txt # Python dependencies
│── README.md # Project documentation

---

## ⚡ Getting Started

### 1. Clone the Repository
2. Create Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
pip install -r requirements.txt
3. Run Ollama
Install Ollama and pull your preferred model:

ollama run llama2
4. Launch the Streamlit App
streamlit run src/ui.py
🔍 Example Workflow
Place PDFs inside the data/ folder.

Jarvis automatically chunks and embeds documents.

Ask contextual questions through the Streamlit interface.

Jarvis retrieves answers using RAG + Faiss.

📌 Roadmap
 Add support for JSON, DOCX, and CSV ingestion

 Advanced metadata-based search

 Multi-model Ollama integration (Mistral, Llama3, etc.)

 Deployment on cloud (AWS/Azure)

🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to open a PR or start a discussion.
