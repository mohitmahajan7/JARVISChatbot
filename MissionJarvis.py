import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
import subprocess
import streamlit as st
from datetime import datetime
import random
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
PDF_FOLDER = r"C:/Users/mmahajan/Downloads/Mohit/ProjectO/PDFs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "pdf_index.faiss"
TEXTS_FILE = "texts.pkl"
OLLAMA_MODEL = "llama2:latest"  # pulled local model

# -----------------------------
# PERSONALITY SETUP
# -----------------------------
def get_greeting():
    hour = datetime.now().hour
    base = " I am JARVIS here at your service. I have got the sarcasm dial set to 90%. "
    joke = "Shall we fire up the arc reactor for some Enovia/3DEXPERIENCE/MQL wizardry (policies, lifecycles, object models—your corporate Infinity Stones), or do you want to wrestle Spring Boot demons again?"
    if hour < 12:
        return f"Good morning, Mr. Stark. {base}{joke}"
    elif hour < 18:
        return f"Good afternoon, Mr. Stark. {base}{joke}"
    else:
        return f"Good evening, Mr. Stark. {base}{joke}"

def witty_opening():
    jokes = [
        "Initializing sarcasm module… Running 3DExperience boot sequence… Did you know ENOVIA was originally MATRIXONE before Dassault acquired it in 2006?",
        "Checking servers… ah yes, they are as slow as usual.",
    ]
    return random.choice(jokes)

# -----------------------------
# STEP 1: Extract & Chunk PDFs
# -----------------------------
def load_and_chunk_pdfs():
    all_texts = []
    for pdf_file in os.listdir(PDF_FOLDER):
        if pdf_file.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, pdf_file)
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            all_texts.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for text in all_texts:
        chunks.extend(splitter.split_text(text))
    return chunks

# -----------------------------
# STEP 2: Build or Load Vector DB
# -----------------------------
def build_or_load_faiss(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)

    if os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, "rb") as f:
            texts = pickle.load(f)
            st.write(f"Loaded {len(texts)} chunks from saved FAISS index.")
        return index, texts, model

    embeddings = model.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(chunks, f)
        st.write(f"Built FAISS index with {len(chunks)} chunks.")
        
    return index, chunks, model

# -----------------------------
# STEP 3: Query PDF Bot
# -----------------------------
def query_pdf_bot(query, index, texts, model, k=5):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k)
    results = [texts[i] for i in I[0]]
    context = "\n---\n".join(results)
    return context

# -----------------------------
# STEP 4: Streamlit Interface
# -----------------------------
st.set_page_config(page_title="Mission Jarvis")
st.title("Project JARVIS by Mohit")

# Greeting (once per session)
if "intro_shown" not in st.session_state:
    st.markdown(f"<p style='font-size:16px;'> {get_greeting()}</p>", unsafe_allow_html=True)
    st.markdown(f"**Current time: {datetime.now().strftime('%I:%M %p')}**")
    st.markdown(f"*{witty_opening()}*")
    st.session_state["intro_shown"] = True

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for role, content in st.session_state["messages"]:
    st.chat_message(role).markdown(content)

# Load PDFs & Vector DB
with st.spinner("Loading PDFs and building vector database..."):
    if os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, "rb") as f:
            texts = pickle.load(f)
        model = SentenceTransformer(EMBEDDING_MODEL)
    else:
        chunks = load_and_chunk_pdfs()
        index, texts, model = build_or_load_faiss(chunks)

# User input
if prompt := st.chat_input("What can I do for you, Mr. Stark?"):
    st.session_state["messages"].append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    # Retrieve PDF context
    context = query_pdf_bot(prompt, index, texts, model)

    # Construct final prompt
    full_prompt = (
        "You are JARVIS, the AI assistant of Iron Man."
        "Always address the user as Mr. Stark. "
        f"PDF Context:\n{context}\n\n"
        f"User Question: {prompt}"
    )

    #  FIXED: Pass full_prompt directly to Ollama
    command = ["ollama", "run", OLLAMA_MODEL, full_prompt]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        answer = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        answer = f"Ollama error: {e.stderr}"
    except Exception as e:
        answer = f"Unexpected error: {e}"

    # Show answer & save to chat
    st.session_state["messages"].append(("assistant", answer))
    st.chat_message("assistant").markdown(answer)
