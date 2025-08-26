@echo off
echo =======================================
echo   🚀 Starting Streamlit PDFBot Setup
echo =======================================

REM --- Check Python version
python --version

REM --- Move to project folder
cd /d C:\Users\mmahajan\Downloads\Mohit\ProjectO

REM --- Create virtual environment
if not exist pdfbot_env (
    echo Creating virtual environment...
    python -m venv pdfbot_env
)

REM --- Activate venv
call pdfbot_env\Scripts\activate

REM --- Upgrade pip
python -m pip install --upgrade pip

REM --- Install dependencies
pip install streamlit
pip install PyMuPDF
pip install langchain-text-splitters
pip install sentence-transformers
pip install faiss-cpu
pip install numpy

REM --- Extra grouped install (to avoid version hell)
pip install langchain streamlit faiss-cpu sentence-transformers pymupdf

REM --- Ollama check
ollama --version
ollama pull llama2
ollama list

REM --- Finally run your app
echo Starting Streamlit app...
streamlit run MissionJarvis.py

pause
