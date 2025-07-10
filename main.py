import os
import json
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from PyPDF2 import PdfReader

try:
    from docx import Document
except ImportError:
    raise ImportError("Install 'python-docx' using: pip install python-docx")

try:
    import faiss
except ImportError:
    raise ImportError("Install FAISS using: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Install 'sentence-transformers' using: pip install sentence-transformers")

# Load environment
load_dotenv()

# Constants
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# FastAPI app
app = FastAPI()

# Jira + Together config
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_AUTH = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
JIRA_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

class GenerateRequest(BaseModel):
    issue_key: str
    user_story: str
    project: str = "default"

# Helpers
def extract_text(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            reader = PdfReader(tmp_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif ext == ".docx":
            doc = Document(tmp_path)
            return "\n".join(p.text for p in doc.paragraphs)
    finally:
        os.remove(tmp_path)
    return ""

def chunk_text(text, size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def save_index(project: str, texts: List[str]):
    vecs = EMBEDDING_MODEL.encode(texts)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(np.array(vecs))
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{project}.index"))
    with open(os.path.join(TEXT_DIR, f"{project}.json"), "w") as f:
        json.dump(texts, f)

def load_index(project: str):
    index_path = os.path.join(INDEX_DIR, f"{project}.index")
    text_path = os.path.join(TEXT_DIR, f"{project}.json")
    if not os.path.exists(index_path) or not os.path.exists(text_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(text_path) as f:
        texts = json.load(f)
    return index, texts

def search_chunks(index, texts, query, k=5):
    vec = EMBEDDING_MODEL.encode([query])
    _, I = index.search(np.array(vec), k)
    return [texts[i] for i in I[0]]

def build_prompt(story, chunks):
    context = "\n".join(chunks)
    return f"""
You are a QA expert.
Generate test cases based on the user story and supporting context.

User Story:
{story}

Context:
{context}

Generate:
- 5 Functional
- 3 Negative
- 2 Edge cases
"""

def call_together_ai(prompt):
    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "messages": [
                {"role": "system", "content": "You are a QA expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# Routes
@app.post("/upload-docs")
async def upload_docs(project: str = Query("default"), files: List[UploadFile] = File(...)):
    all_chunks = []
    for file in files:
        text = extract_text(file)
        if text:
            all_chunks.extend(chunk_text(text))
    save_index(project, all_chunks)
    return {"message": f"{len(all_chunks)} chunks saved for project '{project}'."}

@app.post("/generate")
def generate(request: GenerateRequest):
    index, texts = load_index(request.project)
    if not index:
        return {"error": f"No vector index found for project '{request.project}'"}

    top_chunks = search_chunks(index, texts, request.user_story)
    prompt = build_prompt(request.user_story, top_chunks)
    result = call_together_ai(prompt)
    return {
        "issue_key": request.issue_key,
        "project": request.project,
        "used_chunks": top_chunks,
        "generated_test_cases": result
    }
