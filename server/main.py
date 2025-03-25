from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader

from config import DEFAULT_MODEL
from server.ollama_client import query_ollama
from server.models.request import ChatRequest
from server.vectorstore import vector_store
from pathlib import Path

app = FastAPI()

# Allow CORS from your laptop or any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (You can restrict to your laptop IP later)
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def health():
    return {"status": "running", "model": DEFAULT_MODEL}

@app.post("/chat")
async def chat(request: ChatRequest):
    response = query_ollama(request.prompt, request.model, dev=True)
    return {"response": response}

@app.post("/add-docs")
def add_docs(docs: list[str]):
    vector_store.add_texts(docs)
    return {"status": "added", "count": len(docs)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    ext = Path(filename).suffix.lower()

    content = ""
    try:
        if ext == ".pdf":
            pdf = PdfReader(file.file)
            content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            content = (await file.read()).decode("utf-8")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read file: {str(e)}"})

    vector_store.add_text(content, source=filename)
    return {"status": "uploaded", "filename": filename}

@app.post("/search")
def search_docs(request: ChatRequest):
    results = vector_store.query(request.prompt)
    return {"matches": results}