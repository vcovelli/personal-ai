from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config import DEFAULT_MODEL
from server.ollama_client import query_ollama
from server.models.request import ChatRequest
from server.vectorstore import vector_store

app = FastAPI()

# Allow CORS from your laptop or any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (You can restrict to your laptop IP later)
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: ChatRequest):
    response = query_ollama(request.prompt, request.model)
    return {"response": response}

@app.get("/")
def health():
    return {"status": "running", "model": DEFAULT_MODEL}

@app.post("/add-docs")
def add_docs(docs: list[str]):
    vector_store.add_texts(docs)
    return {"status": "added", "count": len(docs)}

@app.post("/search")
def search_docs(request: ChatRequest):
    results = vector_store.query(request.prompt)
    return {"matches": results}