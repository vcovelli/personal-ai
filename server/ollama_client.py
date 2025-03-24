import requests
from server.vectorstore import vector_store

def query_ollama(prompt: str, model: str) -> str:
    # Retrieve relevant docs
    context_docs = vector_store.query(prompt)
    context = "\n".join(context_docs)

    full_prompt = f"""Use the following context to answer the question:

Context:
{context}

Question:
{prompt}
"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=60
        )
        return res.json()["response"]
    except Exception as e:
        return f"⚠️ Ollama error: {e}"
