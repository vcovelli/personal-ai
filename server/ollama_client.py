import requests
from server.vectorstore import vector_store

def query_ollama(prompt: str, model: str, dev: bool = False) -> str:
    # Retrieve relevant docs
    context_docs = vector_store.query(prompt)

    # Early exit if nothing useful was found
    if context_docs == ["⚠️ I couldn't find anything relevant in the context."]:
        return "⚠️ I couldn't find anything relevant to answer your question. Try uploading more specific files or rephrasing your question."

    # Format the prompt
    context = "\n".join(context_docs)

    full_prompt = f"""
You are a helpful assistant with access to project documentation and notes.
Your job is to answer user questions as accurately as possible using only the provided context.
Do not assume anything beyond what is given. The context may be partial.

If the answer is not clearly contained in the context, reply with:
"I don't know based on the provided context."

Keep your response factual, concise, and grounded in the context. Do not generate additional commentary.

CONTEXT:
```text
{context}
---

USER QUESTION:
{prompt}

ASSISTANT:
"""

    # Optional Dev output
    if dev:
        print("\n🧠 Prompt sent to model:\n", full_prompt)

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=60
        )
        return res.json()["response"]
    except Exception as e:
        return f"⚠️ Ollama error: {e}"
