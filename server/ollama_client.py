import requests
from server.vectorstore import vector_store
from server.redis_client import r

def query_ollama(prompt: str, model: str, dev: bool = False) -> str:
    cache_key = f"{model}:{prompt}"
    cached = r.get(cache_key)
    if cached:
        return f"🧠 (cached)\n{cached}"
    
    # Retrieve relevant docs
    context_docs = vector_store.query(prompt)

    # Early exit if nothing useful was found
    if context_docs == ["⚠️ I couldn't find anything relevant in the context."]:
        return "⚠️ I couldn't find anything relevant to answer your question. Try uploading more specific files or rephrasing your question."

    # Format the prompt
    context = "\n".join(context_docs)

    example_block = """
### Example:
USER: What is in this file?

CONTEXT:
[Source: ExampleDoc.txt]
SECTION: Compute
Details about how Snowflake uses virtual warehouses.

SECTION: File Formatting
Settings such as COMPRESSION, BINARY_FORMAT, and NULL_IF.

SECTION: Best Practices
Recommendations for version control, remote state, and planning workflows.

ASSISTANT:
The document includes:
- **Compute** – How Snowflake uses warehouses for processing
- **File Formatting** – Optional parameters for loading data
- **Best Practices** – Tips for managing infrastructure with versioning and state
"""

    full_prompt = f"""
You are a helpful technical assistant trained to explain documentation and notes clearly and naturally.
Answer the user's question based only on the CONTEXT below.

If the answer is not directly stated in the context, do your best to summarize what the context implies, but do not make anything up.
Answer ONLY about the topic the user is asking for.
If other topics are also in context, ignore them unless directly relevant.

Be clear, concise, and friendly. Use examples or bullet points if helpful. Format your answer with markdown if the context allows.

{example_block}

Now answer the following:

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
            timeout=90
        )
        response = res.json()["response"]
        r.set(cache_key, response)
        return response
    except Exception as e:
        return f"⚠️ Ollama error: {e}"
