from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

model = SentenceTransformer("all-mpnet-base-v2")  # Embedding Model

class VectorStore:
    def __init__(self):
        self.texts = []
        self.metadata = []
        self.index = faiss.IndexFlatL2(768)  # 768 = embedding size for MiniLM

    def smart_chunk(self, text: str, chunk_size=500, overlap=100) -> list[str]:
        chunks = []
        current_section = "General"
        buffer = ""

        def is_heading(line):
            stripped = line.strip()
            return (
                stripped.startswith("*") or
                stripped.endswith(":") or
                (stripped.isupper() and len(stripped.split()) <= 6)
            )

        lines = text.splitlines()

        for line in lines:
            # If line looks like a heading, store it
            if is_heading(line):
                current_section = re.sub(r"[*:]", "", line.strip()).strip()

            buffer += line + "\n"

            if len(buffer) >= chunk_size:
                if current_section:
                    labeled = f"SECTION: {current_section}\n" + buffer
                else:
                    labeled = buffer
                chunks.append(labeled.strip())
                buffer = buffer[-overlap:]

        if buffer.strip():
            if current_section:
                buffer = f"SECTION: {current_section}\n" + buffer
            chunks.append(buffer.strip())

        return chunks


    def embed(self, texts: list[str]):
        return model.encode(texts).astype("float32")

    def add_text(self, text: str, source: str = "unknown"):
        chunks = self.smart_chunk(text)
        self.texts.extend(chunks)
        self.metadata.extend([{"source": source}] * len(chunks))
        embeddings = self.embed(chunks)
        self.index.add(embeddings)

    def add_texts(self, docs: list[str]):
        self.texts.extend(docs)
        self.metadata.extend([{"source": "bulk_add"}] * len(docs))
        embeddings = self.embed(docs)
        self.index.add(embeddings)

    def query(self, query: str, k: int = 5):
        if not self.texts:
            return ["I don't know. No documents were uploaded or available."]

        D, I = self.index.search(self.embed([query]), k)
        if all(i == -1 or D[0][idx] > 1.5 for idx, i in enumerate(I[0])):
            return ["⚠️ I couldn't find anything relevant in the context."]

        seen_chunks = set()
        results = []
        for i in I[0]:
            if i < len(self.texts):
                chunk = self.texts[i]
                if chunk not in seen_chunks and any(kw in chunk.lower() for kw in query.lower().split()):
                    seen_chunks.add(chunk)
                    source = self.metadata[i].get("source", "unknown")
                    results.append(f"[Source: {source}]\n{chunk}")
        return results

vector_store = VectorStore()
