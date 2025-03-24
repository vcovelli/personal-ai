from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast

class VectorStore:
    def __init__(self):
        self.texts = []
        self.index = faiss.IndexFlatL2(384)  # 384 = embedding size for MiniLM

    def chunk_text(self, text: str, chunk_size: int = 300) -> list[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def embed(self, texts: list[str]):
        return model.encode(texts).astype("float32")

    def add_texts(self, docs: list[str]):
        self.texts.extend(docs)
        embeddings = model.encode(docs)
        self.index.add(np.array(embeddings).astype("float32"))

    def query(self, query: str, k: int = 3):
        if not self.texts:
            return ["⚠️ No documents available."]
        D, I = self.index.search(self.embed([query]), k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def add_text(self, text: str):
        chunks = self.chunk_text(text)
        self.texts.extend(chunks)
        embeddings = self.embed(chunks)
        self.index.add(embeddings)


vector_store = VectorStore()
