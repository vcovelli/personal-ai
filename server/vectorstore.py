from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast

class VectorStore:
    def __init__(self):
        self.texts = []
        self.index = faiss.IndexFlatL2(384)  # 384 = embedding size for MiniLM

    def add_texts(self, docs: list[str]):
        self.texts.extend(docs)
        embeddings = model.encode(docs)
        self.index.add(np.array(embeddings).astype("float32"))

    def query(self, query: str, k: int = 3):
        if not self.texts:
            return ["⚠️ No documents available."]
        D, I = self.index.search(self.embed(query), k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

vector_store = VectorStore()
