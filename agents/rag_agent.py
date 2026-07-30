"""
RAG agent.

Real retrieval-augmented generation over markdown/text documents in
data/knowledge_base/ - e.g. policy documents, SLAs, definitions - things
that live in prose, not in the master_orders table. Embeddings are built
with a local sentence-transformers model (no external embedding API needed)
and cached to disk so they're only recomputed when a document changes.
"""

import glob
import os

# Defensive: same OpenMP-conflict workaround as app.py, in case this module
# gets imported before app.py's own os.environ.setdefault calls run.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from groq import Groq

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

KB_DIR = os.path.join("data", "knowledge_base")
EMBED_CACHE = os.path.join("data", "kb_embeddings.npz")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _chunk(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


class RAGAgent:
    name = "rag"

    def __init__(self, client: Groq, model: str = "llama-3.3-70b-versatile", top_k: int = 4):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for the RAG agent. "
                "Install it with: pip install sentence-transformers"
            )
        self.client = client
        self.model = model
        self.top_k = top_k
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.chunks, self.sources, self.embeddings = self._load_or_build_index()

    def _load_or_build_index(self):
        os.makedirs(os.path.dirname(EMBED_CACHE), exist_ok=True)
        doc_paths = sorted(
            glob.glob(os.path.join(KB_DIR, "*.md")) + glob.glob(os.path.join(KB_DIR, "*.txt"))
        )
        if not doc_paths:
            return [], [], np.zeros((0, 384), dtype="float32")

        latest_doc_mtime = max(os.path.getmtime(p) for p in doc_paths)
        if os.path.exists(EMBED_CACHE) and os.path.getmtime(EMBED_CACHE) > latest_doc_mtime:
            data = np.load(EMBED_CACHE, allow_pickle=True)
            return list(data["chunks"]), list(data["sources"]), data["embeddings"]

        chunks, sources = [], []
        for path in doc_paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in _chunk(text):
                chunks.append(chunk)
                sources.append(os.path.basename(path))

        embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        np.savez(
            EMBED_CACHE,
            chunks=np.array(chunks, dtype=object),
            sources=np.array(sources, dtype=object),
            embeddings=embeddings,
        )
        return chunks, sources, embeddings

    def retrieve(self, query: str):
        if len(self.chunks) == 0:
            return []
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_emb
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def run(self, instruction: str) -> dict:
        retrieved = self.retrieve(instruction)
        if not retrieved:
            return {
                "agent": self.name,
                "success": False,
                "error": "No knowledge base documents found in data/knowledge_base/.",
            }

        context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
        prompt = (
            "Answer the question using ONLY the context below. If the "
            "context doesn't contain the answer, say so plainly.\n\n"
            f"Context:\n{context}\n\nQuestion: {instruction}\n\nAnswer:"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        answer = response.choices[0].message.content
        return {
            "agent": self.name,
            "success": True,
            "answer": answer,
            "sources": sorted(set(r["source"] for r in retrieved)),
        }
