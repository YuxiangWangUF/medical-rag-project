"""Sequential batch embedder for Ollama - wraps OllamaEmbeddings so Chroma can use it"""
import requests
from typing import List
from langchain_ollama import OllamaEmbeddings

class OllamaEmbeddingsBatch(OllamaEmbeddings):
    """Wrapper that makes OllamaEmbeddings.embed_documents work
    by calling /api/embeddings sequentially (Ollama has no batch endpoint).
    First call warms up (~4s), subsequent ~5ms each."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results = []
        for text in texts:
            r = requests.post(
                f"{self.base_url or 'http://localhost:11434'}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60
            )
            results.append(r.json()["embedding"])
        return results

    def embed_query(self, text: str) -> List[float]:
        r = requests.post(
            f"{self.base_url or 'http://localhost:11434'}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60
        )
        return r.json()["embedding"]