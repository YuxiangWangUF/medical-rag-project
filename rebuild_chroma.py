"""Rebuild ChromaDB with bge-m3 embeddings - re-parse XML and rebuild"""
import os, sys, time

sys.path.insert(0, r'C:\Users\Administrator\Desktop\medical_rag_project')

DATA_PATH = r"C:\Users\Administrator\Desktop\medical_rag_project\data\medical_papers"
PERSIST_DIR = r"C:\Users\Administrator\Desktop\medical_rag_project\chroma_db"
COLLECTION_NAME = "medical_papers_v2_bgem3"
MAX_FILES = None  # None = 全量 3028 篇

import rag_medical
rag_medical.MAX_FILES = MAX_FILES

from rag_medical import load_documents, split_documents, build_vectorstore
from ollama_batch_embeddings import OllamaEmbeddingsBatch

embeddings = OllamaEmbeddingsBatch(model="bge-m3")

print("=" * 60)
print("ChromaDB rebuild with bge-m3 embeddings")
print("=" * 60)

# 1. Load docs
print("\n[1/4] Loading XML files...")
docs = load_documents(DATA_PATH)
if not docs:
    print("ERROR: No documents loaded")
    sys.exit(1)
print(f"  Loaded {len(docs)} documents")

# 2. Chunk
print("\n[2/4] Chunking documents...")
chunks = split_documents(docs)
print(f"  Got {len(chunks)} chunks")

# 3. bge-m3 via Ollama
print("\n[3/4] Setting up bge-m3 embeddings via Ollama...")
embeddings = OllamaEmbeddingsBatch(model="bge-m3")
_ = embeddings.embed_query("warmup text")  # first call ~4s warmup
print("  bge-m3 warmup done")

# Override collection name so build_vectorstore creates new one
rag_medical.COLLECTION_NAME = COLLECTION_NAME

# 4. Build vector store
print(f"\n[4/4] Building new ChromaDB ({COLLECTION_NAME})...")
t0 = time.time()
vs = build_vectorstore(chunks, PERSIST_DIR, embeddings)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s ({elapsed/len(chunks)*1000:.1f}ms per chunk)")

# Sanity check
print("\n[Verify] Testing similarity search...")
test_query = "metformin cardiovascular effects"
results = vs.similarity_search(test_query, k=3)
print(f"  Query: {test_query}")
for i, r in enumerate(results, 1):
    src = r.metadata.get("source", "?")
    print(f"  {i}. [{src}] {r.page_content[:80]}...")

print(f"\nRebuild complete! New collection: {COLLECTION_NAME}")
print("NOTE: stage6_retrieval_pipeline.py and multi_path_retriever.py already point to this collection.")