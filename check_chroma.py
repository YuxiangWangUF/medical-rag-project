"""Check existing ChromaDB contents"""
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb

client = chromadb.PersistentClient(path=r'C:\Users\Administrator\Desktop\medical_rag_project\chroma_db')
col = client.get_collection('medical_papers_v2')
all_data = col.get(include=['documents', 'metadatas'])
print(f'Total chunks: {len(all_data["documents"])}')
print(f'Sample doc: {all_data["documents"][0][:100]}')
print(f'Sample meta: {all_data["metadatas"][0]}')

# Check unique sources
sources = set(m.get('source', '') for m in all_data['metadatas'])
print(f'Unique source files: {len(sources)}')