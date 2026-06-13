# verify_index.py - 阶段四:轻量验证脚本
#
# 不重建索引,只跑质量验证(用已构建的 ./vector_db/)
#
# 跑法:
#   python verify_index.py

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# ==================== 配置 ====================
VECTOR_DB_DIR = "./vector_db"
COLLECTION_NAME = "medical_papers_v4"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNKS_PATH = "./output/chunks.parquet"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
REPORT_PATH = "./output/verify_report.json"
# ===============================================


class BGEEmbedder:
    def __init__(self, model_name: str):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.query_instruction = QUERY_INSTRUCTION

    def embed_query(self, query: str):
        if not query or not query.strip():
            return None
        instructed = self.query_instruction + query
        return self.model.encode([instructed], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]


def query_collection(collection, embedder, query_text, n_results=5, where_filter=None):
    emb = embedder.embed_query(query_text)
    if emb is None:
        return {"ids": [], "distances": [], "documents": [], "metadatas": []}
    result = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=n_results,
        where=where_filter,
    )
    return {
        "ids": result.get("ids", [[]])[0],
        "distances": result.get("distances", [[]])[0],
        "documents": result.get("documents", [[]])[0],
        "metadatas": result.get("metadatas", [[]])[0],
    }


def main():
    print("=" * 70)
    print("阶段四:索引质量验证(轻量版,不重建)")
    print("=" * 70)

    # 1. 加载
    print(f"\n[1/4] 加载索引与模型...")
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = BGEEmbedder(EMBEDDING_MODEL)
    print(f"  合集: {COLLECTION_NAME}")
    print(f"  向量数: {collection.count()}")

    chunks_df = pd.read_parquet(CHUNKS_PATH)
    print(f"  文本块: {len(chunks_df)}")

    report = {}

    # 2. 基础统计
    print(f"\n[2/4] 基础统计...")
    count = collection.count()
    sample = collection.peek(1)
    metadata_keys = list(sample["metadatas"][0].keys()) if sample["metadatas"] else []
    report["basic"] = {
        "vector_count": count,
        "sample_metadata_keys": metadata_keys,
    }
    print(f"  ✅ 向量数: {count}")
    print(f"  ✅ metadata 字段: {metadata_keys}")

    # 3. 元数据过滤验证(修复后版)
    print(f"\n[3/4] 元数据过滤验证...")
    # 3a. $eq 精确匹配 doc_id
    target_doc = "12969509"  # PMC212319 (ARNO 那篇)
    result = query_collection(
        collection, embedder, "What is ARNO?",
        n_results=3,
        where_filter={"doc_id": target_doc}
    )
    print(f"  [3a] $eq doc_id={target_doc}:返回 {len(result['ids'])} 个结果")
    if result["ids"]:
        for i, (id_, m) in enumerate(zip(result["ids"], result["metadatas"]), 1):
            print(f"      {i}. {id_} | {m['source_title'][:60]}")
    report["filter_eq_doc_id"] = {
        "filter": {"doc_id": target_doc},
        "op": "$eq",
        "results_count": len(result["ids"]),
        "sample_ids": result["ids"][:3],
    }

    # 3b. 多个 doc_id 用 $in
    target_docs = ["12969509", "15291972"]
    result = query_collection(
        collection, embedder, "ARF protein",
        n_results=5,
        where_filter={"doc_id": {"$in": target_docs}}
    )
    print(f"\n  [3b] $in doc_id={target_docs}:返回 {len(result['ids'])} 个结果")
    print(f"      实际召回 doc_id: {set(m['doc_id'] for m in result['metadatas'])}")
    report["filter_in_doc_ids"] = {
        "filter": {"doc_id": {"$in": target_docs}},
        "op": "$in",
        "results_count": len(result["ids"]),
        "doc_ids_in_results": list(set(m["doc_id"] for m in result["metadatas"])),
    }

    # 3c. chunk_index 范围 — ChromaDB 限制:每个 dict 只能有一个 operator,多条件用 $and
    result = query_collection(
        collection, embedder, "ARF",
        n_results=5,
        where_filter={
            "$and": [
                {"chunk_index": {"$gte": 0}},
                {"chunk_index": {"$lte": 5}},
            ]
        }
    )
    print(f"\n  [3c] $and chunk_index [0, 5]:返回 {len(result['ids'])} 个结果")
    report["filter_chunk_index_range"] = {
        "filter": {"$and": [{"chunk_index": {"$gte": 0}}, {"chunk_index": {"$lte": 5}}]},
        "op": "$and",
        "results_count": len(result["ids"]),
    }

    # 4. 自相似性 + 边界(快速跑,2 个 sample)
    print(f"\n[4/4] 快速自相似性 + 边界...")
    np.random.seed(42)
    sample_idx = np.random.choice(len(chunks_df), 3, replace=False)
    self_sim = []
    for idx in sample_idx:
        text = chunks_df.iloc[idx]["text"]
        cid = chunks_df.iloc[idx]["chunk_id"]
        r = query_collection(collection, embedder, text, n_results=3)
        top1 = r["ids"][0] if r["ids"] else None
        dist = r["distances"][0] if r["distances"] else None
        hit = (top1 == cid)
        status = "✅" if hit else "⚠️"
        print(f"  {status} {cid} → top-1: {top1} (距离 {dist:.4f})")
        self_sim.append({"query_id": cid, "top1": top1, "is_self": hit})
    report["self_similarity"] = {
        "tested": len(self_sim),
        "passed": sum(1 for s in self_sim if s["is_self"]),
    }

    # 边界
    boundary = {}
    for label, q in [("empty", ""), ("whitespace", "   "), ("single_char", "A"),
                     ("long_4000", "ARF " * 1000)]:
        r = query_collection(collection, embedder, q, n_results=5)
        n = len(r["ids"])
        status = "✅" if (q.strip() == "" and n == 0) or (q.strip() != "" and n > 0) else "⚠️"
        boundary[label] = {"results": n}
        print(f"  {status} {label}({len(q)} 字符):返回 {n} 个结果")
    report["boundary"] = boundary

    # 保存
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✅ 报告: {REPORT_PATH}")
    print("\n" + "=" * 70)
    print("✅ 验证完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
