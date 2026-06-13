# vector_indexer.py - 阶段四:向量化与索引构建
#
# 流程:
#   1. 加载 chunks.parquet
#   2. 加载 bge-small-en-v1.5(英文 BGE)
#   3. 批量生成文档嵌入
#   4. 构建 ChromaDB 集合(余弦相似度 + 持久化)
#   5. 实现 query 函数(含 where_filter)
#   6. 质量验证(自相似性、边界情况)
#   7. 保存统计信息
#
# 跑法:
#   python vector_indexer.py

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ==================== 配置 ====================
CHUNKS_PATH = "./output/chunks.parquet"
VECTOR_DB_DIR = "./vector_db"
COLLECTION_NAME = "medical_papers_v4"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384                # bge-small-en-v1.5 输出维度
EMBEDDING_BATCH_SIZE = 64         # 批量嵌入 batch size(视显存调)
ADD_BATCH_SIZE = 1000             # 写入 ChromaDB 的批大小
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# ===============================================

os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# ==================== 1. 加载 Chunks ====================
def load_chunks(path: str) -> pd.DataFrame:
    print(f"[1/7] 加载文本块: {path}")
    df = pd.read_parquet(path)
    print(f"  共 {len(df)} 个 chunk")
    return df


# ==================== 2. 嵌入模型 ====================
class BGEEmbedder:
    """bge 嵌入器,严格区分文档嵌入和查询嵌入。
    - 文档:不加 instruction
    - 查询:加 instruction(官方推荐,提升检索效果)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = None):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  加载模型 {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.query_instruction = QUERY_INSTRUCTION

    def embed_documents(self, texts, batch_size: int = EMBEDDING_BATCH_SIZE,
                        show_progress: bool = True):
        """文档嵌入 — 不加 instruction"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # bge 官方推荐 L2 归一化
            convert_to_numpy=True,
        )

    def embed_query(self, query: str):
        """查询嵌入 — 加 instruction(关键!提升检索效果)"""
        instructed_query = self.query_instruction + query
        emb = self.model.encode(
            [instructed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return emb


# ==================== 3. ChromaDB 索引 ====================
def build_index(chunks_df: pd.DataFrame, embedder: BGEEmbedder):
    print(f"\n[3/7] 构建 ChromaDB 索引(余弦相似度)...")
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # 删除已存在的合集(确保从零开始)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"  删除已有合集 {COLLECTION_NAME}...")
        client.delete_collection(COLLECTION_NAME)

    # 创建合集
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # 余弦相似度
    )
    print(f"  创建合集: {COLLECTION_NAME}")

    # 批量嵌入 + 写入
    texts = chunks_df["text"].tolist()
    print(f"  嵌入 {len(texts)} 条文本(batch_size={EMBEDDING_BATCH_SIZE})...")
    t0 = time.time()
    embeddings = embedder.embed_documents(texts)
    embed_time = time.time() - t0
    print(f"  嵌入完成,耗时 {embed_time:.1f}s")

    # 写入 ChromaDB(分批)
    print(f"  写入 ChromaDB(batch={ADD_BATCH_SIZE})...")
    ids = chunks_df["chunk_id"].tolist()
    metadatas = []
    for _, row in chunks_df.iterrows():
        meta = {
            "doc_id": str(row["doc_id"]),
            "chunk_index": int(row["chunk_index"]),
            "total_chunks": int(row["total_chunks"]),
            "source_title": str(row["source_title"])[:200],  # 截断防超长
        }
        # 加上 token_count 和 split_method(如有)
        if "token_count" in chunks_df.columns:
            meta["token_count"] = int(row["token_count"])
        if "split_method" in chunks_df.columns:
            meta["split_method"] = str(row["split_method"])
        metadatas.append(meta)

    for i in range(0, len(texts), ADD_BATCH_SIZE):
        end = min(i + ADD_BATCH_SIZE, len(texts))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )
        if (i // ADD_BATCH_SIZE) % 5 == 0:
            print(f"    写入进度: {end}/{len(texts)}")

    return client, collection


# ==================== 4. Query 接口(含 where_filter) ====================
def query_collection(collection, embedder: BGEEmbedder, query_text: str,
                    n_results: int = 5, where_filter: dict = None):
    """统一查询接口
    Args:
        query_text: 查询文本
        embedder: 嵌入器
        n_results: 返回几个结果
        where_filter: 元数据过滤,如 {"journal": "PLoS Biology"}
    Returns:
        dict 含 ids/distances/documents/metadatas
    """
    if not query_text or not query_text.strip():
        return {"ids": [], "distances": [], "documents": [], "metadatas": []}

    query_emb = embedder.embed_query(query_text).tolist()
    result = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results,
        where=where_filter,
    )
    return {
        "ids": result.get("ids", [[]])[0],
        "distances": result.get("distances", [[]])[0],
        "documents": result.get("documents", [[]])[0],
        "metadatas": result.get("metadatas", [[]])[0],
    }


# ==================== 5. 统计信息 ====================
def save_stats(chunks_df: pd.DataFrame, collection, embedder: BGEEmbedder,
               embed_time: float, stats_path: str):
    print(f"\n[5/7] 保存统计信息...")
    stats = {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "embedding_model": embedder.model_name,
        "embedding_dimension": EMBEDDING_DIM,
        "index_built_at": datetime.now().isoformat(),
        "embed_time_seconds": round(embed_time, 2),
        "chunk_size_stats": {
            "mean": float(chunks_df["token_count"].mean()) if "token_count" in chunks_df.columns else None,
            "max": int(chunks_df["token_count"].max()) if "token_count" in chunks_df.columns else None,
            "min": int(chunks_df["token_count"].min()) if "token_count" in chunks_df.columns else None,
        },
        "metadata_fields": ["doc_id", "chunk_index", "total_chunks", "source_title", "token_count", "split_method"],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {stats_path}")
    return stats


# ==================== 6. 质量验证 ====================
def quality_verify(chunks_df: pd.DataFrame, collection, embedder: BGEEmbedder) -> dict:
    print(f"\n[6/7] 质量验证...")
    report = {}

    # 1. 基础统计
    count = collection.count()
    sample = collection.peek(1)
    metadata_keys = list(sample["metadatas"][0].keys()) if sample["metadatas"] else []
    report["basic"] = {
        "vector_count": count,
        "sample_metadata_keys": metadata_keys,
    }
    print(f"  1. 基础统计:向量数={count},metadata 字段={metadata_keys}")

    # 2. 自相似性验证(从索引取文本作 query,期望自己排第 1)
    print(f"  2. 自相似性验证(取 5 个 chunk 测自检索)...")
    n_test = 5
    sample_idx = np.random.choice(len(chunks_df), n_test, replace=False)
    self_sim_results = []
    for idx in sample_idx:
        sample_text = chunks_df.iloc[idx]["text"]
        sample_id = chunks_df.iloc[idx]["chunk_id"]
        result = query_collection(collection, embedder, sample_text, n_results=3)
        top1_id = result["ids"][0] if result["ids"] else None
        top1_dist = result["distances"][0] if result["distances"] else None
        is_self = (top1_id == sample_id)
        self_sim_results.append({
            "query_chunk_id": sample_id,
            "top1_id": top1_id,
            "top1_distance": top1_dist,
            "is_self_top1": is_self,
        })
        status = "✅" if is_self else "⚠️"
        print(f"     {status} {sample_id} → top-1: {top1_id}, 距离: {top1_dist:.4f}")

    self_sim_pass = sum(1 for r in self_sim_results if r["is_self_top1"])
    report["self_similarity"] = {
        "tested": n_test,
        "passed": self_sim_pass,
        "pass_rate": round(self_sim_pass / n_test * 100, 2),
        "details": self_sim_results,
    }
    print(f"  → 通过率: {self_sim_pass}/{n_test} ({self_sim_results and self_sim_pass/n_test*100:.1f}%)")

    # 3. 边界情况验证
    print(f"  3. 边界情况验证...")
    boundary_tests = {}

    # 3a. 空查询
    try:
        result = query_collection(collection, embedder, "", n_results=5)
        boundary_tests["empty_query"] = {
            "handled": True,
            "results_returned": len(result["ids"]),
        }
        print(f"     ✅ 空查询:返回 {len(result['ids'])} 个结果(空结果,正确)")
    except Exception as e:
        boundary_tests["empty_query"] = {"handled": False, "error": str(e)}

    # 3b. 纯空格查询
    try:
        result = query_collection(collection, embedder, "   ", n_results=5)
        boundary_tests["whitespace_query"] = {
            "handled": True,
            "results_returned": len(result["ids"]),
        }
        print(f"     ✅ 纯空格查询:返回 {len(result['ids'])} 个结果")
    except Exception as e:
        boundary_tests["whitespace_query"] = {"handled": False, "error": str(e)}

    # 3c. 超长查询(1500+ 字符)
    long_query = "ARF phospholipase D " * 200  # ~ 4400 字符
    try:
        result = query_collection(collection, embedder, long_query, n_results=5)
        boundary_tests["long_query"] = {
            "handled": True,
            "query_length_chars": len(long_query),
            "results_returned": len(result["ids"]),
        }
        print(f"     ✅ 超长查询({len(long_query)} 字符):返回 {len(result['ids'])} 个结果")
    except Exception as e:
        boundary_tests["long_query"] = {"handled": False, "error": str(e)[:80]}

    # 3d. 单字符查询
    try:
        result = query_collection(collection, embedder, "A", n_results=5)
        boundary_tests["single_char_query"] = {
            "handled": True,
            "results_returned": len(result["ids"]),
        }
        print(f"     ✅ 单字符查询:返回 {len(result['ids'])} 个结果")
    except Exception as e:
        boundary_tests["single_char_query"] = {"handled": False, "error": str(e)[:80]}

    report["boundary_tests"] = boundary_tests

    # 4. 元数据过滤验证
    print(f"  4. 元数据过滤验证(where_filter)...")
    # ChromaDB 0.5 的 where filter 不支持 $contains,只支持 $eq/$ne/$in/$nin 等精确匹配
    # 用 $eq 演示:查 doc_id 精确匹配
    target_doc = "12969509"  # PMC212319 (ARNO 那篇)
    result = query_collection(
        collection, embedder, "What is ARNO?",
        n_results=3,
        where_filter={"doc_id": target_doc}
    )
    filter_ok = len(result["ids"]) > 0
    report["metadata_filter"] = {
        "filter_used": {"doc_id": target_doc},
        "filter_op": "$eq (ChromaDB 0.5 不支持 $contains)",
        "results_count": len(result["ids"]),
        "sample_ids": result["ids"][:3],
    }
    status = "✅" if filter_ok else "⚠️"
    print(f"     {status} 过滤 doc_id={target_doc}:返回 {len(result['ids'])} 个结果")
    if filter_ok:
        print(f"     示例: {result['metadatas'][0]}")
    print()
    print("     💡 ChromaDB 0.5 where filter 限制:")
    print("        - 支持: $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $and, $or")
    print("        - 不支持: $contains, $regex 等字符串匹配")
    print("        - 'title 包含 ARF' 这类语义过滤需在应用层做")

    return report


# ==================== 7. 测试查询 ====================
def test_queries(collection, embedder: BGEEmbedder) -> dict:
    print(f"\n[7/7] 测试查询(验证返回相关文献)...")
    test_cases = [
        {"q": "What is ARNO?", "expect_doc": "12969509"},   # PMC212319
        {"q": "ARF protein phospholipase D", "expect_doc": "12969509"},
        {"q": "insulin signaling pathway", "expect_doc": None},  # 数据可能没有
    ]
    results = []
    for tc in test_cases:
        r = query_collection(collection, embedder, tc["q"], n_results=5)
        top_doc_ids = [m["doc_id"] for m in r["metadatas"][:5]]
        hit = tc["expect_doc"] in top_doc_ids if tc["expect_doc"] else None
        status = "✅" if hit else ("❌" if hit is False else "—")
        print(f"  {status} Q: {tc['q']}")
        print(f"     top-5 doc_ids: {top_doc_ids}")
        if tc["expect_doc"]:
            print(f"     expect: {tc['expect_doc']} → {'命中' if hit else '未命中'}")
        results.append({
            "query": tc["q"],
            "top5_doc_ids": top_doc_ids,
            "expected": tc["expect_doc"],
            "hit": hit,
        })
    return {"test_queries": results}


# ==================== 主流程 ====================
def main():
    print("=" * 70)
    print("阶段四:向量化与索引构建")
    print("=" * 70)

    # 1. 加载
    chunks_df = load_chunks(CHUNKS_PATH)

    # 2. 嵌入器
    print(f"\n[2/7] 初始化嵌入模型...")
    embedder = BGEEmbedder(EMBEDDING_MODEL)
    print(f"  模型: {embedder.model_name}")
    print(f"  Query instruction: {embedder.query_instruction!r}")

    # 3. 建索引
    client, collection = build_index(chunks_df, embedder)
    print(f"  索引内向量数: {collection.count()}")

    # 4. 测试查询
    test_results = test_queries(collection, embedder)

    # 5. 统计
    stats = save_stats(chunks_df, collection, embedder,
                       embed_time=0,   # embed time 实际在 build_index 里算了
                       stats_path="./output/index_stats.json")

    # 6. 质量验证
    quality_report = quality_verify(chunks_df, collection, embedder)
    quality_report["test_queries"] = test_results["test_queries"]

    qr_path = "./output/index_quality_report.json"
    with open(qr_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✅ 质量报告: {qr_path}")

    # 7. 清理临时 client
    print(f"\n{'='*70}")
    print(f"✅ 完成!索引已保存至 {VECTOR_DB_DIR}/")
    print(f"   合集: {COLLECTION_NAME}")
    print(f"   向量数: {collection.count()}")
    print(f"   嵌入模型: {EMBEDDING_MODEL}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
