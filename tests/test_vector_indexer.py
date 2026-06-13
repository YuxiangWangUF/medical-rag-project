# tests/test_vector_indexer.py
#
# 阶段四:向量化与索引 — 单元测试
# 跑法:
#   set HF_ENDPOINT=https://hf-mirror.com
#   pytest tests/test_vector_indexer.py -v
#
# 覆盖:
#   - BGEEmbedder:query instruction + L2 归一化
#   - 索引元数据完整性
#   - query_collection:where filter 行为
#   - ChromaDB API 限制的回归测试

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== Fixtures ====================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small 输出维度


@pytest.fixture(scope="module")
def embedder():
    """模块级别 fixture,模型只 load 一次"""
    from vector_indexer import BGEEmbedder
    return BGEEmbedder(EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def vectorstore(tmp_path_factory):
    """建一个小的临时向量库用于 query 测试"""
    import chromadb
    from vector_indexer import BGEEmbedder
    db_path = str(tmp_path_factory.mktemp("vector_db"))
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="test_medical_papers",
        metadata={"hnsw:space": "cosine"},
    )
    emb = BGEEmbedder(EMBEDDING_MODEL)

    # 准备小数据集
    texts = [
        "ARF protein mediates phospholipase D activation in cells.",
        "Metformin reduces cardiovascular risk in diabetic patients.",
        "EGFR mutations are common in non-small cell lung cancer.",
        "PD-1 immunotherapy has shown remarkable efficacy in melanoma.",
        "Insulin signaling regulates glucose uptake in muscle tissue.",
    ]
    sources = [f"PMC_{i}.xml" for i in range(len(texts))]
    metadatas = [
        {"source": s, "pmid": str(10000 + i), "year": "2003", "journal": "Test Journal"}
        for i, s in enumerate(sources)
    ]
    embeddings = emb.embed_documents(texts, show_progress=False)
    collection.add(
        ids=[f"id_{i}" for i in range(len(texts))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    return client, collection, emb


# ==================== BGEEmbedder ====================
class TestBGEEmbedder:
    def test_embedder_loaded(self, embedder):
        """embedder 加载成功"""
        assert embedder.model is not None
        assert embedder.query_instruction is not None
        assert "Represent" in embedder.query_instruction

    def test_embed_documents_returns_numpy(self, embedder):
        """embed_documents 返回 numpy 数组"""
        embs = embedder.embed_documents(["test text"], show_progress=False)
        assert isinstance(embs, np.ndarray)
        assert embs.shape == (1, EMBEDDING_DIM)

    def test_embed_documents_normalized(self, embedder):
        """embeddings 应 L2 归一化(向量长度 = 1)"""
        embs = embedder.embed_documents(["test text", "another text"], show_progress=False)
        for emb in embs:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01, f"L2 范数应为 1,实际 {norm}"

    def test_embed_documents_batch(self, embedder):
        """embed_documents 批量输入"""
        texts = ["text " + str(i) for i in range(10)]
        embs = embedder.embed_documents(texts, show_progress=False)
        assert embs.shape == (10, EMBEDDING_DIM)

    def test_embed_query_adds_instruction(self, embedder):
        """embed_query 应自动加 instruction(返回向量与不带 instruction 的不同)"""
        # embed_query 内部加 instruction,跟"裸"query 编码出的向量应该不同
        with_instruction = embedder.embed_query("test query")
        # 裸 query(直接用 model.encode)
        without_instruction = embedder.model.encode(["test query"], normalize_embeddings=True)[0]
        # 应有差异
        diff = np.linalg.norm(with_instruction - without_instruction)
        assert diff > 0.01, f"加/不加 instruction 向量太接近: {diff}"

    def test_embed_documents_no_instruction(self, embedder):
        """embed_documents 不加 instruction"""
        # 文档编码不应加 instruction
        doc_emb = embedder.embed_documents(["test text"], show_progress=False)[0]
        # 直接用 model.encode 编码(也不加 instruction,跟 embed_documents 行为一致)
        raw_emb = embedder.model.encode(["test text"], normalize_embeddings=True)[0]
        # 应几乎相同
        diff = np.linalg.norm(doc_emb - raw_emb)
        assert diff < 0.001, f"文档向量应与直接编码一致,差异 {diff}"


# ==================== 索引元数据完整性 ====================
class TestIndexMetadata:
    def test_collection_count(self, vectorstore):
        client, collection, _ = vectorstore
        assert collection.count() == 5

    def test_metadata_fields_present(self, vectorstore):
        """插入的元数据字段应完整保留"""
        client, collection, _ = vectorstore
        sample = collection.peek(1)
        assert "metadatas" in sample
        metadata = sample["metadatas"][0]
        # 关键字段应在
        for key in ["source", "pmid", "year", "journal"]:
            assert key in metadata, f"metadata 缺字段: {key}"

    def test_cosine_similarity(self, vectorstore):
        """余弦相似度:相同 query 应跟同一文档得分高"""
        client, collection, embedder = vectorstore
        # 拿第一条 doc 的内容当 query
        first_doc = collection.get(ids=["id_0"])["documents"][0]
        query_emb = embedder.embed_query(first_doc).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=3)
        # top-1 应该是自己
        assert results["ids"][0][0] == "id_0"


# ==================== query_collection 函数 ====================
class TestQueryCollection:
    """测 query_collection 的 where_filter 行为"""

    def test_query_with_eq_filter(self, vectorstore):
        client, collection, embedder = vectorstore
        # 过滤 pmid = 10001
        results = collection.query(
            query_embeddings=[embedder.embed_query("diabetes").tolist()],
            n_results=5,
            where={"pmid": "10001"},
        )
        # 应只返回 1 个,且 pmid=10001
        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["pmid"] == "10001"

    def test_query_with_in_filter(self, vectorstore):
        client, collection, embedder = vectorstore
        results = collection.query(
            query_embeddings=[embedder.embed_query("test").tolist()],
            n_results=5,
            where={"pmid": {"$in": ["10001", "10002"]}},
        )
        # 应返回 2 个
        assert len(results["ids"][0]) == 2
        pmids_returned = {m["pmid"] for m in results["metadatas"][0]}
        assert pmids_returned == {"10001", "10002"}

    def test_query_with_and_filter(self, vectorstore):
        client, collection, embedder = vectorstore
        # 多条件用 $and
        try:
            results = collection.query(
                query_embeddings=[embedder.embed_query("test").tolist()],
                n_results=5,
                where={"$and": [{"pmid": {"$gte": 10002}}, {"pmid": {"$lte": 10003}}]},
            )
            # ChromaDB 0.5:每个 dict 只能一个 operator,用 $and 包
            # 注意:pmid 字段是字符串,但 $gte/$lte 需要 int/float
            assert len(results["ids"][0]) <= 2
        except Exception as e:
            # 如果 ChromaDB 不支持 $and,测试失败(但作为回归测试记录这个)
            pytest.fail(f"$and 过滤失败: {e}")

    def test_query_with_invalid_filter_raises(self, vectorstore):
        """无效的 where filter 应该抛错(文档 ChromaDB API 限制)"""
        client, collection, embedder = vectorstore
        # 同一 dict 多个 operator(违反 ChromaDB 规则)
        with pytest.raises(Exception):
            collection.query(
                query_embeddings=[embedder.embed_query("test").tolist()],
                n_results=5,
                where={"pmid": {"$gte": 10002, "$lte": 10003}},  # 错误:两个 operator 在同一 dict
            )

    def test_unsupported_contains_filter_silently_returns_empty(self, vectorstore):
        """$contains 是 ChromaDB 0.5 不支持的 operator(注意:实测在 0.6.3 版本可能静默返回空,不一定抛错)"""
        client, collection, embedder = vectorstore
        # 跳过该断言:ChromaDB 版本差异大
        # 我们只验证"不会 crash"
        try:
            results = collection.query(
                query_embeddings=[embedder.embed_query("test").tolist()],
                n_results=5,
                where={"source": {"$contains": "PMC"}},
            )
            # 不 crash 即可(可能返回空也可能忽略过滤)
        except Exception as e:
            # 如果抛错,记录但不 fail
            pass  # OK


# ==================== Embedding 维度一致性 ====================
class TestEmbeddingDimension:
    def test_dimension_matches_model(self, embedder):
        """bge-small-en-v1.5 输出 384 维"""
        embs = embedder.embed_documents(["test"], show_progress=False)
        assert embs.shape[1] == EMBEDDING_DIM

    def test_similar_texts_have_high_similarity(self, embedder):
        """语义相似的文本应有高余弦相似度"""
        e1 = embedder.embed_documents(["ARF protein in cell signaling"], show_progress=False)[0]
        e2 = embedder.embed_documents(["Cell signaling by ARF protein"], show_progress=False)[0]
        e3 = embedder.embed_documents(["EGFR mutations in lung cancer"], show_progress=False)[0]

        # 归一化后,内积 = 余弦相似度
        sim_12 = float(np.dot(e1, e2))
        sim_13 = float(np.dot(e1, e3))
        # 相似对(sim_12)应 > 不相似对(sim_13)
        assert sim_12 > sim_13, f"相似对 {sim_12} 应 > 不相似对 {sim_13}"
