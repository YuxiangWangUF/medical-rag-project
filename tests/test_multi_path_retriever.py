"""
阶段六融合层测试:multi_path_retriever.py 的融合函数 + _doc_key

覆盖:
- _doc_key:优先级 pmid > source > doc_id > id
- rrf_fusion:同 pmid 跨路径合并(关键 bug 修复)
- weighted_fusion:同 pmid 合并 + 单路命中不稀释
- simple_fusion:同 pmid 去重

跑法:
    $env:HF_ENDPOINT="https://hf-mirror.com"
    D:\Anaconda\envs\medical_rag\python.exe -m pytest tests/test_multi_path_retriever.py -v
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multi_path_retriever import (
    _doc_key,
    rrf_fusion,
    weighted_fusion,
    simple_fusion,
    BM25Index,
    MultiPathRetriever,
)


# ==================== _doc_key 测试 ====================
class TestDocKey:
    """_doc_key 提取稳定的 doc 标识"""

    def test_01_doc_key_by_pmid(self):
        """有 pmid → 用 pmid"""
        d = Document(page_content="x", metadata={"pmid": "12345", "source": "PMC.xml"})
        assert _doc_key(d) == "pmid:12345"

    def test_02_doc_key_by_source(self):
        """没 pmid 但有 source → 用 source"""
        d = Document(page_content="x", metadata={"source": "PMC123.xml"})
        assert _doc_key(d) == "pmid:PMC123.xml"

    def test_03_doc_key_by_doc_id(self):
        """没 pmid/source 但有 doc_id → 用 doc_id"""
        d = Document(page_content="x", metadata={"doc_id": "d1"})
        assert _doc_key(d) == "pmid:d1"

    def test_04_doc_key_fallback_id(self):
        """啥都没有 → id(doc) 兜底"""
        d = Document(page_content="x", metadata={})
        assert _doc_key(d) == f"id:{id(d)}"

    def test_05_doc_key_from_tuple(self):
        """传 (doc, score) tuple 也能拿 key"""
        d = Document(page_content="x", metadata={"pmid": "999"})
        assert _doc_key((d, 0.5)) == "pmid:999"

    def test_06_doc_key_consistency(self):
        """同 pmid 不同 doc object → 同样的 key(这是修复的核心)"""
        d1 = Document(page_content="A", metadata={"pmid": "100"})
        d2 = Document(page_content="B", metadata={"pmid": "100"})
        assert _doc_key(d1) == _doc_key(d2)
        # 旧版用 id() 这里会失败;新版用 pmid 一致

    def test_07_doc_key_pmid_wins_over_source(self):
        """pmid 优先级 > source"""
        d = Document(page_content="x", metadata={"pmid": "P1", "source": "S1"})
        assert _doc_key(d) == "pmid:P1"


# ==================== rrf_fusion 测试 ====================
class TestRRFFusion:
    """RRF 融合,关键是修复后能合并同 pmid"""

    def test_08_rrf_basic(self):
        """基本 RRF:两路各 2 条,4 条融合结果"""
        path1 = [
            (Document(page_content="a", metadata={"pmid": "1"}), 1.0),
            (Document(page_content="b", metadata={"pmid": "2"}), 0.5),
        ]
        path2 = [
            (Document(page_content="c", metadata={"pmid": "3"}), 0.8),
            (Document(page_content="d", metadata={"pmid": "4"}), 0.3),
        ]
        fused = rrf_fusion([path1, path2])
        assert len(fused) == 4

    def test_09_rrf_merges_same_pmid_across_paths(self):
        """
        **关键 bug 修复**:
        同一 pmid 在 BM25 路径和向量路径都出现 → 应该合并为 1 条,分数累加
        旧版用 id(doc) 会变成 2 条(分数都不高)
        """
        # 模拟:BM25 拿到 doc1(p=X),向量也拿到 doc2(同 p=X)
        doc_bm25 = Document(page_content="text A", metadata={"pmid": "X"})
        doc_vec = Document(page_content="text B", metadata={"pmid": "X"})
        path_bm25 = [(doc_bm25, 5.0), (Document(page_content="y", metadata={"pmid": "Y"}), 3.0)]
        path_vec = [(doc_vec, 0.9), (Document(page_content="z", metadata={"pmid": "Z"}), 0.5)]

        fused = rrf_fusion([path_bm25, path_vec])
        # X 应该只出现 1 次(被合并),分数是两路累加
        x_entries = [d for d, _ in fused if d.metadata.get("pmid") == "X"]
        assert len(x_entries) == 1, f"同 pmid 没被合并!结果: {[(d.metadata, s) for d, s in fused]}"

    def test_10_rrf_score_higher_for_dual_hit(self):
        """双路命中的 doc 分数 > 单路命中"""
        # X 双路都中,Y 只在 path1 中
        path1 = [
            (Document(page_content="x1", metadata={"pmid": "X"}), 1.0),
            (Document(page_content="y1", metadata={"pmid": "Y"}), 0.5),
        ]
        path2 = [
            (Document(page_content="x2", metadata={"pmid": "X"}), 0.9),
        ]
        fused = rrf_fusion([path1, path2])
        fused_pmid = [(d.metadata["pmid"], s) for d, s in fused]
        # X 排第 1(双路,1/(1+60) + 1/(1+60) = 2/61)
        # Y 排第 2(单路,1/(2+60) = 1/62)
        assert fused_pmid[0][0] == "X"
        assert fused_pmid[0][1] > fused_pmid[1][1]

    def test_11_rrf_empty_input(self):
        """空输入"""
        assert rrf_fusion([]) == []
        assert rrf_fusion([[]]) == []
        assert rrf_fusion([[], []]) == []

    def test_12_rrf_k_parameter(self):
        """k 越小,排名越敏感"""
        path = [(Document(page_content=str(i), metadata={"pmid": str(i)}), float(10 - i))
                for i in range(5)]
        fused_small_k = rrf_fusion([path], k=1)
        fused_large_k = rrf_fusion([path], k=1000)
        # k=1 时 rank 1 的分数 = 1/2, k=1000 时 = 1/1001
        assert fused_small_k[0][1] > fused_large_k[0][1]


# ==================== weighted_fusion 测试 ====================
class TestWeightedFusion:
    """加权融合:修复单路命中不被稀释"""

    def test_13_weighted_merges_same_pmid(self):
        """同 pmid 跨路径合并"""
        path1 = [(Document(page_content="x1", metadata={"pmid": "X"}), 1.0)]
        path2 = [(Document(page_content="x2", metadata={"pmid": "X"}), 0.5)]
        fused = weighted_fusion([path1, path2], weights=[0.5, 0.5])
        x_count = sum(1 for d, _ in fused if d.metadata.get("pmid") == "X")
        assert x_count == 1

    def test_14_weighted_no_dilution_single_path(self):
        """
        **P0-3 修复验证**:
        单路命中的 doc,分数应等于该路归一化分数(不被另一路 0 拉低)
        """
        # Y 只在 path1 出现,path2 没有任何 doc
        path1 = [(Document(page_content="y1", metadata={"pmid": "Y"}), 1.0)]
        path2 = []
        fused = weighted_fusion([path1, path2], weights=[0.6, 0.4])
        # Y 的分数 = 0.6 * 1.0(归一化后) = 0.6
        # 如果按"总分/总路径"会变成 0.3
        y_score = next(s for d, s in fused if d.metadata.get("pmid") == "Y")
        assert abs(y_score - 0.6) < 0.01, f"单路分数被稀释: {y_score}"

    def test_15_weighted_empty(self):
        """空输入"""
        assert weighted_fusion([]) == []
        assert weighted_fusion([[]]) == []

    def test_16_weighted_weights_normalized(self):
        """权重自动归一化"""
        path1 = [(Document(page_content="x", metadata={"pmid": "X"}), 1.0)]
        path2 = [(Document(page_content="y", metadata={"pmid": "Y"}), 1.0)]
        # 传 weights=[2, 2] → 归一化为 [0.5, 0.5]
        fused = weighted_fusion([path1, path2], weights=[2, 2])
        # 两条 doc 分数应该都是 0.5(归一化后)
        assert all(abs(s - 0.5) < 0.01 for _, s in fused)

    def test_17_weighted_default_equal_weights(self):
        """不传 weights → 均分"""
        path1 = [(Document(page_content="x", metadata={"pmid": "X"}), 1.0)]
        path2 = [(Document(page_content="y", metadata={"pmid": "Y"}), 1.0)]
        fused = weighted_fusion([path1, path2])
        # 两条分数都 = 0.5
        assert all(abs(s - 0.5) < 0.01 for _, s in fused)


# ==================== simple_fusion 测试 ====================
class TestSimpleFusion:
    """简单合并去重"""

    def test_18_simple_dedup_same_pmid(self):
        """同 pmid 去重"""
        path1 = [(Document(page_content="x1", metadata={"pmid": "X"}), 1.0)]
        path2 = [(Document(page_content="x2", metadata={"pmid": "X"}), 0.5)]
        fused = simple_fusion([path1, path2])
        x_count = sum(1 for d, _ in fused if d.metadata.get("pmid") == "X")
        assert x_count == 1

    def test_19_simple_keeps_max_score(self):
        """同 pmid 保留先遇到的那条(不一定是最高分)"""
        path1 = [(Document(page_content="x1", metadata={"pmid": "X"}), 0.5)]
        path2 = [(Document(page_content="x2", metadata={"pmid": "X"}), 0.9)]
        fused = simple_fusion([path1, path2])
        # 第一条进(0.5),第二条跳过
        x_score = next(s for d, s in fused if d.metadata.get("pmid") == "X")
        assert x_score == 0.5

    def test_20_simple_empty(self):
        """空输入"""
        assert simple_fusion([]) == []


# ==================== MultiPathRetriever.vector_query 测试 ====================
class TestMultiPathRetrieverVectorQuery:
    """**关键测试**:BM25 拿 keyword_query,向量检索拿带 BGE instruction 的 vector_query"""

    def test_21_retrieve_passes_vector_query_separately(self, monkeypatch):
        """retrieve 时 BM25 拿 query,向量拿 vector_query"""
        from langchain_chroma import Chroma
        from multi_path_retriever import MultiPathRetriever, _doc_key

        # mock BM25.search 记录传入的 query
        bm25_called_with = []
        class FakeBM25:
            def search(self, q, **kwargs):
                bm25_called_with.append(q)
                return [(Document(page_content="bm25 doc", metadata={"pmid": "BM25"}), 1.0)]
        # mock vectorstore.similarity_search_with_score 记录传入的 query
        vec_called_with = []
        class FakeVectorstore:
            def similarity_search_with_score(self, q, **kwargs):
                vec_called_with.append(q)
                return [(Document(page_content="vec doc", metadata={"pmid": "VEC"}), 0.1)]

        class FakeReranker:
            def predict(self, pairs, **kwargs):
                return [1.0] * len(pairs)
            def _get_recency_score(self, m): return 0.5
            def _get_authority_score(self, m): return 0.5
            criteria_weights = {"relevance": 0.6, "recency": 0.25, "authority": 0.15}
            def rerank(self, query, candidates, top_n=5):
                # mock 简单返回前 N
                return [(d, 1.0) for d, _ in candidates[:top_n]]

        # 构造 MultiPathRetriever(绕开真的 init)
        pipe = MultiPathRetriever.__new__(MultiPathRetriever)
        pipe._vectorstore = FakeVectorstore()
        pipe._chunks = []
        pipe._bm25 = FakeBM25()
        pipe._reranker = FakeReranker()
        pipe._config = {
            "fusion_strategy": "rrf",
            "vector_weight": 0.6,
            "top_k_vector": 5,
            "top_k_bm25": 5,
            "reranker_top_k": 5,
        }

        # 调 retrieve,传两个不同 query
        pipe.retrieve(
            query="metformin 二甲双胍 血糖",  # BM25 query(带同义词)
            vector_query="Represent this question for searching relevant passages: metformin",  # 向量 query(带 BGE instruction)
        )

        # 验证 BM25 拿的是 keyword query
        assert bm25_called_with[0] == "metformin 二甲双胍 血糖"
        # 验证向量拿的是 vector query(带 BGE instruction)
        assert vec_called_with[0] == "Represent this question for searching relevant passages: metformin"
        # 关键:两个 query 不一样
        assert bm25_called_with[0] != vec_called_with[0]

    def test_22_retrieve_vector_query_defaults_to_query(self, monkeypatch):
        """不传 vector_query → 默认等于 query(向后兼容)"""
        from multi_path_retriever import MultiPathRetriever

        vec_called_with = []
        class FakeVectorstore:
            def similarity_search_with_score(self, q, **kwargs):
                vec_called_with.append(q)
                return []
        class FakeBM25:
            def search(self, q, **kwargs):
                return []
        class FakeReranker:
            def predict(self, pairs, **kwargs): return []
            def _get_recency_score(self, m): return 0.5
            def _get_authority_score(self, m): return 0.5
            criteria_weights = {"relevance": 0.6, "recency": 0.25, "authority": 0.15}
            def rerank(self, query, candidates, top_n=5): return []

        pipe = MultiPathRetriever.__new__(MultiPathRetriever)
        pipe._vectorstore = FakeVectorstore()
        pipe._chunks = []
        pipe._bm25 = FakeBM25()
        pipe._reranker = FakeReranker()
        pipe._config = {
            "fusion_strategy": "rrf",
            "vector_weight": 0.6,
            "top_k_vector": 5,
            "top_k_bm25": 5,
            "reranker_top_k": 5,
        }

        # 不传 vector_query
        pipe.retrieve(query="test query")
        # 向量检索应该用同一个 query
        assert vec_called_with[0] == "test query"

    def test_23_bm25_uses_synonyms_extended_query(self, monkeypatch):
        """
        **真实场景**:BM25 拿 enhanced.keyword_query(cleaned + 同义词),
        向量拿 enhanced.vector_query(BGE instruction + cleaned)
        """
        from multi_path_retriever import MultiPathRetriever

        bm25_q = []
        vec_q = []
        class FakeBM25:
            def search(self, q, **kwargs):
                bm25_q.append(q)
                return [(Document(page_content="d", metadata={"pmid": "1"}), 1.0)]
        class FakeVectorstore:
            def similarity_search_with_score(self, q, **kwargs):
                vec_q.append(q)
                return [(Document(page_content="d", metadata={"pmid": "1"}), 0.1)]
        class FakeReranker:
            def predict(self, pairs, **kwargs): return [1.0]
            def _get_recency_score(self, m): return 0.5
            def _get_authority_score(self, m): return 0.5
            criteria_weights = {"relevance": 0.6, "recency": 0.25, "authority": 0.15}
            def rerank(self, query, candidates, top_n=5):
                return [(d, 1.0) for d, _ in candidates[:top_n]]

        pipe = MultiPathRetriever.__new__(MultiPathRetriever)
        pipe._vectorstore = FakeVectorstore()
        pipe._chunks = []
        pipe._bm25 = FakeBM25()
        pipe._reranker = FakeReranker()
        pipe._config = {
            "fusion_strategy": "rrf",
            "vector_weight": 0.6,
            "top_k_vector": 5,
            "top_k_bm25": 5,
            "reranker_top_k": 5,
        }

        # 模拟 stage6 调 retrieve
        pipe.retrieve(
            query="metformin diabetes",  # keyword_query
            vector_query="Represent this question for searching relevant passages: metformin",  # vector_query
        )
        # BM25 拿到了含"metformin diabetes"全文,带同义词(虽然没具体)
        assert "metformin" in bm25_q[0]
        # 向量拿到了带 BGE instruction 的 query
        assert "Represent this question" in vec_q[0]
        # 两者文本不一样
        assert bm25_q[0] != vec_q[0]
