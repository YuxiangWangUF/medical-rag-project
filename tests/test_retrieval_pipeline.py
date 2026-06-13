"""
阶段六测试套件:retrieval_pipeline.py 的 30 个单元测试

覆盖:
- MultiPathRetriever:_doc_key、_fuse_simple、_fuse_rrf、_fuse_weighted、_normalize_scores、retrieval 端到端
- MultiCriteriaReranker:_recency_score、_authority_score、rerank 端到端
- RetrievalPipeline:summary、Pipeline 组件装配
- 集成:QueryEnhancer + Pipeline 联调(用 mock 嵌入 + mock reranker)

跑法:
    $env:HF_ENDPOINT="https://hf-mirror.com"
    D:\Anaconda\envs\medical_rag\python.exe -m pytest tests/test_retrieval_pipeline.py -v
"""

import math
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# 兼容:在仓库根目录跑测试
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_pipeline import (
    MultiPathRetriever,
    MultiCriteriaReranker,
    RetrievalPipeline,
    RetrievalResult,
    JOURNAL_AUTHORITY,
    DEFAULT_CRITERIA_WEIGHTS,
)


# ==================== Fixtures ====================
@pytest.fixture
def sample_chunks():
    """15 个 chunks,5 篇文章,每篇 3 chunks"""
    articles = {
        "11111": ("ARNO protein activation", 2020, "Nature"),
        "22222": ("metformin diabetes", 2023, "Cell"),
        "33333": ("hypertension treatment", 2010, "PLoS Medicine"),
        "44444": ("cancer immunotherapy", 2022, "Lancet"),
        "55555": ("cardiovascular disease", 2018, "BMC"),
    }
    chunks = []
    for pmid, (topic, year, journal) in articles.items():
        for i in range(3):
            chunks.append(Document(
                page_content=f"{topic} - content chunk {i}",
                metadata={
                    "doc_id": pmid,
                    "source": pmid,
                    "source_title": f"Article about {topic}",
                    "year": str(year),
                    "journal": journal,
                    "chunk_id": f"{pmid}_chunk_{i:03d}",
                },
            ))
    return chunks


@pytest.fixture
def mock_vectorstore(sample_chunks, tmp_path):
    """ChromaDB 实例,基于 mock 嵌入(随机但确定的向量)"""
    # 固定维度的伪嵌入:用 hash
    def fake_embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for t in texts:
            # 用 text 长度 + ord 求和生成一个固定向量(只用于构造 ChromaDB)
            h = sum(ord(c) for c in t)
            v = [(h + i) % 100 / 100.0 for i in range(384)]
            result.append(v)
        return result

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return fake_embed(texts)

        def embed_query(self, text):
            return fake_embed([text])[0]

    vs = Chroma.from_documents(
        documents=sample_chunks,
        embedding=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_collection",
    )
    return vs


@pytest.fixture
def mock_bge_reranker(monkeypatch):
    """替换 CrossEncoder,返回固定分数(基于 query/doc 长度差异)"""
    fake_ce = MagicMock()

    def fake_predict(pairs, show_progress_bar=False):
        scores = []
        for q, d in pairs:
            # 让"ARNO" query 在 ARNO 文档上分数高
            if "ARNO" in q and "ARNO" in d:
                scores.append(5.0)
            elif "metformin" in q.lower() and "metformin" in d.lower():
                scores.append(4.0)
            elif "cancer" in q.lower() and "cancer" in d.lower():
                scores.append(3.0)
            elif "hypertension" in q.lower() and "hypertension" in d.lower():
                scores.append(2.0)
            else:
                # 长度差异:越短越高
                scores.append(2.0 - abs(len(q) - len(d)) * 0.001)
        return scores

    fake_ce.predict = fake_predict
    monkeypatch.setattr("retrieval_pipeline.CrossEncoder", lambda *a, **k: fake_ce)
    return fake_ce


# ==================== MultiPathRetriever 测试 ====================
class TestMultiPathRetriever:

    def test_01_init_default(self, mock_vectorstore, sample_chunks):
        """默认参数初始化"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        assert mp.fusion_strategy == "rrf"
        assert mp.rrf_k == 60
        assert mp.vector_weight == 0.6
        assert mp.keyword_weight == 0.4
        assert len(mp.chunks) == 15

    def test_02_init_invalid_strategy(self, mock_vectorstore, sample_chunks):
        """无效融合策略要抛错"""
        with pytest.raises(ValueError, match="Unknown fusion_strategy"):
            MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="foo")

    @pytest.mark.parametrize("strategy", ["simple", "rrf", "weighted"])
    def test_03_three_strategies(self, mock_vectorstore, sample_chunks, strategy):
        """三种策略都能初始化"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy=strategy)
        assert mp.fusion_strategy == strategy

    def test_04_normalize_scores_basic(self, mock_vectorstore, sample_chunks):
        """Min-max 归一化:最小=0,最大=1"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        result = mp._normalize_scores([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_05_normalize_scores_all_same(self, mock_vectorstore, sample_chunks):
        """所有分数相同 → 全 1.0"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        result = mp._normalize_scores([3.0, 3.0, 3.0])
        assert result == [1.0, 1.0, 1.0]

    def test_06_normalize_scores_empty(self, mock_vectorstore, sample_chunks):
        """空列表"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        assert mp._normalize_scores([]) == []

    def test_07_doc_key_priority(self, mock_vectorstore, sample_chunks):
        """_doc_key 优先级:doc_id > source > id(doc)"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        d1 = Document(page_content="x", metadata={"doc_id": "111", "source": "222"})
        d2 = Document(page_content="x", metadata={"source": "333"})
        d3 = Document(page_content="x", metadata={})
        assert mp._doc_key(d1) == "111"
        assert mp._doc_key(d2) == "333"
        assert mp._doc_key(d3) == str(id(d3))

    def test_08_fuse_simple_dedup(self, mock_vectorstore, sample_chunks):
        """simple 融合:同一 doc_id 去重,向量优先"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="simple")
        # 构造 2 个 vec + 2 个 kw,1 个重复
        v1 = Document(page_content="vec1", metadata={"doc_id": "111", "source": "111"})
        v2 = Document(page_content="vec2", metadata={"doc_id": "222", "source": "222"})
        k1 = Document(page_content="kw1", metadata={"doc_id": "111", "source": "111"})  # 重复
        k2 = Document(page_content="kw2", metadata={"doc_id": "333", "source": "333"})
        fused = mp._fuse_simple([v1, v2], [k1, k2])
        # 应该 3 个:111(vec), 222(vec), 333(kw)
        assert len(fused) == 3
        assert [d.page_content for d in fused] == ["vec1", "vec2", "kw2"]

    def test_09_fuse_rrf_ordering(self, mock_vectorstore, sample_chunks):
        """RRF:在两个列表都出现的 doc,分数 = 1/(rank+60) * 0.6 + 1/(rank+60) * 0.4"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        v = [Document(page_content=f"v{i}", metadata={"doc_id": f"d{i}", "source": f"d{i}"}) for i in range(3)]
        k = [Document(page_content=f"k{i}", metadata={"doc_id": f"d{i}", "source": f"d{i}"}) for i in range(2)]
        fused = mp._fuse_rrf(v, k)
        # d0 在两个都排第 1,分数最高
        assert fused[0].metadata["doc_id"] == "d0"
        # d1,k1 排第 2 各自独占
        assert fused[1].metadata["doc_id"] in ("d1", "k1")

    def test_10_fuse_weighted(self, mock_vectorstore, sample_chunks):
        """weighted:用归一化分数加权"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="weighted")
        v = [Document(page_content="v0", metadata={"doc_id": "A", "source": "A"}),
             Document(page_content="v1", metadata={"doc_id": "B", "source": "B"})]
        v_scores = [0.9, 0.1]
        k = [Document(page_content="k0", metadata={"doc_id": "B", "source": "B"}),
             Document(page_content="k1", metadata={"doc_id": "C", "source": "C"})]
        k_scores = [0.8, 0.2]
        fused = mp._fuse_weighted(v, k, v_scores, k_scores)
        # 算分:
        # A: 0.6 * 1.0 = 0.6
        # B: 0.6 * 0.0 + 0.4 * 1.0 = 0.4
        # C: 0.4 * 0.0 = 0.0
        # A 排第 1
        assert [d.metadata["doc_id"] for d in fused] == ["A", "B", "C"]

    def test_11_retrieve_rrf_top_k(self, mock_vectorstore, sample_chunks):
        """端到端 retrieve:RRF 策略返回 top_k"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        emb = [0.5] * 384
        docs = mp.retrieve("ARNO protein", emb, top_k_vector=5, top_k_keyword=5, top_k_final=3)
        assert len(docs) <= 3
        assert all(isinstance(d, Document) for d in docs)

    def test_12_retrieve_weighted(self, mock_vectorstore, sample_chunks):
        """weighted 端到端"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="weighted")
        emb = [0.5] * 384
        docs = mp.retrieve("cancer", emb, top_k_vector=5, top_k_keyword=5, top_k_final=5)
        assert len(docs) <= 5

    def test_13_retrieve_simple(self, mock_vectorstore, sample_chunks):
        """simple 端到端"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="simple")
        emb = [0.5] * 384
        docs = mp.retrieve("metformin", emb, top_k_vector=5, top_k_keyword=5, top_k_final=10)
        assert len(docs) <= 10

    def test_14_keyword_search_returns_docs(self, mock_vectorstore, sample_chunks):
        """BM25 keyword_search 直接调用"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        docs = mp.keyword_search("metformin diabetes", top_k=3)
        assert len(docs) == 3
        assert any("metformin" in d.page_content for d in docs)

    def test_15_keyword_search_k_change(self, mock_vectorstore, sample_chunks):
        """改 top_k 后 BM25 返回数也变"""
        mp = MultiPathRetriever(mock_vectorstore, sample_chunks, fusion_strategy="rrf")
        assert len(mp.keyword_search("cancer", top_k=2)) == 2
        assert len(mp.keyword_search("cancer", top_k=5)) == 5


# ==================== MultiCriteriaReranker 测试 ====================
class TestMultiCriteriaReranker:

    def test_16_init_default_weights(self, mock_bge_reranker):
        """默认权重与权重字典一致"""
        rr = MultiCriteriaReranker()
        assert rr.weights == DEFAULT_CRITERIA_WEIGHTS

    def test_17_init_custom_weights(self, mock_bge_reranker):
        """自定义权重"""
        custom = {"relevance": 0.5, "recency": 0.3, "authority": 0.2}
        rr = MultiCriteriaReranker(criteria_weights=custom)
        assert rr.weights["relevance"] == 0.5

    def test_18_recency_score_current_year(self, mock_bge_reranker):
        """今年=1.0"""
        rr = MultiCriteriaReranker()
        assert rr._recency_score("2026") == pytest.approx(1.0)

    def test_19_recency_score_decay(self, mock_bge_reranker):
        """5 年前 = 0.5(每年 -0.1)"""
        rr = MultiCriteriaReranker(current_year=2026)
        assert rr._recency_score("2021") == pytest.approx(0.5)

    def test_20_recency_score_old(self, mock_bge_reranker):
        """>40 年前 → 最低 0.1"""
        rr = MultiCriteriaReranker(current_year=2026)
        assert rr._recency_score("1970") == pytest.approx(0.1)

    def test_21_recency_score_invalid(self, mock_bge_reranker):
        """无效年份 = 0.5"""
        rr = MultiCriteriaReranker()
        assert rr._recency_score("") == 0.5
        assert rr._recency_score("abc") == 0.5
        assert rr._recency_score(None) == 0.5

    def test_22_authority_known_journal(self, mock_bge_reranker):
        """Nature 10.0 / 10 = 1.0"""
        rr = MultiCriteriaReranker()
        assert rr._authority_score("Nature") == pytest.approx(1.0)
        assert rr._authority_score("Nature Medicine") == pytest.approx(1.0)

    def test_23_authority_bmc(self, mock_bge_reranker):
        """BMC 3.0 / 10 = 0.3"""
        rr = MultiCriteriaReranker()
        assert rr._authority_score("BMC Genomics") == pytest.approx(0.3)

    def test_24_authority_unknown(self, mock_bge_reranker):
        """未知期刊 = default 4.0 / 10 = 0.4"""
        rr = MultiCriteriaReranker()
        assert rr._authority_score("Random Journal XYZ") == pytest.approx(0.4)

    def test_25_authority_case_insensitive(self, mock_bge_reranker):
        """大小写不敏感"""
        rr = MultiCriteriaReranker()
        assert rr._authority_score("nature") == pytest.approx(1.0)
        assert rr._authority_score("NATURE") == pytest.approx(1.0)

    def test_26_authority_empty(self, mock_bge_reranker):
        """空字符串 / None = default"""
        rr = MultiCriteriaReranker()
        assert rr._authority_score("") == pytest.approx(0.4)
        assert rr._authority_score(None) == pytest.approx(0.4)

    def test_27_rerank_empty(self, mock_bge_reranker):
        """空列表 → 空结果"""
        rr = MultiCriteriaReranker()
        assert rr.rerank("query", []) == []

    def test_28_rerank_top_k(self, mock_bge_reranker, sample_chunks):
        """rerank 端到端:返回 top_k,按分数降序"""
        rr = MultiCriteriaReranker()
        result = rr.rerank("ARNO protein", sample_chunks[:5], top_k=3)
        assert len(result) == 3
        # 按 score 降序
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
        # ARNO chunk 应该排第 1
        assert "ARNO" in result[0][0].page_content

    def test_29_rerank_recency_bias(self, mock_bge_reranker, sample_chunks):
        """时效性 bias:同 relevance,recency 高的排前"""
        rr = MultiCriteriaReranker(
            criteria_weights={"relevance": 0.0, "recency": 1.0, "authority": 0.0}
        )
        chunks = [
            Document(page_content="x", metadata={"doc_id": "1", "year": "2000"}),
            Document(page_content="x", metadata={"doc_id": "2", "year": "2025"}),
        ]
        result = rr.rerank("q", chunks, top_k=2)
        # 2025 的排第 1
        assert result[0][0].metadata["doc_id"] == "2"

    def test_30_rerank_authority_bias(self, mock_bge_reranker, sample_chunks):
        """权威性 bias:同 relevance,authority 高的排前"""
        rr = MultiCriteriaReranker(
            criteria_weights={"relevance": 0.0, "recency": 0.0, "authority": 1.0}
        )
        chunks = [
            Document(page_content="x", metadata={"doc_id": "1", "journal": "BMC"}),
            Document(page_content="x", metadata={"doc_id": "2", "journal": "Nature"}),
        ]
        result = rr.rerank("q", chunks, top_k=2)
        assert result[0][0].metadata["doc_id"] == "2"


# ==================== RetrievalResult / Integration 测试 ====================
class TestRetrievalResultSummary:
    """测试 RetrievalResult.summary() 的输出格式"""

    def test_31_summary_basic(self):
        """基础 summary"""
        r = RetrievalResult(
            query="test",
            vector_query="vec_query",
            keyword_query="kw_query",
            fusion_strategy="rrf",
        )
        s = r.summary()
        assert "test" in s
        assert "vec_query" in s
        assert "kw_query" in s
        assert "rrf" in s
        assert "召回数" in s

    def test_32_summary_with_reranked(self):
        """有重排结果时显示 Top 3"""
        r = RetrievalResult(query="q", fusion_strategy="weighted")
        r.reranked_docs = [
            (Document(page_content="a", metadata={"doc_id": "111", "source_title": "Title A"}), 0.9),
            (Document(page_content="b", metadata={"doc_id": "222", "source_title": "Title B"}), 0.5),
        ]
        s = r.summary()
        assert "PMID:111" in s
        assert "PMID:222" in s
        assert "Title A" in s

    def test_33_summary_no_reranked(self):
        """无重排结果时不显示 Top"""
        r = RetrievalResult(query="q")
        s = r.summary()
        assert "Top" not in s


class TestRetrievalPipelineIntegration:
    """端到端 Pipeline 集成(用 mock 嵌入 + mock reranker)"""

    def test_34_pipeline_init_components(self, mock_bge_reranker, monkeypatch, tmp_path):
        """Pipeline 初始化所有组件"""
        # mock QueryEnhancer
        mock_eq = MagicMock()
        mock_qe = MagicMock()
        mock_qe.enhance = MagicMock(return_value=mock_eq)
        mock_eq.vector_query = "vq"
        mock_eq.keyword_query = "kq"
        mock_eq.filter_conditions = {}
        monkeypatch.setitem(sys.modules, "query_enhancer", MagicMock(QueryEnhancer=lambda: mock_qe))

        # mock HuggingFaceEmbeddings(避免真实加载)
        def fake_embed_query(text):
            return [0.1] * 384

        def fake_embed_documents(texts):
            return [[0.1] * 384 for _ in texts]

        FakeEmbeddings = MagicMock()
        FakeEmbeddings.return_value.embed_query = fake_embed_query
        FakeEmbeddings.return_value.embed_documents = fake_embed_documents
        monkeypatch.setattr("retrieval_pipeline.HuggingFaceEmbeddings", FakeEmbeddings)

        # 准备一个小的 ChromaDB
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        def fake_embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            return [[0.1 * i for i in range(384)] for _ in texts]

        class FakeEmb:
            def embed_documents(self, texts):
                return fake_embed(texts)

            def embed_query(self, text):
                return fake_embed([text])[0]

        chunks = [
            Document(page_content=f"doc {i}", metadata={"doc_id": f"d{i}", "source": f"d{i}"})
            for i in range(3)
        ]
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=FakeEmb(),
            persist_directory=str(tmp_path / "vs"),
            collection_name="test_pipe",
        )

        # 准备 parquet
        import pandas as pd
        df = pd.DataFrame({
            "chunk_id": [f"d{i}_chunk_000" for i in range(3)],
            "text": [f"doc {i}" for i in range(3)],
            "doc_id": [f"d{i}" for i in range(3)],
            "source_title": [f"Title {i}" for i in range(3)],
        })
        df.to_parquet(tmp_path / "chunks.parquet")

        pipeline = RetrievalPipeline(
            vector_db_dir=str(tmp_path / "vs"),
            collection_name="test_pipe",
            embedding_model="fake",
            reranker_model="fake",
            fusion_strategy="weighted",
            enable_enhancer=True,
        )

        # 校验组件都存在
        assert pipeline.multi_path is not None
        assert pipeline.reranker is not None
        assert pipeline.enhancer is not None
        assert pipeline.multi_path.fusion_strategy == "weighted"
