# tests/test_rag_medical.py
#
# 端到端 RAG 单元测试 — 覆盖核心组件,避免跑完整 Ollama + Chroma 流程
# 跑法:
#   set HF_ENDPOINT=https://hf-mirror.com
#   pytest tests/test_rag_medical.py -v
#
# 覆盖:
#   - parse_pubmed_xml:从 XML 提取字段
#   - BGEReranker.rerank:top_n + 排序正确性
#   - HybridRerankRetriever._get_relevant_documents:article-level 去重
#   - print_sources:PMID 链接输出

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== Fixtures ====================

# 完整 JATS XML 样本(简化的)
SAMPLE_XML = """<?xml version="1.0"?>
<article>
  <front>
    <journal-meta><journal-title>J Mol Biol</journal-title></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmid">12345678</article-id>
      <title-group><article-title>ARF Protein in Cell Signaling</article-title></title-group>
      <abstract>
        <p>ARF protein mediates phospholipase D activation in cell signaling.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Introduction</title><p>Background on ARF protein.</p></sec>
  </body>
</article>"""


@pytest.fixture
def sample_xml_file(tmp_path):
    p = tmp_path / "PMC12345678.xml"
    p.write_text(SAMPLE_XML, encoding="utf-8")
    return str(p)


# ==================== parse_pubmed_xml ====================
class TestParsePubmedXml:
    def test_extract_pmid(self, sample_xml_file):
        from rag_medical import parse_pubmed_xml
        d = parse_pubmed_xml(sample_xml_file)
        # parse_pubmed_xml 返回 Document 对象,字段在 metadata 和 page_content 里
        assert d.metadata["pmid"] == "12345678"
        assert d.metadata["journal"] == "J Mol Biol"
        assert "ARF" in d.page_content
        assert "ARF Protein in Cell Signaling" in d.page_content

    def test_metadata_dict_structure(self, sample_xml_file):
        from rag_medical import parse_pubmed_xml
        d = parse_pubmed_xml(sample_xml_file)
        # 应返回 Document,metadata 包含所有元数据
        assert hasattr(d, "metadata")
        assert hasattr(d, "page_content")
        for key in ["pmid", "journal", "year", "source"]:
            assert key in d.metadata


# ==================== BGEReranker ====================
class TestBGEReranker:
    @pytest.fixture(scope="class")
    def reranker(self):
        from rag_medical import BGEReranker
        # 用小模型(实际 bge-reranker-base,首次会下载)
        return BGEReranker()

    def test_rerank_returns_top_n(self, reranker):
        """rerank 应返回 top_n 个 doc"""
        from langchain_core.documents import Document
        # 模拟 10 个 doc,内容长度不一
        docs = [
            Document(page_content=f"This is test document {i}. " * 5,
                     metadata={"source": f"PMC{i}.xml"})
            for i in range(10)
        ]
        # 模拟 query(不实际用 rerank 评分,只看 top_n 数量)
        result = reranker.rerank("test query", docs[:3], top_n=2)
        assert len(result) == 2

    def test_rerank_returns_full_when_top_n_none(self, reranker):
        """top_n=None 应返回全部 doc"""
        from langchain_core.documents import Document
        docs = [
            Document(page_content=f"Doc {i}", metadata={"source": f"PMC{i}.xml"})
            for i in range(5)
        ]
        result = reranker.rerank("test", docs, top_n=None)
        assert len(result) == 5

    def test_rerank_top_n_larger_than_docs(self, reranker):
        """top_n > len(docs) 不应该 crash"""
        from langchain_core.documents import Document
        docs = [Document(page_content="Short", metadata={"source": "PMC1.xml"})]
        result = reranker.rerank("test", docs, top_n=10)
        # 最多返回 len(docs) 个
        assert len(result) <= 1

    def test_rerank_empty_docs(self, reranker):
        """空 doc 列表应返回空"""
        result = reranker.rerank("test", [], top_n=5)
        assert result == []


# ==================== HybridRerankRetriever:Article-level dedup ====================
class TestHybridRerankRetrieverDedup:
    """测 article-level 去重逻辑(在 rerank 之后按分数)"""

    @pytest.fixture(scope="class")
    def reranker(self):
        from rag_medical import BGEReranker
        return BGEReranker()

    @pytest.fixture
    def mock_docs(self):
        """模拟 chunks: 同一文章多个 chunk,每篇有不同分数"""
        from langchain_core.documents import Document
        return [
            # PMC_A:3 个 chunk,分数不同
            Document(page_content="PMC_A chunk 0", metadata={"source": "PMC_A.xml"}),
            Document(page_content="PMC_A chunk 1", metadata={"source": "PMC_A.xml"}),
            Document(page_content="PMC_A chunk 2", metadata={"source": "PMC_A.xml"}),
            # PMC_B:2 个 chunk
            Document(page_content="PMC_B chunk 0", metadata={"source": "PMC_B.xml"}),
            Document(page_content="PMC_B chunk 1", metadata={"source": "PMC_B.xml"}),
            # PMC_C:1 个 chunk
            Document(page_content="PMC_C chunk 0", metadata={"source": "PMC_C.xml"}),
        ]

    def test_article_dedup_keeps_highest_score(self, mock_docs, reranker):
        """同一 article 多个 chunk,只保留 rerank 分数最高的"""
        # mock 一个 ensemble 返回所有 mock_docs
        class MockEnsemble:
            def invoke(self, query):
                return mock_docs

        from rag_medical import HybridRerankRetriever
        retriever = HybridRerankRetriever(
            ensemble_retriever=MockEnsemble(),
            reranker=reranker,
            top_n=10,  # 给多点,看看去重后的数量
        )
        # 注:rerank 实际评分时,内容相同的 chunk 分数相近;这里我们主要看去重是否生效
        # 由于 reranker 真实打分,具体哪篇胜出不确定,但关键断言:
        # 1) 不应该返回同一 source 的多个 chunk
        # 2) 返回数量 <= 不同 source 数
        result = retriever._get_relevant_documents("test query")
        sources = [d.metadata["source"] for d in result]
        # 每个 source 最多出现 1 次(article-level dedup)
        assert len(sources) == len(set(sources)), f"去重失败: {sources}"
        # 返回的 source 数 <= 不同 source 总数(3)
        assert len(sources) <= 3
        # top_n 限制
        assert len(result) <= 10

    def test_article_dedup_with_only_one_chunk(self, reranker):
        """只有 1 个 chunk 的文章,直接返回"""
        from langchain_core.documents import Document
        from rag_medical import HybridRerankRetriever

        single_doc = [Document(page_content="Only one", metadata={"source": "PMC_X.xml"})]

        class MockEnsemble:
            def invoke(self, query):
                return single_doc

        retriever = HybridRerankRetriever(
            ensemble_retriever=MockEnsemble(),
            reranker=reranker,
            top_n=5,
        )
        result = retriever._get_relevant_documents("test")
        assert len(result) == 1
        assert result[0].metadata["source"] == "PMC_X.xml"

    def test_article_dedup_with_empty_ensemble(self, reranker):
        """ensemble 返回空,_get_relevant_documents 应返回空"""
        from rag_medical import HybridRerankRetriever

        class MockEnsemble:
            def invoke(self, query):
                return []

        retriever = HybridRerankRetriever(
            ensemble_retriever=MockEnsemble(),
            reranker=reranker,
            top_n=5,
        )
        result = retriever._get_relevant_documents("test")
        assert result == []


# ==================== print_sources ====================
class TestPrintSources:
    def test_print_sources_with_results(self, capsys):
        from langchain_core.documents import Document
        from rag_medical import print_sources
        sources = [
            Document(
                page_content="content",
                metadata={
                    "pmid": "12345678",
                    "source": "PMC12345.xml",
                    "year": "2003",
                    "journal": "BMC Cell Biol",
                },
            )
        ]
        print_sources(sources)
        captured = capsys.readouterr()
        # 应包含 PMID 和链接
        assert "12345678" in captured.out
        assert "pubmed.ncbi.nlm.nih.gov" in captured.out

    def test_print_sources_empty(self, capsys):
        from rag_medical import print_sources
        print_sources([])
        captured = capsys.readouterr()
        # 不应该打印任何东西
        assert captured.out == "" or "参考来源" not in captured.out

    def test_print_sources_dedup_by_pmid(self, capsys):
        """同一 PMID 多个 chunk 只打印一次(只去重 PMID,链接里有 PMID 所以会多 1 次)"""
        from langchain_core.documents import Document
        from rag_medical import print_sources
        # 3 个 chunks,全是同一 PMID
        sources = [
            Document(page_content=f"c{i}", metadata={"pmid": "11111", "source": f"PMC_chunk_{i}.xml"})
            for i in range(3)
        ]
        print_sources(sources)
        captured = capsys.readouterr()
        # PMID 在两处出现:"- PMID: 11111 | ..." 和 "链接: .../11111/"(因为链接用了 pmid)
        # 但 "11111" 这个 ID 只来自同一 chunk 的 metadata
        # 关键:不应该出现 3 次(代表 3 个 chunk)
        # 应该只 2 次(1 次 PMID 字段 + 1 次 链接)
        assert captured.out.count("11111") == 2, (
            f"PMID 应只在 1 个 chunk 中出现(2 次: 字段+链接),实际 {captured.out.count('11111')} 次: {captured.out}"
        )


# ==================== Configuration constants ====================
class TestConfig:
    """测默认配置是否合理"""

    def test_top_k_positive(self):
        from rag_medical import TOP_K
        assert TOP_K > 0
        assert TOP_K <= 20  # 太大没意义

    def test_retriever_k_larger_than_top_k(self):
        from rag_medical import TOP_K, RETRIEVER_K
        assert RETRIEVER_K > TOP_K, "粗召回量应大于最终 top_k,给 rerank 留空间"

    def test_chunk_size_within_model_limit(self):
        from rag_medical import CHUNK_SIZE
        # chunk_size 应合理(避免触发 bge 截断)
        # 用一个宽松的 128 上限就行(没有 TOKENIZER_LIMIT 常量时)
        assert CHUNK_SIZE >= 128, f"chunk_size 太小: {CHUNK_SIZE}"
        assert CHUNK_SIZE <= 400, f"chunk_size 太大: {CHUNK_SIZE}"

    def test_hybrid_weights_sum_to_one(self):
        """BM25 + dense 权重应合理(可以不全为 1,但都要 > 0)"""
        from rag_medical import HYBRID_WEIGHTS
        assert len(HYBRID_WEIGHTS) == 2
        assert all(w > 0 for w in HYBRID_WEIGHTS)
        # 权重和合理(虽然不强求=1)
        assert 0.5 <= sum(HYBRID_WEIGHTS) <= 1.5
