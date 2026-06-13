"""
阶段七测试套件:stage6_retrieval_pipeline.py 的 LLM 生成层
(重点覆盖 prompt 构造、答案结构、结果序列化、批量流程 — 不直接调真 LLM)

跑法:
    $env:HF_ENDPOINT="https://hf-mirror.com"
    D:\Anaconda\envs\medical_rag\python.exe -m pytest tests/test_stage7_answer_generation.py -v
"""

import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import asdict

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 静态 import (不会触发 __init__ 中的网络/模型加载)
from stage6_retrieval_pipeline import (
    RetrievalResult,
    RetrievalPipeline,
    MAX_FILES,
    VECTOR_PERSIST_DIR,
)


# ==================== Fixtures ====================
@pytest.fixture
def sample_docs():
    """5 篇模拟医学文献"""
    return [
        Document(
            page_content="ARNO contains three domains: a Sec7 domain in the middle, "
                         "a PH domain at the C-terminus, and a coiled-coil at the N-terminus.",
            metadata={
                "pmid": "12969509",
                "year": "2003",
                "journal": "BMC Cell Biology",
                "source": "PMC212319.xml",
                "title": "ARF protein and PLD activation",
            },
        ),
        Document(
            page_content="Phospholipase D (PLD) is activated by ARF proteins, "
                         "which promote membrane recruitment.",
            metadata={
                "pmid": "12969510",
                "year": "2004",
                "journal": "Nature",
                "source": "PMC212320.xml",
                "title": "ARF-dependent PLD signaling",
            },
        ),
        Document(
            page_content="Metformin reduces cardiovascular risk in type 2 diabetes patients.",
            metadata={
                "pmid": "15736996",
                "year": "2005",
                "journal": "Lancet",
                "source": "PMC300000.xml",
                "title": "Metformin CV outcomes",
            },
        ),
        Document(
            page_content="PD-1 blockade enhances T cell-mediated tumor rejection.",
            metadata={
                "pmid": "39018400",
                "year": "2024",
                "journal": "Science Advances",
                "source": "PMC466951.xml",
                "title": "PD-1 immunotherapy",
            },
        ),
        Document(
            page_content="ARF6 GTPase localizes to the plasma membrane and regulates endocytosis.",
            metadata={
                "pmid": "14737185",
                "year": "2004",
                "journal": "PLoS Biology",
                "source": "PMC314465.xml",
                "title": "ARF6 function",
            },
        ),
    ]


# ==================== RetrievalResult 结构测试 ====================
class TestRetrievalResult:
    """测试结果数据类的字段和行为"""

    def test_01_default_fields(self):
        """默认字段"""
        r = RetrievalResult(
            query="q", enhanced=None, retrieved_docs=[],
            retrieval_time_ms=0.0, fusion_strategy="rrf",
        )
        assert r.query == "q"
        assert r.enhanced is None
        assert r.retrieved_docs == []
        assert r.retrieval_time_ms == 0.0
        assert r.answer == ""
        assert r.generation_time_ms == 0.0
        assert r.total_time_ms == 0.0
        assert r.error is None
        assert r.fusion_strategy == "rrf"
        assert r.fusion_stats == {}

    def test_02_to_dict_basic(self):
        """to_dict 基本结构"""
        r = RetrievalResult(
            query="test",
            enhanced=None,
            retrieved_docs=[],
            retrieval_time_ms=100.5,
            fusion_strategy="weighted",
        )
        d = r.to_dict()
        assert d["query"] == "test"
        assert d["retrieved_count"] == 0
        assert d["retrieval_time_ms"] == 100.5
        assert d["fusion_strategy"] == "weighted"
        assert "answer" in d
        assert "sources" in d
        assert d["error"] is None

    def test_03_to_dict_with_docs(self, sample_docs):
        """带文档时 sources 字段正确"""
        r = RetrievalResult(
            query="q", enhanced=None, retrieved_docs=sample_docs[:3],
            retrieval_time_ms=0.0, fusion_strategy="rrf",
        )
        d = r.to_dict()
        assert d["retrieved_count"] == 3
        assert len(d["sources"]) == 3
        # 检查每条 source 都有 pmid/source/year/journal/preview
        for s in d["sources"]:
            assert "pmid" in s
            assert "year" in s
            assert "journal" in s
            assert "preview" in s
            assert len(s["preview"]) <= 102  # 100 + ...

    def test_04_to_dict_answer_truncation(self):
        """答案 > 200 字自动截断"""
        r = RetrievalResult(
            query="q", enhanced=None, retrieved_docs=[],
            retrieval_time_ms=0.0, fusion_strategy="rrf",
        )
        r.answer = "a" * 500
        d = r.to_dict()
        assert d["answer"].endswith("...")
        assert len(d["answer"]) == 203  # 200 + "..."

    def test_05_to_dict_short_answer(self):
        """短答案不截断"""
        r = RetrievalResult(
            query="q", enhanced=None, retrieved_docs=[],
            retrieval_time_ms=0.0, fusion_strategy="rrf",
        )
        r.answer = "短答案"
        d = r.to_dict()
        assert d["answer"] == "短答案"

    def test_06_to_dict_with_error(self):
        """错误信息被序列化"""
        r = RetrievalResult(
            query="q", enhanced=None, retrieved_docs=[],
            retrieval_time_ms=0.0, fusion_strategy="rrf",
            error="LLM call failed",
        )
        d = r.to_dict()
        assert d["error"] == "LLM call failed"


# ==================== Prompt 构造测试 ====================
class TestPromptConstruction:
    """测试 _build_prompt 生成的 prompt 模板"""

    @pytest.fixture
    def pipeline_with_mock_llm(self, sample_docs):
        """构造 Pipeline 但不触发真实模型加载,只测 prompt 构造"""
        pipe = RetrievalPipeline.__new__(RetrievalPipeline)
        pipe.config = {
            "fusion_strategy": "rrf",
            "criteria_weights": {"relevance": 0.6, "recency": 0.25, "authority": 0.15},
        }
        pipe.llm = MagicMock()
        return pipe

    def test_07_prompt_contains_question(self, pipeline_with_mock_llm, sample_docs):
        """prompt 包含用户问题"""
        prompt = pipeline_with_mock_llm._build_prompt("What is ARNO?", sample_docs)
        assert "What is ARNO?" in prompt

    def test_08_prompt_contains_all_docs(self, pipeline_with_mock_llm, sample_docs):
        """prompt 包含所有文档内容"""
        prompt = pipeline_with_mock_llm._build_prompt("test", sample_docs)
        for doc in sample_docs:
            # 至少有一段内容被引用
            assert doc.page_content[:50] in prompt

    def test_09_prompt_contains_pmid(self, pipeline_with_mock_llm, sample_docs):
        """prompt 包含 PMID 引用标号"""
        prompt = pipeline_with_mock_llm._build_prompt("test", sample_docs)
        for doc in sample_docs:
            assert doc.metadata["pmid"] in prompt

    def test_10_prompt_has_safety_rules(self, pipeline_with_mock_llm, sample_docs):
        """prompt 包含安全/严谨规则"""
        prompt = pipeline_with_mock_llm._build_prompt("test", sample_docs)
        # 至少 3 条规则:不得编造、引用标号、不知道就直说
        assert "不得编造" in prompt or "不得" in prompt
        assert "[1]" in prompt or "引用" in prompt
        assert "无法回答" in prompt or "不知道" in prompt

    def test_11_prompt_medical_disclaimer(self, pipeline_with_mock_llm, sample_docs):
        """prompt 包含医学免责(不建议诊断/用药)"""
        prompt = pipeline_with_mock_llm._build_prompt("test", sample_docs)
        # 必须包含医学警告
        assert "诊断" in prompt or "用药" in prompt or "执业医师" in prompt

    def test_12_prompt_doc_numbering(self, pipeline_with_mock_llm, sample_docs):
        """文档按顺序标号 【文献 1】..【文献 N】"""
        prompt = pipeline_with_mock_llm._build_prompt("test", sample_docs)
        for i in range(1, len(sample_docs) + 1):
            assert f"【文献 {i}】" in prompt

    def test_13_prompt_with_empty_docs(self, pipeline_with_mock_llm):
        """空文档列表仍能生成合法 prompt"""
        prompt = pipeline_with_mock_llm._build_prompt("test", [])
        assert "test" in prompt
        assert "文献片段" in prompt

    def test_14_prompt_metadata_headers(self, pipeline_with_mock_llm, sample_docs):
        """每篇文献的元数据在 header 里"""
        prompt = pipeline_with_mock_llm._build_prompt("test", sample_docs)
        # 年份、期刊、PMID 都应出现
        assert "2003" in prompt
        assert "BMC Cell Biology" in prompt
        assert "Nature" in prompt


# ==================== _generate_answer 测试 ====================
class TestGenerateAnswer:

    @pytest.fixture
    def mock_pipeline(self):
        pipe = RetrievalPipeline.__new__(RetrievalPipeline)
        pipe.llm = MagicMock()
        pipe.llm.invoke.return_value = "这是生成的答案 [1][2]"
        return pipe

    def test_15_generate_answer_calls_llm(self, mock_pipeline, sample_docs):
        """_generate_answer 会调 LLM"""
        result = mock_pipeline._generate_answer("q", sample_docs)
        assert mock_pipeline.llm.invoke.called

    def test_16_generate_answer_returns_text(self, mock_pipeline, sample_docs):
        """返回值是字符串"""
        result = mock_pipeline._generate_answer("q", sample_docs)
        assert isinstance(result, str)
        assert "[1][2]" in result

    def test_17_generate_answer_handles_exception(self, mock_pipeline, sample_docs):
        """LLM 异常时返回错误字符串(不抛出)"""
        mock_pipeline.llm.invoke.side_effect = Exception("Ollama connection refused")
        result = mock_pipeline._generate_answer("q", sample_docs)
        assert "LLM 调用失败" in result
        assert "Ollama" in result

    def test_18_generate_answer_passes_prompt(self, mock_pipeline, sample_docs):
        """传给 LLM 的是 prompt 字符串"""
        mock_pipeline._generate_answer("test question", sample_docs)
        # call_args[0][0] 是第一个位置参数(prompt)
        call_args = mock_pipeline.llm.invoke.call_args
        prompt = call_args[0][0]
        assert "test question" in prompt
        assert "文献片段" in prompt


# ==================== _print_sources 测试 ====================
class TestPrintSources:

    @pytest.fixture
    def pipeline(self):
        return RetrievalPipeline.__new__(RetrievalPipeline)

    def test_19_print_sources_empty(self, pipeline, capsys):
        """空文档不打印"""
        pipeline._print_sources([])
        captured = capsys.readouterr()
        assert "参考来源" not in captured.out

    def test_20_print_sources_basic(self, pipeline, sample_docs, capsys):
        """正常打印"""
        pipeline._print_sources(sample_docs[:3])
        captured = capsys.readouterr()
        assert "📚" in captured.out or "参考来源" in captured.out
        # PMID 都应该出现
        for doc in sample_docs[:3]:
            assert doc.metadata["pmid"] in captured.out

    def test_21_print_sources_dedup_by_pmid(self, pipeline, capsys):
        """同一 PMID 只打印一次"""
        docs = [
            Document(page_content="A", metadata={"pmid": "111", "source": "S1", "year": "2020"}),
            Document(page_content="B", metadata={"pmid": "111", "source": "S2", "year": "2021"}),
            Document(page_content="C", metadata={"pmid": "222", "source": "S3", "year": "2022"}),
        ]
        pipeline._print_sources(docs)
        captured = capsys.readouterr()
        # PMID 111 应该只出现 1 次(在 - PMID: 111 这一行)
        # 不算链接里的 111,只算"PMID: 111"出现次数
        pmid_count = captured.out.count("PMID: 111")
        assert pmid_count == 1
        # PMID 222 出现 1 次
        assert captured.out.count("PMID: 222") == 1

    def test_22_print_sources_has_pubmed_link(self, pipeline, sample_docs, capsys):
        """打印 PubMed 链接"""
        pipeline._print_sources(sample_docs[:2])
        captured = capsys.readouterr()
        for doc in sample_docs[:2]:
            expected_url = f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata['pmid']}/"
            assert expected_url in captured.out


# ==================== query() 端到端 (用 mock) ====================
class TestQueryMethodMocked:
    """端到端 query() 方法,但 mock 掉 LLM 和 retriever"""

    @pytest.fixture
    def pipeline_all_mocked(self):
        """完全 mock 化的 Pipeline"""
        pipe = RetrievalPipeline.__new__(RetrievalPipeline)
        pipe.config = {
            "fusion_strategy": "rrf",
            "criteria_weights": {"relevance": 0.6, "recency": 0.25, "authority": 0.15},
        }
        # mock enhancer
        pipe.query_enhancer = MagicMock()
        mock_enhanced = MagicMock()
        mock_enhanced.keyword_query = "enhanced query"
        mock_enhanced.cleaned = "enhanced"
        mock_enhanced.entities = {}
        mock_enhanced.synonyms = []
        mock_enhanced.filter_conditions = {}
        pipe.query_enhancer.enhance = MagicMock(return_value=mock_enhanced)

        # mock retriever
        pipe.retriever = MagicMock()
        pipe.retriever.retrieve = MagicMock(return_value=[
            Document(page_content="doc1", metadata={"pmid": "1", "year": "2020", "journal": "Nature"}),
            Document(page_content="doc2", metadata={"pmid": "2", "year": "2021", "journal": "Cell"}),
        ])

        # mock LLM
        pipe.llm = MagicMock()
        pipe.llm.invoke = MagicMock(return_value="Generated answer [1][2]")
        return pipe

    def test_23_query_uses_enhancer(self, pipeline_all_mocked):
        """query 调 QueryEnhancer"""
        result = pipeline_all_mocked.query("test", verbose=False)
        pipeline_all_mocked.query_enhancer.enhance.assert_called_once_with("test")

    def test_24_query_uses_retriever(self, pipeline_all_mocked):
        """query 调 retriever"""
        result = pipeline_all_mocked.query("test", verbose=False)
        pipeline_all_mocked.retriever.retrieve.assert_called_once()

    def test_25_query_generates_answer(self, pipeline_all_mocked):
        """默认 generate_answer=True 会调 LLM"""
        result = pipeline_all_mocked.query("test", verbose=False)
        pipeline_all_mocked.llm.invoke.assert_called_once()
        assert result.answer == "Generated answer [1][2]"

    def test_26_query_skip_answer(self, pipeline_all_mocked):
        """generate_answer=False 不调 LLM"""
        result = pipeline_all_mocked.query("test", generate_answer=False, verbose=False)
        pipeline_all_mocked.llm.invoke.assert_not_called()
        assert result.answer == ""

    def test_27_query_skip_enhancer(self, pipeline_all_mocked):
        """use_enhanced=False 直接用 raw query"""
        result = pipeline_all_mocked.query("raw", use_enhanced=False, verbose=False)
        pipeline_all_mocked.query_enhancer.enhance.assert_not_called()
        # retriever 拿到的应该是原始 query(raw),不是 enhanced.keyword_query("kq")
        pipeline_all_mocked.retriever.retrieve.assert_called_once()
        call_args = pipeline_all_mocked.retriever.retrieve.call_args
        # stage6 调用形式:retrieve(query=..., year_filter=..., journal_filter=...)
        assert call_args.kwargs.get("query") == "raw" or call_args[1].get("query") == "raw"

    def test_28_query_timing_recorded(self, pipeline_all_mocked):
        """query 记录耗时"""
        result = pipeline_all_mocked.query("test", verbose=False)
        assert result.total_time_ms > 0
        assert result.retrieval_time_ms >= 0
        assert result.generation_time_ms >= 0

    def test_29_query_handles_exception(self, pipeline_all_mocked):
        """query 内部异常被捕获到 result.error"""
        pipeline_all_mocked.retriever.retrieve.side_effect = RuntimeError("boom")
        result = pipeline_all_mocked.query("test", verbose=False)
        assert result.error is not None
        assert "boom" in result.error
        assert result.retrieved_docs == []

    def test_30_query_no_docs_message(self, pipeline_all_mocked):
        """召回为空时给友好提示"""
        pipeline_all_mocked.retriever.retrieve.return_value = []
        result = pipeline_all_mocked.query("test", verbose=False)
        assert "未找到" in result.answer or "无法" in result.answer


# ==================== 集成测试 — evaluate() ====================
class TestEvaluateMethod:
    """测试 evaluate() 批量统计功能"""

    @pytest.fixture
    def mock_pipeline(self):
        pipe = RetrievalPipeline.__new__(RetrievalPipeline)
        pipe.config = {
            "fusion_strategy": "rrf",
            "criteria_weights": {"relevance": 0.6, "recency": 0.25, "authority": 0.15},
        }
        pipe.llm = MagicMock()
        pipe.query_enhancer = MagicMock()
        mock_eq = MagicMock()
        mock_eq.keyword_query = "kq"
        mock_eq.cleaned = "c"
        mock_eq.entities = {}
        mock_eq.synonyms = []
        mock_eq.filter_conditions = {}
        pipe.query_enhancer.enhance = MagicMock(return_value=mock_eq)
        pipe.retriever = MagicMock()
        pipe.retriever.retrieve = MagicMock(return_value=[
            Document(page_content="d1", metadata={"pmid": "1", "year": "2020"}),
            Document(page_content="d2", metadata={"pmid": "2", "year": "2021"}),
        ])
        return pipe

    def test_31_evaluate_runs_all_queries(self, mock_pipeline):
        """evaluate 跑所有 queries"""
        queries = ["q1", "q2", "q3"]
        report = mock_pipeline.evaluate(queries, use_enhanced=True)
        assert report["total_queries"] == 3
        assert report["success"] == 3
        assert report["failure"] == 0

    def test_32_evaluate_no_llm_calls(self, mock_pipeline):
        """evaluate 不调 LLM(为了快)"""
        queries = ["q1", "q2"]
        report = mock_pipeline.evaluate(queries)
        mock_pipeline.llm.invoke.assert_not_called()

    def test_33_evaluate_avg_metrics(self, mock_pipeline):
        """evaluate 算平均召回/耗时"""
        queries = ["q1", "q2", "q3"]
        report = mock_pipeline.evaluate(queries)
        # 每次返回 2 docs → avg = 2.0
        assert report["avg_recall_per_query"] == 2.0
        assert report["has_documents"] == 3
        assert report["no_documents"] == 0

    def test_34_evaluate_handles_failure(self, mock_pipeline):
        """evaluate 统计失败数"""
        # 第一次抛错
        call_count = [0]
        def maybe_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first call fails")
            return [Document(page_content="ok", metadata={"pmid": "1"})]
        mock_pipeline.retriever.retrieve = maybe_fail

        queries = ["q1", "q2"]
        report = mock_pipeline.evaluate(queries)
        assert report["success"] == 1
        assert report["failure"] == 1


# ==================== batch_query 测试 ====================
class TestBatchQuery:

    def test_35_batch_query_default_no_answer(self):
        """batch_query 默认不调 LLM"""
        pipe = RetrievalPipeline.__new__(RetrievalPipeline)
        pipe.config = {"fusion_strategy": "rrf",
                       "criteria_weights": {"relevance": 0.6, "recency": 0.25, "authority": 0.15}}
        pipe.llm = MagicMock()
        pipe.query_enhancer = MagicMock()
        eq = MagicMock(keyword_query="k", cleaned="c", entities={}, synonyms=[], filter_conditions={})
        pipe.query_enhancer.enhance = MagicMock(return_value=eq)
        pipe.retriever = MagicMock()
        pipe.retriever.retrieve = MagicMock(return_value=[Document(page_content="d", metadata={"pmid": "1"})])

        results = pipe.batch_query(["q1", "q2"])
        assert len(results) == 2
        pipe.llm.invoke.assert_not_called()

    def test_36_batch_query_returns_results_list(self):
        """返回 list[RetrievalResult]"""
        pipe = RetrievalPipeline.__new__(RetrievalPipeline)
        pipe.config = {"fusion_strategy": "rrf",
                       "criteria_weights": {"relevance": 0.6, "recency": 0.25, "authority": 0.15}}
        pipe.llm = MagicMock()
        pipe.query_enhancer = MagicMock()
        eq = MagicMock(keyword_query="k", cleaned="c", entities={}, synonyms=[], filter_conditions={})
        pipe.query_enhancer.enhance = MagicMock(return_value=eq)
        pipe.retriever = MagicMock()
        pipe.retriever.retrieve = MagicMock(return_value=[Document(page_content="d", metadata={"pmid": "1"})])

        results = pipe.batch_query(["q1", "q2"])
        for r in results:
            assert isinstance(r, RetrievalResult)
