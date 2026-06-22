"""
Stage 8 Part 2.3 — Tests for medical_generation_pipeline.py

覆盖:
1. 辅助方法:_filter_by_evaluation / _format_sources / _postprocess
2. 流水线接口(mock LLMGenerator,验证每个阶段都调用了)
3. 错误降级(LLM 失败时整个 pipeline 不崩)
4. 完整结果结构完整性
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from context_assembler import DocumentChunk
from llm_generator import GenerationConfig, GenerationResult
from medical_generation_pipeline import (
    GenerationMetrics,
    MedicalGenerationPipeline,
    PipelineResult,
    quick_generate,
)
from prompt_templates import (
    ANSWER_GENERATOR,
    CRITICAL_REVIEWER,
    EVIDENCE_EVALUATOR,
    FINAL_ASSEMBLER,
)


def _ok_result(text: str, prompt_tokens: int = 50, response_tokens: int = 100) -> GenerationResult:
    """构造一个成功的 mock GenerationResult"""
    return GenerationResult(
        text=text,
        parsed_json=None,
        raw_response={"message": {"content": text}},
        elapsed_ms=100.0,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        retries=0,
        success=True,
    )


def _fail_result(error: str = "mock failure") -> GenerationResult:
    return GenerationResult(text="", success=False, error=error, retries=2)


def _make_mock_pipeline(enable_review: bool = True, fail_at: str = "") -> MedicalGenerationPipeline:
    """
    构造一个 mock 了 LLMGenerator 的 pipeline。
    fail_at: "" / "eval" / "gen" / "review" / "final"
    """
    pipeline = MedicalGenerationPipeline(
        llm_model="qwen3:8b",
        enable_review=enable_review,
    )

    # mock LLMGenerator — 用调用计数器按顺序映射 stage
    mock_llm = MagicMock()
    call_counter = {"n": 0}

    def fake_generate(user_prompt, system_prompt="", config=None):
        n = call_counter["n"]
        call_counter["n"] += 1
        # 按 run() 里的调用顺序:0=eval, 1=gen, 2=review(若启用), 3=final(若启用)
        if not enable_review:
            order = ["eval", "gen"]
        else:
            order = ["eval", "gen", "review", "final"]
        stage = order[n] if n < len(order) else "extra"

        if stage == fail_at:
            return _fail_result(f"failed at {stage}")

        if stage == "eval":
            text = """【文档1】
- 相关性:高
- 证据等级:1b
- 可用性:可作为直接证据
PMID:12345
【文档2】
- 相关性:中
- 证据等级:2a
- 可用性:间接参考
PMID:67890
"""
        elif stage == "gen":
            text = "二甲双胍降低血糖 [PMID:12345]。"
        elif stage == "review":
            text = "整体评级:A。\n### 修订建议:\n- 保持现有结构"
        else:  # final
            text = "最终答案:二甲双胍 [PMID:12345]。"

        return _ok_result(text)

    mock_llm.generate.side_effect = fake_generate
    pipeline.llm = mock_llm
    return pipeline


def _sample_chunks() -> list:
    return [
        DocumentChunk(
            text="二甲双胍是治疗2型糖尿病的一线药物。",
            metadata={"pmid": "12345", "year": "1998", "journal": "Lancet", "title": "UKPDS 34"},
            relevance_score=0.92,
        ),
        DocumentChunk(
            text="SGLT2 抑制剂对心衰有益。",
            metadata={"pmid": "67890", "year": "2019", "journal": "NEJM"},
            relevance_score=0.78,
        ),
        DocumentChunk(
            text="GLP-1 激动剂可减重。",
            metadata={"pmid": "11111", "year": "2021", "journal": "NEJM"},
            relevance_score=0.65,
        ),
    ]


# ==================== 辅助方法测试 ====================

class TestHelperMethods(unittest.TestCase):

    def test_filter_by_evaluation_keeps_high_quality(self):
        chunks = _sample_chunks()
        eval_text = """【文档1】
- 相关性:高
- 证据等级:1b
- 可用性:可作为直接证据
PMID:12345
【文档2】
- 相关性:中
- 证据等级:2a
- 可用性:间接参考
PMID:67890
【文档3】
- 相关性:低
- 证据等级:3
- 可用性:不可用
PMID:11111
"""
        filtered = MedicalGenerationPipeline._filter_by_evaluation(chunks, eval_text)
        pmids = [c.metadata["pmid"] for c in filtered]
        self.assertEqual(pmids, ["12345"])

    def test_filter_by_evaluation_fallback_when_no_pmid(self):
        chunks = _sample_chunks()
        eval_text = "评估结果格式很奇怪,根本没有 PMID"
        filtered = MedicalGenerationPipeline._filter_by_evaluation(chunks, eval_text)
        # 没有 PMID 块 → fallback
        self.assertEqual(len(filtered), 3)

    def test_filter_by_evaluation_empty(self):
        chunks = _sample_chunks()
        filtered = MedicalGenerationPipeline._filter_by_evaluation(chunks, "")
        self.assertEqual(len(filtered), 3)

    def test_format_sources_dedup(self):
        chunks = [
            DocumentChunk(text="x", metadata={"pmid": "12345"}, relevance_score=0.9),
            DocumentChunk(text="x", metadata={"pmid": "12345"}, relevance_score=0.8),  # dup
            DocumentChunk(text="y", metadata={"pmid": "67890"}, relevance_score=0.7),
        ]
        sources = MedicalGenerationPipeline._format_sources(chunks)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["pmid"], "12345")

    def test_format_sources_includes_metadata(self):
        chunks = _sample_chunks()
        sources = MedicalGenerationPipeline._format_sources(chunks)
        for s in sources:
            self.assertIn("pmid", s)
            self.assertIn("relevance_score", s)
            self.assertIn("year", s)

    def test_postprocess_adds_disclaimer(self):
        ans = "二甲双胍降血糖"
        result = MedicalGenerationPipeline._postprocess(ans, _sample_chunks())
        self.assertIn("重要提示", result)
        self.assertIn("PMID:12345", result)

    def test_postprocess_includes_references(self):
        ans = "二甲双胍降血糖 [PMID:12345]"
        result = MedicalGenerationPipeline._postprocess(ans, _sample_chunks())
        self.assertIn("### 参考来源", result)
        self.assertIn("PMID:12345", result)
        self.assertIn("PMID:67890", result)

    def test_postprocess_handles_empty(self):
        result = MedicalGenerationPipeline._postprocess("", _sample_chunks())
        # 空答案也应当返回空(不应当崩)
        self.assertEqual(result, "")

    def test_postprocess_skips_when_llm_already_added(self):
        """如果 LLM 输出里已经有'参考来源'和'重要提示',不再重复加"""
        ans = (
            "二甲双胍降血糖 [PMID:12345]。\n\n"
            "### 参考来源\n- PMID:12345\n\n"
            "**重要提示**:这是一条免责声明。\n"
        )
        result = MedicalGenerationPipeline._postprocess(ans, _sample_chunks())
        # 不应当重复加"参考来源"和"重要提示"
        self.assertEqual(result.count("### 参考来源"), 1)
        self.assertEqual(result.count("重要提示"), 1)

    def test_postprocess_adds_when_llm_missing(self):
        """如果 LLM 没输出'参考来源',自动补"""
        ans = "二甲双胍降血糖 [PMID:12345]。"
        result = MedicalGenerationPipeline._postprocess(ans, _sample_chunks())
        # 应当补上
        self.assertIn("### 参考来源", result)
        self.assertIn("PMID:12345", result)
        self.assertIn("重要提示", result)

    def test_postprocess_compresses_blank_lines(self):
        ans = "段落1\n\n\n\n\n段落2"
        result = MedicalGenerationPipeline._postprocess(ans, _sample_chunks())
        # 连续多个换行应当被压缩
        self.assertNotIn("\n\n\n\n", result)


# ==================== Pipeline 主流程 ====================

class TestPipelineRun(unittest.TestCase):

    def test_run_with_review(self):
        pipeline = _make_mock_pipeline(enable_review=True)
        result = pipeline.run(
            query="二甲双胍对心血管的影响?",
            retrieved_docs=_sample_chunks(),
        )

        # 检查结构
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.query, "二甲双胍对心血管的影响?")
        self.assertIsInstance(result.answer, str)
        self.assertGreater(len(result.answer), 0)
        self.assertIn("重要提示", result.answer)

        # 阶段都被执行
        self.assertGreater(len(pipeline.llm.generate.call_args_list), 3)
        # 4 个 LLM 调用:eval / gen / review / final
        self.assertEqual(pipeline.llm.generate.call_count, 4)

    def test_run_without_review(self):
        pipeline = _make_mock_pipeline(enable_review=False)
        result = pipeline.run(
            query="Q",
            retrieved_docs=_sample_chunks(),
        )

        # 没有 review:只调 eval + gen,跳过 review 和 final,直接用草稿作最终答案
        self.assertEqual(pipeline.llm.generate.call_count, 2)
        # query 不一定出现在答案中(取决于 LLM),但草稿内容应当保留
        self.assertIn("二甲双胍降低血糖", result.answer)
        # 没有 review_feedback
        self.assertNotIn("review_feedback", result.intermediate_results)

    def test_run_handles_eval_failure(self):
        """证据评估失败时,降级到原始上下文"""
        pipeline = _make_mock_pipeline(enable_review=True, fail_at="eval")
        result = pipeline.run(query="Q", retrieved_docs=_sample_chunks())
        # 流水线仍然完成
        self.assertTrue(result.generation_metrics.stage_success["context_assembly"])
        self.assertFalse(result.generation_metrics.stage_success["evidence_evaluation"])
        # 后续步骤仍然成功
        self.assertTrue(result.generation_metrics.stage_success["answer_generation"])

    def test_run_handles_final_failure(self):
        """最终组装失败时,fallback 到草稿答案"""
        pipeline = _make_mock_pipeline(enable_review=True, fail_at="final")
        result = pipeline.run(query="Q", retrieved_docs=_sample_chunks())
        # final 失败 → fallback 到 draft_answer
        self.assertFalse(result.generation_metrics.stage_success["final_assembly"])
        # 答案应当包含草稿内容
        self.assertGreater(len(result.answer), 0)
        self.assertIn("二甲双胍", result.answer)

    def test_run_metrics_present(self):
        pipeline = _make_mock_pipeline()
        result = pipeline.run(query="Q", retrieved_docs=_sample_chunks())
        m = result.generation_metrics
        self.assertIsInstance(m, GenerationMetrics)
        # total_time_seconds 可能非常接近 0(mock 模式下),用 >= 0
        self.assertGreaterEqual(m.total_time_seconds, 0)
        # 关键字段都要在(即使部分为空 dict)
        self.assertIn("context_assembly", m.stage_times)
        self.assertIn("answer_generation", m.stage_times)
        self.assertIn("context_assembly", m.stage_success)

    def test_run_sources_dedup(self):
        pipeline = _make_mock_pipeline()
        result = pipeline.run(query="Q", retrieved_docs=_sample_chunks())
        # sources 应当去重
        pmids = [s["pmid"] for s in result.sources]
        self.assertEqual(len(pmids), len(set(pmids)))

    def test_run_intermediate_results_populated(self):
        pipeline = _make_mock_pipeline(enable_review=True)
        result = pipeline.run(query="Q", retrieved_docs=_sample_chunks())
        self.assertIn("evidence_evaluation", result.intermediate_results)
        self.assertIn("draft_answer", result.intermediate_results)
        self.assertIn("review_feedback", result.intermediate_results)

    def test_run_intermediate_no_review(self):
        pipeline = _make_mock_pipeline(enable_review=False)
        result = pipeline.run(query="Q", retrieved_docs=_sample_chunks())
        # 没有 review 就不应当有 review_feedback
        self.assertNotIn("review_feedback", result.intermediate_results)


# ==================== 配置校验 ====================

class TestPipelineConfig(unittest.TestCase):

    @patch("medical_generation_pipeline.LLMGenerator")
    def test_config_from_stage(self, mock_llm_class):
        mock_llm_class.return_value = MagicMock()
        pipeline = MedicalGenerationPipeline()
        # 评估阶段温度应当 ≤ 0.2
        self.assertLessEqual(pipeline.eval_config.temperature, 0.2)
        # 答案生成温度应当 ≥ 0.2
        self.assertGreaterEqual(pipeline.gen_config.temperature, 0.2)
        # 审查阶段温度应当 ≤ 0.2
        self.assertLessEqual(pipeline.review_config.temperature, 0.2)

    @patch("medical_generation_pipeline.LLMGenerator")
    def test_custom_configs_override(self, mock_llm_class):
        mock_llm_class.return_value = MagicMock()
        pipeline = MedicalGenerationPipeline(
            gen_config=GenerationConfig(temperature=0.9, max_tokens=2000),
        )
        self.assertEqual(pipeline.gen_config.temperature, 0.9)
        self.assertEqual(pipeline.gen_config.max_tokens, 2000)


# ==================== Metrics jsonl 持久化 ====================

class TestMetricsPersistence(unittest.TestCase):

    @patch("medical_generation_pipeline.LLMGenerator")
    def test_metrics_persisted_to_jsonl(self, mock_llm_class):
        """run() 应当把 metrics 追加到 jsonl 文件"""
        # mock LLMGenerator 避免真实连接
        mock_llm_class.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = str(Path(tmpdir) / "metrics.jsonl")

            # 直接构造 mock pipeline + 走真实 persist 逻辑
            pipeline = _make_mock_pipeline(enable_review=True)
            pipeline.metrics_path = metrics_path

            pipeline.run(query="test query 1", retrieved_docs=_sample_chunks())

            # 验证 jsonl 文件存在 + 含 1 条记录
            self.assertTrue(Path(metrics_path).exists())
            with Path(metrics_path).open("r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 1)

            record = json.loads(lines[0])
            # 关键字段都要在
            self.assertEqual(record["query"], "test query 1")
            self.assertIn("timestamp", record)
            self.assertIn("total_time_seconds", record)
            self.assertIn("stage_times", record)
            self.assertIn("stage_success", record)
            self.assertIn("token_counts", record)
            self.assertIn("sources_count", record)
            self.assertIn("answer_length", record)
            self.assertGreater(record["answer_length"], 0)

    @patch("medical_generation_pipeline.LLMGenerator")
    def test_metrics_appended_across_runs(self, mock_llm_class):
        """多次 run() 调用 → jsonl 多行(追加,不覆盖)"""
        mock_llm_class.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = str(Path(tmpdir) / "metrics.jsonl")
            pipeline = _make_mock_pipeline(enable_review=False)
            pipeline.metrics_path = metrics_path

            for q in ["q1", "q2", "q3"]:
                pipeline.run(query=q, retrieved_docs=_sample_chunks())

            with Path(metrics_path).open("r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 3)
            queries = [json.loads(l)["query"] for l in lines]
            self.assertEqual(queries, ["q1", "q2", "q3"])

    @patch("medical_generation_pipeline.LLMGenerator")
    def test_metrics_skipped_when_path_none(self, mock_llm_class):
        """metrics_path=None 时不应当报错"""
        mock_llm_class.return_value = MagicMock()
        pipeline = _make_mock_pipeline(enable_review=False)
        pipeline.metrics_path = None
        # 应当顺利跑完
        result = pipeline.run(query="x", retrieved_docs=_sample_chunks())
        self.assertIsNotNone(result.answer)

    @patch("medical_generation_pipeline.LLMGenerator")
    def test_metrics_handles_cache_stats_failure(self, mock_llm_class):
        """cache_stats 抛异常时也不应当让 persist 失败"""
        mock_llm_class.return_value = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = str(Path(tmpdir) / "metrics.jsonl")
            pipeline = _make_mock_pipeline(enable_review=False)
            # 只覆盖 cache_stats,保留 generate 的真实行为
            pipeline.llm.cache_stats = MagicMock(side_effect=Exception("boom"))
            pipeline.metrics_path = metrics_path

            # 应当顺利完成
            pipeline.run(query="x", retrieved_docs=_sample_chunks())
            self.assertTrue(Path(metrics_path).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)