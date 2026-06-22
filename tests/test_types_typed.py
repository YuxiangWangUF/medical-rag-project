"""
测试 types_typed.py — TypedDict 定义和实际使用。
"""

import unittest
from typing import get_type_hints

from llm_generator import LLMGenerator
from types_typed import (
    CacheStats,
    ContextAssemblyMetadata,
    ContextAssemblyResult,
    GenerationConfigDict,
    PipelineMetricsRecord,
    QualityIssue,
    QualityReport,
    RetrievalHit,
    RetrievalResult,
    SelectedChunkInfo,
    StageSuccess,
    StageTimings,
)


class TestTypedDictDefinitions(unittest.TestCase):
    """TypedDict 类应当存在 + 字段齐全"""

    def test_cache_stats_has_required_fields(self):
        # TypedDict 字段可以通过 __annotations__ 拿到
        hints = get_type_hints(CacheStats)
        self.assertIn("size", hints)
        self.assertIn("max_size", hints)
        self.assertIn("utilization", hints)
        self.assertEqual(hints["size"], int)
        self.assertEqual(hints["max_size"], int)
        self.assertEqual(hints["utilization"], float)

    def test_pipeline_metrics_record_has_all_fields(self):
        hints = get_type_hints(PipelineMetricsRecord)
        for field in (
            "timestamp", "query", "total_time_seconds", "stage_times",
            "stage_success", "token_counts", "sources_count",
            "answer_length", "llm_cache_stats",
        ):
            self.assertIn(field, hints, f"missing field: {field}")

    def test_stage_timings_has_all_stages(self):
        hints = get_type_hints(StageTimings)
        for stage in (
            "context_assembly", "evidence_evaluation", "answer_generation",
            "critical_review", "final_assembly", "postprocess",
        ):
            self.assertIn(stage, hints, f"missing stage: {stage}")

    def test_quality_report_fields(self):
        hints = get_type_hints(QualityReport)
        for field in ("total", "passed", "failed", "issues", "summary"):
            self.assertIn(field, hints)

    def test_retrieval_result_structure(self):
        hints = get_type_hints(RetrievalResult)
        self.assertIn("hits", hints)
        self.assertIn("query", hints)
        self.assertIn("elapsed_ms", hints)


class TestTypedDictRuntimeBehavior(unittest.TestCase):
    """TypedDict 在运行时就是普通 dict(零开销)"""

    def test_cache_stats_is_dict_at_runtime(self):
        stats = CacheStats(size=5, max_size=32, utilization=0.156)
        # TypedDict 实例就是普通 dict
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["size"], 5)
        self.assertEqual(stats["utilization"], 0.156)

    def test_can_be_json_serialized(self):
        """TypedDict 实例可以直接 json.dumps"""
        import json
        record = PipelineMetricsRecord(
            timestamp="2026-06-23",
            query="test",
            total_time_seconds=1.23,
            stage_times={"context_assembly": 0.1, "answer_generation": 1.0},
            stage_success={"context_assembly": True, "answer_generation": True},
            token_counts={"context_assembly": 100, "answer_generation": 200},
            sources_count=3,
            answer_length=500,
            llm_cache_stats=CacheStats(size=0, max_size=32, utilization=0.0),
        )
        s = json.dumps(record, ensure_ascii=False)
        # 应当能解析回来
        parsed = json.loads(s)
        self.assertEqual(parsed["query"], "test")
        self.assertEqual(parsed["sources_count"], 3)


class TestTypedDictUsageInLLM(unittest.TestCase):
    """LLMGenerator.cache_stats() 应当返回 TypedDict 实例"""

    def test_cache_stats_returns_typed_dict(self):
        from unittest.mock import patch

        with patch("llm_generator.requests.get") as mock_get:
            mock_get.return_value.json = lambda: {"models": [{"name": "qwen3:8b"}]}
            mock_get.return_value.raise_for_status = lambda: None

            gen = LLMGenerator()
            stats = gen.cache_stats()

        # 应当是 CacheStats(dict 实例)
        self.assertIsInstance(stats, dict)
        self.assertIn("size", stats)
        self.assertIn("max_size", stats)
        self.assertIn("utilization", stats)
        # 类型应当匹配 TypedDict 声明
        self.assertIsInstance(stats["size"], int)
        self.assertIsInstance(stats["max_size"], int)
        self.assertIsInstance(stats["utilization"], float)


if __name__ == "__main__":
    unittest.main(verbosity=2)