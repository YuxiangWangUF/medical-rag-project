"""
测试 conftest 自身 — 验证 fixtures 可用 + 基本类型正确。
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from context_assembler import DocumentChunk
from llm_generator import GenerationConfig, GenerationResult
from medical_generation_pipeline import MedicalGenerationPipeline


def test_sample_query_fixture(sample_query):
    assert isinstance(sample_query, str)
    assert "二甲双胍" in sample_query


def test_sample_chunks_fixture(sample_chunks):
    assert len(sample_chunks) == 3
    assert all(isinstance(c, DocumentChunk) for c in sample_chunks)
    pmids = [c.metadata["pmid"] for c in sample_chunks]
    assert "12345" in pmids


def test_sample_documents_fixture(sample_documents):
    assert len(sample_documents) == 2
    assert all("title" in d for d in sample_documents)


def test_mock_llm_response_factory(mock_llm_response):
    """工厂应当能构造各种状态的 GenerationResult"""
    ok = mock_llm_response("hello")
    assert isinstance(ok, GenerationResult)
    assert ok.success is True
    assert ok.text == "hello"

    fail = mock_llm_response("x", success=False, error="boom")
    assert fail.success is False
    assert "boom" in fail.error


def test_mock_ollama_tags_fixture(mock_ollama_tags):
    assert "models" in mock_ollama_tags
    assert any("qwen3" in m["name"] for m in mock_ollama_tags["models"])


def test_mock_pipeline_runs(sample_query, sample_chunks, mock_pipeline):
    """mock_pipeline 应当能完整跑完一遍"""
    result = mock_pipeline.run(query=sample_query, retrieved_docs=sample_chunks)
    assert result.answer
    assert "重要提示" in result.answer


def test_temp_metrics_file_writable(temp_metrics_file):
    """临时文件 fixture 应当可用 + 可写"""
    assert isinstance(temp_metrics_file, Path)
    temp_metrics_file.write_text('{"x": 1}\n', encoding="utf-8")
    assert temp_metrics_file.exists()
    assert temp_metrics_file.stat().st_size > 0


def test_temp_cache_file(temp_cache_file):
    assert isinstance(temp_cache_file, Path)
    assert temp_cache_file.parent.exists()


def test_deterministic_config_fixture(deterministic_config):
    """温度 0 让输出更可复现(如果模型支持)"""
    assert isinstance(deterministic_config, GenerationConfig)
    assert deterministic_config.temperature == 0.0


def test_patched_ollama_doesnt_call_real_ollama(patched_ollama, mock_ollama_tags):
    """patch 应当让 LLMGenerator 不再连真实 Ollama"""
    from llm_generator import LLMGenerator
    gen = LLMGenerator(model_name="qwen3:8b")
    # 不抛 ConnectionError → 说明 GET 已经被 patch 掉
    assert gen.model_name == "qwen3:8b"