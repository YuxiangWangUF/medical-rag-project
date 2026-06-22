"""
pytest 共享 fixtures — 减少各测试文件的样板代码。

提供:
- sample_chunks / sample_documents:常用的 DocumentChunk 样本
- sample_query: 示例医学问题
- mock_pipeline: 一个 mock 掉 LLMGenerator 的 pipeline
- mock_llm_response: 快速构造一个 GenerationResult
- temp_metrics_file: 临时 jsonl metrics 文件
- mock_ollama_tags: 模拟 Ollama /api/tags 响应
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock

import pytest

from context_assembler import DocumentChunk
from llm_generator import GenerationConfig, GenerationResult
from medical_generation_pipeline import MedicalGenerationPipeline


# ==================== Sample data ====================

@pytest.fixture
def sample_query() -> str:
    """一个常用的示例医学问题"""
    return "二甲双胍对心血管疾病有什么影响?"


@pytest.fixture
def sample_chunks() -> List[DocumentChunk]:
    """3 个不同 PMID 的样本 chunks"""
    return [
        DocumentChunk(
            text="二甲双胍是治疗2型糖尿病的一线药物。UKPDS 34 研究显示,"
                 "在超重 2 型糖尿病患者中,二甲双胍可降低心肌梗死风险达 39% (p=0.01)。",
            metadata={
                "pmid": "12345", "year": "1998",
                "journal": "Lancet", "title": "UKPDS 34",
            },
            relevance_score=0.92,
        ),
        DocumentChunk(
            text="SGLT2 抑制剂(恩格列净、达格列净)在 EMPA-REG 中显示对心衰患者有显著获益,"
                 "全因死亡率下降约 13%。",
            metadata={
                "pmid": "67890", "year": "2019",
                "journal": "NEJM", "title": "EMPA-REG",
            },
            relevance_score=0.78,
        ),
        DocumentChunk(
            text="GLP-1 激动剂可显著减重,LEADER 试验显示利拉鲁肽降低心血管死亡风险。",
            metadata={
                "pmid": "11111", "year": "2021",
                "journal": "NEJM", "title": "LEADER",
            },
            relevance_score=0.65,
        ),
    ]


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """原始文档样本(供 chunk_processor / vector_indexer 测试用)"""
    return [
        {
            "id": "doc1",
            "title": "二甲双胍心血管获益研究",
            "abstract": "本研究评估了二甲双胍对 2 型糖尿病患者心血管预后的影响。",
            "body_text": "本研究纳入 5000 名患者,平均随访 10 年。结果显示..." * 10,
            "metadata": {"pmid": "12345", "year": "1998"},
        },
        {
            "id": "doc2",
            "title": "SGLT2 抑制剂心衰研究",
            "abstract": "EMPA-REG 试验评估恩格列净对心血管结局的影响。",
            "body_text": "EMPA-REG OUTCOME 试验纳入 7020 名患者..." * 10,
            "metadata": {"pmid": "67890", "year": "2019"},
        },
    ]


# ==================== LLM Mock helpers ====================

@pytest.fixture
def mock_llm_response():
    """
    工厂 fixture: 接受 text,生成一个 GenerationResult。

    用法:
        def test_xxx(mock_llm_response):
            result = mock_llm_response("hello", success=True)
    """
    def factory(
        text: str = "mock response",
        success: bool = True,
        prompt_tokens: int = 50,
        response_tokens: int = 100,
        retries: int = 0,
        error: str = "",
    ) -> GenerationResult:
        return GenerationResult(
            text=text,
            parsed_json=None,
            raw_response={"message": {"content": text}},
            elapsed_ms=100.0,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            retries=retries,
            success=success,
            error=error,
        )
    return factory


@pytest.fixture
def mock_ollama_tags() -> Dict[str, Any]:
    """Ollama /api/tags 端点的标准 mock 响应"""
    return {
        "models": [
            {"name": "qwen3:8b"},
            {"name": "bge-m3:latest"},
        ]
    }


@pytest.fixture
def patched_ollama(mock_ollama_tags):
    """
    Patch 掉 llm_generator 里的 requests.get/post — 默认连通性 OK。

    适用:大多数 LLMGenerator 测试只需要"不连真 Ollama + 返回固定结构"。
    """
    from unittest.mock import patch

    def make_response(payload: Dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()
        return resp

    with patch("llm_generator.requests.get", return_value=make_response(mock_ollama_tags)):
        # 默认 POST 应当由各测试自行 patch(post response 内容随测试不同)
        yield {"get_response": make_response}


# ==================== Pipeline fixtures ====================

@pytest.fixture
def mock_pipeline():
    """
    构造一个 mock 掉 LLMGenerator 的 MedicalGenerationPipeline。

    默认 enable_review=True,所有 stage 返回 OK + 固定文本。
    """
    pipeline = MedicalGenerationPipeline(
        llm_model="qwen3:8b",
        enable_review=True,
    )
    mock_llm = MagicMock()

    call_counter = {"n": 0}

    def fake_generate(user_prompt: str = "", system_prompt: str = "", config=None):
        n = call_counter["n"]
        call_counter["n"] += 1
        # 0=eval, 1=gen, 2=review, 3=final
        order = ["eval", "gen", "review", "final"]
        stage = order[n] if n < len(order) else "extra"
        texts = {
            "eval": (
                "【文档1】\n- 相关性:高\n- 证据等级:1b\n"
                "- 可用性:可作为直接证据\nPMID:12345\n"
            ),
            "gen": "二甲双胍降低血糖 [PMID:12345]。",
            "review": "整体评级:A。\n### 修订建议:\n- 保持现有结构",
            "final": "最终答案:二甲双胍 [PMID:12345]。",
        }
        return GenerationResult(text=texts.get(stage, "mock"), success=True)

    mock_llm.generate.side_effect = fake_generate
    pipeline.llm = mock_llm
    return pipeline


# ==================== File system fixtures ====================

@pytest.fixture
def temp_metrics_file() -> Iterator[Path]:
    """临时 metrics jsonl 文件 — 测试完自动清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metrics.jsonl"
        yield path


@pytest.fixture
def temp_cache_file() -> Iterator[Path]:
    """临时 LLM 缓存文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "llm_cache.jsonl"
        yield path


# ==================== Auto-applied fixtures ====================

@pytest.fixture(autouse=False)
def deterministic_config() -> GenerationConfig:
    """确保 LLM 生成参数一致(给某些需要稳定输出的测试用)"""
    return GenerationConfig(
        temperature=0.0,  # 0 = 完全确定(模型支持时)
        max_tokens=512,
        json_mode=False,
    )