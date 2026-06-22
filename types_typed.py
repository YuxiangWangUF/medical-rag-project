"""
跨模块共享的 TypedDict 类型定义。

为什么用 TypedDict 而不是 dataclass:
- 这些 dict 会落 json(jsonl 持久化 / API 返回),dataclass 要多一步转换
- TypedDict 在运行时就是普通 dict,零开销
- 静态类型检查(mypy / pyright)能用,IDE 跳转 / 补全都正常
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


# ==================== LLMGenerator ====================

class CacheStats(TypedDict):
    """LLMGenerator.cache_stats() 的返回结构"""
    size: int          # 当前缓存条数
    max_size: int      # 容量上限
    utilization: float  # 占用率 0~1


class GenerationConfigDict(TypedDict, total=False):
    """GenerationConfig 的 dict 视图(用于 JSON 持久化)"""
    temperature: float
    max_tokens: int
    top_p: float
    repeat_penalty: float
    json_mode: bool
    max_retries: int
    retry_delay: float


# ==================== ContextAssembler ====================

class SelectedChunkInfo(TypedDict):
    """被选中进入上下文的 chunk 简版信息"""
    text: str
    pmid: str
    source: str
    relevance_score: float


class ContextAssemblyMetadata(TypedDict):
    """assemble() 的元数据"""
    chunks_total: int          # 输入总 chunks
    chunks_selected: int       # 选中的 chunks
    estimated_tokens: int      # 估算 token 数
    truncated: bool            # 是否触发了截断


class ContextAssemblyResult(TypedDict):
    """context_assembler.assemble() 的完整返回"""
    context_text: str
    selected_chunks: List[Any]   # List[DocumentChunk]
    metadata: ContextAssemblyMetadata


# ==================== Pipeline metrics ====================

class StageTimings(TypedDict):
    """每个阶段的耗时(秒)"""
    context_assembly: float
    evidence_evaluation: float
    answer_generation: float
    critical_review: float
    final_assembly: float
    postprocess: float


class StageSuccess(TypedDict):
    """每个阶段是否成功"""
    context_assembly: bool
    evidence_evaluation: bool
    answer_generation: bool
    critical_review: bool
    final_assembly: bool
    postprocess: bool


class PipelineMetricsRecord(TypedDict, total=False):
    """metrics_path jsonl 中每行的结构"""
    timestamp: str
    query: str
    total_time_seconds: float
    stage_times: Dict[str, float]
    stage_success: Dict[str, bool]
    token_counts: Dict[str, int]
    sources_count: int
    answer_length: int
    llm_cache_stats: CacheStats


# ==================== Retrieval result ====================

class RetrievalHit(TypedDict):
    """单条检索结果"""
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]


class RetrievalResult(TypedDict):
    """完整检索结果"""
    query: str
    hits: List[RetrievalHit]
    total: int
    elapsed_ms: float


# ==================== Quality check ====================

class QualityIssue(TypedDict, total=False):
    """单条质量问题"""
    level: str        # "warning" / "error" / "info"
    field: str        # 哪个字段出问题
    message: str      # 具体说明
    count: int        # 受影响的样本数


class QualityReport(TypedDict):
    """quality_check() 的返回结构"""
    total: int
    passed: int
    failed: int
    issues: List[QualityIssue]
    summary: str