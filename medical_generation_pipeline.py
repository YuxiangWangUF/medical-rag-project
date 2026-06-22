"""
Stage 8 Part 2.2: 医学生成流水线 (MedicalGenerationPipeline)

把 ContextAssembler + PromptStage + LLMGenerator 串成完整的医学问答流水线:
1. 组装上下文(检索证据 → 干净 context)
2. (可选)证据评估 — 让 LLM 判断证据质量,筛选高质量证据
3. 提取评估结果 + 生成答案草稿
4. (可选)批判性审查 — 让 LLM 检查答案是否有幻觉/错引
5. 生成最终答案 — 用审查后的反馈打磨
6. 后处理 — 加引用、加免责声明、美化格式
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from context_assembler import (
    ContextAssembler,
    DocumentChunk,
)
from llm_generator import (
    GenerationConfig,
    GenerationResult,
    LLMGenerator,
    extract_json,
)
from prompt_templates import (
    ANSWER_GENERATOR,
    CRITICAL_REVIEWER,
    EVIDENCE_EVALUATOR,
    FINAL_ASSEMBLER,
    PromptStage,
    get_full_pipeline,
)
from types_typed import CacheStats, PipelineMetricsRecord

logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

@dataclass
class GenerationMetrics:
    """生成过程度量"""
    total_time_seconds: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    token_counts: Dict[str, int] = field(default_factory=dict)
    stage_success: Dict[str, bool] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """完整流水线输出"""
    query: str
    answer: str
    context_metadata: Dict[str, Any]
    generation_metrics: GenerationMetrics
    intermediate_results: Dict[str, Any]
    sources: List[Dict[str, Any]]
    timestamp: str


# ==================== 流水线主类 ====================

class MedicalGenerationPipeline:
    """
    医学 RAG 答案生成流水线。

    使用示例:
        pipeline = MedicalGenerationPipeline(
            llm_model="qwen3:8b",
            max_context_tokens=3000,
            enable_review=True,
        )
        result = pipeline.run(
            query="二甲双胍对心血管的影响?",
            retrieved_docs=[...],   # DocumentChunk / Document / dict 均可
        )
        print(result.answer)
        print(result.sources)
    """

    def __init__(
        self,
        llm_model: str = "qwen3:8b",
        llm_base_url: str = "http://localhost:11434",
        max_context_tokens: int = 3000,
        enable_review: bool = True,
        # 每个阶段的生成配置(覆盖模板默认)
        eval_config: Optional[GenerationConfig] = None,
        gen_config: Optional[GenerationConfig] = None,
        review_config: Optional[GenerationConfig] = None,
        final_config: Optional[GenerationConfig] = None,
        # metrics 持久化路径 — None 时不持久化
        metrics_path: Optional[str] = None,
    ) -> None:
        self.llm = LLMGenerator(model_name=llm_model, base_url=llm_base_url)
        self.assembler = ContextAssembler(
            max_tokens=max_context_tokens, offline=True,
        )
        self.enable_review = bool(enable_review)
        self.metrics_path = metrics_path
        # 阶段配置 — None 时用 PromptStage 里的默认 temperature/max_tokens
        self.eval_config = eval_config or self._config_from_stage(EVIDENCE_EVALUATOR)
        self.gen_config = gen_config or self._config_from_stage(ANSWER_GENERATOR)
        self.review_config = review_config or self._config_from_stage(CRITICAL_REVIEWER)
        self.final_config = final_config or self._config_from_stage(FINAL_ASSEMBLER)

    @staticmethod
    def _config_from_stage(stage: PromptStage) -> GenerationConfig:
        """把 PromptStage 的 temperature/max_tokens 转成 GenerationConfig"""
        return GenerationConfig(
            temperature=stage.temperature,
            max_tokens=stage.max_tokens,
        )

    # ---------- 主流程 ----------

    def run(
        self,
        query: str,
        retrieved_docs: list,
    ) -> PipelineResult:
        """
        执行完整流水线。

        Args:
            query: 用户问题
            retrieved_docs: 检索结果列表(支持多种类型)

        Returns:
            PipelineResult: 包含答案、元数据、中间结果、引用来源
        """
        t_start = time.time()
        metrics = GenerationMetrics()
        intermediate: Dict[str, Any] = {}

        # 初始化所有 stage 状态 — 防止 assembler 异常时下游 get() 拿到 None
        for stage_name in (
            "context_assembly",
            "evidence_evaluation",
            "answer_generation",
            "critical_review",
            "final_assembly",
            "postprocess",
        ):
            metrics.stage_times[stage_name] = 0.0
            metrics.stage_success[stage_name] = False

        # ===== 第 1 步:上下文组装 =====
        t0 = time.time()
        context_result = self.assembler.assemble(retrieved_docs, question=query)
        metrics.stage_times["context_assembly"] = time.time() - t0
        metrics.stage_success["context_assembly"] = True
        logger.info(
            f"[1/6] 上下文组装: {context_result['metadata']['chunks_selected']} chunks, "
            f"{context_result['metadata']['estimated_tokens']} tokens"
        )
        context_text = context_result["context_text"]

        # ===== 第 2 步:证据评估(可选)=====
        evaluation_text = ""
        evaluated_chunks: List[DocumentChunk] = context_result["selected_chunks"]
        try:
            t0 = time.time()
            eval_user = EVIDENCE_EVALUATOR.render(
                context=context_text, question=query,
            )
            eval_result = self.llm.generate(
                user_prompt=eval_user,
                system_prompt=EVIDENCE_EVALUATOR.system_prompt,
                config=self.eval_config,
            )
            metrics.stage_times["evidence_evaluation"] = time.time() - t0
            metrics.token_counts["evidence_evaluation"] = (
                eval_result.prompt_tokens + eval_result.response_tokens
            )
            metrics.stage_success["evidence_evaluation"] = eval_result.success
            evaluation_text = eval_result.text
            intermediate["evidence_evaluation"] = evaluation_text
            logger.info(
                f"[2/6] 证据评估: success={eval_result.success}, "
                f"{metrics.stage_times['evidence_evaluation']:.2f}s"
            )
            # 从评估中提取可用文档 ID,筛选上下文
            if eval_result.success and evaluation_text:
                evaluated_chunks = self._filter_by_evaluation(
                    context_result["selected_chunks"], evaluation_text,
                )
                if evaluated_chunks:
                    # 用筛选后的 chunks 重建 context_text
                    context_text = self._rebuild_context_text(
                        evaluated_chunks, context_result["metadata"],
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[2/6] 证据评估异常,降级到原始上下文: {e}")
            metrics.stage_success["evidence_evaluation"] = False

        # ===== 第 3 步:生成答案草稿 =====
        t0 = time.time()
        gen_user = ANSWER_GENERATOR.render(
            context=context_text, question=query, evaluation=evaluation_text,
        )
        gen_result = self.llm.generate(
            user_prompt=gen_user,
            system_prompt=ANSWER_GENERATOR.system_prompt,
            config=self.gen_config,
        )
        metrics.stage_times["answer_generation"] = time.time() - t0
        metrics.token_counts["answer_generation"] = (
            gen_result.prompt_tokens + gen_result.response_tokens
        )
        metrics.stage_success["answer_generation"] = gen_result.success
        draft_answer = gen_result.text
        intermediate["draft_answer"] = draft_answer
        logger.info(
            f"[3/6] 答案草稿: success={gen_result.success}, "
            f"{len(draft_answer)} 字符, {metrics.stage_times['answer_generation']:.2f}s"
        )

        # ===== 第 4 步:批判性审查(可选)=====
        review_text = ""
        if self.enable_review:
            t0 = time.time()
            review_user = CRITICAL_REVIEWER.render(
                context=context_text, question=query,
                previous_answer=draft_answer,
            )
            review_result = self.llm.generate(
                user_prompt=review_user,
                system_prompt=CRITICAL_REVIEWER.system_prompt,
                config=self.review_config,
            )
            metrics.stage_times["critical_review"] = time.time() - t0
            metrics.token_counts["critical_review"] = (
                review_result.prompt_tokens + review_result.response_tokens
            )
            metrics.stage_success["critical_review"] = review_result.success
            review_text = review_result.text
            intermediate["review_feedback"] = review_text
            logger.info(
                f"[4/6] 批判审查: success={review_result.success}, "
                f"{metrics.stage_times['critical_review']:.2f}s"
            )

        # ===== 第 5 步:生成最终答案 =====
        t0 = time.time()
        # 如果审查成功 → 用审查结果润色;否则直接用草稿
        if review_text and metrics.stage_success.get("critical_review"):
            final_user = FINAL_ASSEMBLER.render(
                context=context_text, question=query,
                previous_answer=draft_answer, evaluation=review_text,
            )
            final_result = self.llm.generate(
                user_prompt=final_user,
                system_prompt=FINAL_ASSEMBLER.system_prompt,
                config=self.final_config,
            )
            # 如果 final 失败,fallback 到草稿
            if final_result.success and final_result.text:
                base_answer = final_result.text
            else:
                logger.warning("[5/6] 最终组装失败,降级使用草稿答案")
                base_answer = draft_answer
        else:
            final_result = None
            base_answer = draft_answer
        metrics.stage_times["final_assembly"] = time.time() - t0
        if final_result:
            metrics.token_counts["final_assembly"] = (
                final_result.prompt_tokens + final_result.response_tokens
            )
            metrics.stage_success["final_assembly"] = final_result.success
        else:
            metrics.stage_success["final_assembly"] = True  # 跳过阶段视为成功
        logger.info(
            f"[5/6] 最终组装: success={metrics.stage_success['final_assembly']}, "
            f"{metrics.stage_times['final_assembly']:.2f}s"
        )

        # ===== 第 6 步:后处理 — 加引用 + 免责声明 =====
        t0 = time.time()
        answer = self._postprocess(
            answer_text=base_answer,
            chunks=evaluated_chunks,
            include_disclaimer=True,
        )
        metrics.stage_times["postprocess"] = time.time() - t0
        metrics.stage_success["postprocess"] = True
        logger.info(f"[6/6] 后处理完成: {metrics.stage_times['postprocess']:.2f}s")

        # ===== 组装最终结果 =====
        metrics.total_time_seconds = time.time() - t_start

        sources = self._format_sources(evaluated_chunks)

        result = PipelineResult(
            query=query,
            answer=answer,
            context_metadata=context_result["metadata"],
            generation_metrics=metrics,
            intermediate_results=intermediate,
            sources=sources,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # metrics 持久化(jsonl 追加)
        if self.metrics_path:
            self._persist_metrics(result)

        return result

    # ---------- 辅助方法 ----------

    def _persist_metrics(self, result: PipelineResult) -> None:
        """
        把一次流水线运行的 metrics 追加到 jsonl 文件。

        格式:每行一个 JSON,字段:
        - timestamp: 跑流水线的时刻
        - query: 用户问题
        - total_time_seconds: 总耗时
        - stage_times: 各阶段耗时
        - stage_success: 各阶段是否成功
        - token_counts: 各阶段 token 用量
        - sources_count: 引用条数
        - answer_length: 答案长度
        - llm_cache_stats: LLM 缓存命中率/容量
        """
        if not self.metrics_path:
            return
        path = Path(self.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 收集缓存统计(如果 LLMGenerator 暴露了的话,且是 dict 才行)
        cache_stats: CacheStats = CacheStats(size=0, max_size=0, utilization=0.0)
        if hasattr(self.llm, "cache_stats"):
            try:
                stats = self.llm.cache_stats()
                if isinstance(stats, dict):
                    # 安全地构造 TypedDict(缺字段填默认值)
                    cache_stats = CacheStats(
                        size=int(stats.get("size", 0)),
                        max_size=int(stats.get("max_size", 0)),
                        utilization=float(stats.get("utilization", 0.0)),
                    )
            except Exception:  # noqa: BLE001
                pass

        record = PipelineMetricsRecord(
            timestamp=result.timestamp,
            query=result.query,
            total_time_seconds=round(
                result.generation_metrics.total_time_seconds, 4
            ),
            stage_times={
                k: round(v, 4)
                for k, v in result.generation_metrics.stage_times.items()
            },
            stage_success=result.generation_metrics.stage_success,
            token_counts=result.generation_metrics.token_counts,
            sources_count=len(result.sources),
            answer_length=len(result.answer),
            llm_cache_stats=cache_stats,
        )
        # 用 default=str 把不可序列化的值转字符串,避免单条脏数据让整个写入挂掉
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning(f"metrics 写入失败 ({path}): {e}")

    @staticmethod
    def _filter_by_evaluation(
        chunks: List[DocumentChunk], evaluation_text: str,
    ) -> List[DocumentChunk]:
        """
        从评估结果中提取"可用"的文档 ID,返回对应 chunks。

        严格策略 — 只匹配 "PMID:数字" 格式,避免误把"2016年"当 PMID:
        - 段落边界:连续两个换行 或 整段 PMID 块
        - PMID 格式:PMID:数字(1-8 位),或 pmid=数字
        - 同一段落只统计其 PMID 附近的正负关键词
        - 正面关键词:直接证据 / 1a / 1b / 明确支持 / 关键发现
        - 负面关键词:间接参考 / 不可用 / 局限性 / 无依据
        """
        if not evaluation_text:
            return chunks

        # 1. 按段落切分
        doc_blocks = re.split(r"(?=【文档\d+】)", evaluation_text)
        if len(doc_blocks) <= 1:
            doc_blocks = re.split(r"\n\s*\n", evaluation_text)

        # 2. 严格的 PMID 格式匹配(避免误匹配年份)
        # 匹配 PMID:12345 / pmid=12345 / PMID 12345 / (PMID: 12345) 等
        pmid_strict = re.compile(r"PMID[:\s=]+(\d{4,9})", re.IGNORECASE)
        positive_keywords = ["直接证据", "1a", "1b", "明确支持", "关键"]
        negative_keywords = ["间接参考", "不可用", "局限性", "无依据"]
        # 注意:把"高"从正面列表移除 — 容易误匹配"高中""高级"

        positive_pmids: set = set()
        for block in doc_blocks:
            pmids_in_block = pmid_strict.findall(block)
            if not pmids_in_block:
                continue
            pos = sum(block.count(kw) for kw in positive_keywords)
            neg = sum(block.count(kw) for kw in negative_keywords)
            if pos > neg:
                positive_pmids.update(pmids_in_block)

        # 3. 如果识别不出来,fallback 到原列表
        if not positive_pmids:
            return chunks

        # 4. 筛选 — PMID 是字符串
        filtered = [
            c for c in chunks
            if str(c.metadata.get("pmid", "")) in positive_pmids
        ]
        # 保底
        return filtered if filtered else chunks

    @staticmethod
    def _rebuild_context_text(
        chunks: List[DocumentChunk], original_meta: Dict[str, Any],
    ) -> str:
        """从筛选后的 chunks 重建 context_text(同 ContextAssembler 内部格式)"""
        parts: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            header = (
                f"【文档{i}】"
                f"PMID: {chunk.metadata.get('pmid', '?')} | "
                f"来源: {chunk.source or 'unknown'} | "
                f"相关性: {chunk.relevance_score:.3f}"
            )
            parts.append(f"{header}\n{chunk.text.strip()}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_sources(chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """把 chunks 格式化成 sources 列表(给前端 / 日志用)"""
        seen = set()
        sources = []
        for c in chunks:
            pmid = str(c.metadata.get("pmid", ""))
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            sources.append({
                "pmid": pmid,
                "source": c.source or "unknown",
                "title": c.metadata.get("title", ""),
                "year": c.metadata.get("year", ""),
                "journal": c.metadata.get("journal", ""),
                "relevance_score": round(c.relevance_score, 3),
            })
        return sources

    @staticmethod
    def _postprocess(
        answer_text: str,
        chunks: List[DocumentChunk],
        include_disclaimer: bool = True,
    ) -> str:
        """
        后处理:
        - 去掉多余空行
        - 如果 LLM 没生成"参考来源"块 → 自动补
        - 如果 LLM 没生成"重要提示" → 自动加免责声明
        """
        if not answer_text:
            return ""

        text = re.sub(r"\n{3,}", "\n\n", answer_text).strip()

        # 检测 LLM 输出里是否已经有"参考来源"和"重要提示"
        has_references = bool(re.search(r"###\s*参考来源", text))
        has_disclaimer = ("重要提示" in text) or ("免责声明" in text)

        # 只在缺失时补
        if not has_references:
            sources = MedicalGenerationPipeline._format_sources(chunks)
            if sources:
                text += "\n\n### 参考来源\n"
                for s in sources[:10]:
                    pmid = s["pmid"]
                    title = s.get("title") or "医学文献"
                    year = s.get("year", "")
                    journal = s.get("journal", "")
                    bits = [f"- PMID:{pmid}"]
                    if title and title != "医学文献":
                        bits.append(f"《{title}》")
                    if year:
                        bits.append(f"({year})")
                    if journal:
                        bits.append(f"· {journal}")
                    text += " ".join(bits) + "\n"

        if include_disclaimer and not has_disclaimer:
            text += (
                "\n---\n"
                "**重要提示**:本回答由 AI 辅助生成,内容基于已发表的医学文献,"
                "**仅供学术参考**,不构成任何医疗建议。"
                "如需临床决策,请咨询专业医生。\n"
            )

        return text


# ==================== 便捷函数 ====================

def quick_generate(
    query: str,
    retrieved_docs: list,
    llm_model: str = "qwen3:8b",
    enable_review: bool = True,
) -> PipelineResult:
    """一行调用,适合脚本/批处理。"""
    pipeline = MedicalGenerationPipeline(
        llm_model=llm_model, enable_review=enable_review,
    )
    return pipeline.run(query, retrieved_docs)


# ==================== 自测 ====================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=== MedicalGenerationPipeline 自测 ===\n")

    # 内联 mock 数据,不依赖外部 demo 文件,避免生产代码意外 import demo 数据
    from context_assembler import DocumentChunk

    mock_docs = [
        DocumentChunk(
            text="二甲双胍是治疗2型糖尿病的一线药物。UKPDS 34 研究显示,"
                 "在超重 2 型糖尿病患者中,二甲双胍可降低心肌梗死风险达 39% (p=0.01)。",
            metadata={"pmid": "12345", "year": "1998", "journal": "Lancet"},
            relevance_score=0.92,
        ),
        DocumentChunk(
            text="二甲双胍心血管获益的机制可能涉及改善胰岛素抵抗、降低体重。"
                 "CAMERA 研究在非糖尿病心血管患者中未观察到显著获益。",
            metadata={"pmid": "12345", "year": "2016", "journal": "Diabetes Care"},
            relevance_score=0.85,
        ),
        DocumentChunk(
            text="SGLT2 抑制剂(恩格列净、达格列净)在 EMPA-REG 中显示对心衰患者有显著获益,"
                 "全因死亡率下降约 13%。",
            metadata={"pmid": "67890", "year": "2019", "journal": "NEJM"},
            relevance_score=0.78,
        ),
    ]

    pipeline = MedicalGenerationPipeline(
        llm_model="qwen3:8b",
        enable_review=True,
    )

    result = pipeline.run(
        query="二甲双胍对心血管疾病有什么影响?",
        retrieved_docs=mock_docs,
    )

    print(f"\n{'='*60}")
    print(f"问题: {result.query}")
    print(f"耗时: {result.generation_metrics.total_time_seconds:.2f}s")
    print(f"Token: {result.generation_metrics.token_counts}")
    print(f"阶段: {result.generation_metrics.stage_success}")
    print(f"引用 ({len(result.sources)} 条):")
    for s in result.sources:
        print(f"  - PMID:{s['pmid']} relevance={s['relevance_score']}")
    print(f"\n{'='*60}")
    print("【最终答案】")
    print(result.answer)