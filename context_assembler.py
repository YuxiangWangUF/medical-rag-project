"""
Stage 8 Part 1: 上下文组装器 (Context Assembler)

负责把检索器返回的多个文档块(DocumentChunk / Document / dict)整合成一段
适合塞进 LLM prompt 的纯净上下文字符串,同时:

1. 用 Jaccard 相似度去重(避免同一文献被切分的相邻 chunk 重复占用上下文)
2. 按相关性排序,优先选择高相关性文档
3. 多样化惩罚:同一来源的文档过多时降低优先级
4. 完整段落截断:在句号处截断,避免把句子腰斩
5. 输出元数据(token 数、来源分布、选了多少 chunk)供后续 stage 评估
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

# LangChain Document 是可选依赖,运行时动态判断
try:
    from langchain_core.documents import Document as LCDocument
except ImportError:  # pragma: no cover - 项目里没装也不影响
    LCDocument = None  # type: ignore[assignment]


# ==================== 数据类 ====================

@dataclass
class DocumentChunk:
    """统一的文档块数据类 — 检索模块 / 组装模块之间的标准接口"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    source: str = ""
    chunk_id: str = ""

    def __post_init__(self) -> None:
        # 自动从 metadata 补 source / chunk_id
        if not self.source:
            self.source = str(
                self.metadata.get("source")
                or self.metadata.get("pmid")
                or self.metadata.get("doc_id")
                or "unknown"
            )
        if not self.chunk_id:
            pmid = self.metadata.get("pmid", "")
            idx = self.metadata.get("chunk_index", "")
            self.chunk_id = f"{pmid}#{idx}" if pmid and idx != "" else self.source


# 接受三种输入类型:LCDocument / DocumentChunk / dict
InputDoc = Union["LCDocument", DocumentChunk, Dict[str, Any]]


# ==================== 上下文组装器 ====================

class ContextAssembler:
    """
    把检索结果整理成 LLM prompt 用的上下文。

    使用示例:
        assembler = ContextAssembler(max_tokens=3000, tokenizer_name="gpt2")
        result = assembler.assemble(retrieved_docs, question="二甲双胍副作用")
        # result["context_text"] 直接喂给 prompt
        # result["metadata"] 记录在日志里
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        tokenizer_name: Optional[str] = None,
        dedup_threshold: float = 0.7,
        diversity_penalty: float = 0.15,
        max_per_source: int = 3,
        offline: bool = True,
    ) -> None:
        """
        Args:
            max_tokens: 上下文目标 token 上限。实际组装会保证 ≤ 上限。
            tokenizer_name: HuggingFace tokenizer 名字;加载失败时回退到字符估算法。
                传 None / 留空 + offline=True 直接用字符估算,避免在离线环境反复重试。
            dedup_threshold: Jaccard 相似度阈值,超过视为重复。
            diversity_penalty: 同一来源每多出现一次,分数乘以 (1 - penalty)。
            max_per_source: 同一来源最多保留多少 chunk(硬上限)。
            offline: 默认 True — 不联网拉 tokenizer,直接用字符估算。
        """
        self.max_tokens = int(max_tokens)
        self.tokenizer_name = tokenizer_name
        self.dedup_threshold = float(dedup_threshold)
        self.diversity_penalty = float(diversity_penalty)
        self.max_per_source = int(max_per_source)
        self.offline = bool(offline)

        self._tokenizer = None
        if tokenizer_name and not offline:
            self._load_tokenizer()

    # ---------- tokenizer ----------

    def _load_tokenizer(self) -> None:
        """加载 HuggingFace tokenizer;失败时不影响主流程,回退到按字符估算。"""
        try:
            from transformers import AutoTokenizer  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception as e:  # noqa: BLE001
            print(f"[ContextAssembler] tokenizer '{self.tokenizer_name}' 加载失败({e}),"
                  f"回退到字符估算法 (1 token ≈ 1.5 字符)")
            self._tokenizer = None

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数。优先用真实 tokenizer,失败时按字符估算。"""
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                # transformers tokenizer 通常返回 list[int]
                ids = self._tokenizer.encode(text, add_special_tokens=False)
                return len(ids)
            except Exception:  # noqa: BLE001
                pass
        # 中文 + 英文混排的经验值:中文 1 字 ≈ 1 token,英文 4 字符 ≈ 1 token。
        # 这里用混合估算:中文按字数,其他按 4 字符/token。
        chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return chinese_chars + max(1, other_chars // 4)

    # ---------- 标准化输入 ----------

    @staticmethod
    def _normalize(doc: InputDoc) -> DocumentChunk:
        """把 LCDocument / dict 标准化成 DocumentChunk。"""
        if isinstance(doc, DocumentChunk):
            return doc

        if LCDocument is not None and isinstance(doc, LCDocument):
            return DocumentChunk(
                text=doc.page_content or "",
                metadata=dict(doc.metadata or {}),
                relevance_score=float(doc.metadata.get("relevance_score", 0.0))
                if isinstance(doc.metadata, dict) else 0.0,
            )

        if isinstance(doc, dict):
            text = doc.get("text") or doc.get("page_content") or ""
            metadata = dict(doc.get("metadata") or {})
            return DocumentChunk(
                text=str(text),
                metadata=metadata,
                relevance_score=float(doc.get("relevance_score", 0.0)),
                source=str(doc.get("source", "")),
                chunk_id=str(doc.get("chunk_id", "")),
            )

        raise TypeError(f"不支持的文档类型: {type(doc).__name__}")

    # ---------- Jaccard 相似度 ----------

    @staticmethod
    def _tokenize_for_jaccard(text: str) -> set:
        """把文本切成 token 集合 — 用来算 Jaccard 相似度。
        用 jieba 切中文,英文按词切,降低复杂度。"""
        if not text:
            return set()
        try:
            import jieba  # type: ignore
            tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
        except ImportError:
            # 没装 jieba 时,中文按 2-gram 切,英文按空格
            tokens = re.findall(r"[\u4e00-\u9fff]{2}|[A-Za-z]+|\d+", text)
        return set(tokens)

    @staticmethod
    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _deduplicate(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """用 Jaccard 去重:相似度 >= 阈值时,保留相关性更高的那个。"""
        # 按 relevance_score 降序排,优先保留高分
        ordered = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        kept: List[DocumentChunk] = []
        kept_tokens: List[set] = []
        for chunk in ordered:
            tokens = self._tokenize_for_jaccard(chunk.text)
            is_dup = False
            for kept_chunk, kept_tok in zip(kept, kept_tokens):
                if self.jaccard(tokens, kept_tok) >= self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(chunk)
                kept_tokens.append(tokens)
        return kept

    # ---------- 多样化排序 ----------

    def _diversity_rerank(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """在相关性基础上,给同一来源的 chunk 逐步打折,并应用 max_per_source 硬上限。"""
        source_count: Counter = Counter()
        scored: List[Tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            src = chunk.source or "unknown"
            cnt = source_count[src]
            if cnt >= self.max_per_source:
                # 直接 0 分,排序时会被淘汰
                adjusted = 0.0
            else:
                adjusted = chunk.relevance_score * ((1 - self.diversity_penalty) ** cnt)
            source_count[src] += 1
            scored.append((adjusted, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    # ---------- 截断到句号 ----------

    @staticmethod
    def _truncate_at_sentence(text: str, target_chars: int) -> str:
        """把文本截到 target_chars 长度,优先在"后 10%"窗口内找句号截断。

        "后 10%" 指 target_chars 的后 10% 字符窗口;在这个窗口里找句末标点,
        找到则在标点处截断(避免把句子腰斩),找不到则硬截到 target_chars。

        例子:
          - target_chars=100, 窗口 = 100 // 10 = 10 字符
          - 候选区域 = text[90:100]
          - 在候选区域里 rfind 标点 → 找到则截到该标点之后;否则硬截 text[:100]
          - target_chars=20, 窗口 = max(1, 20 // 10) = 2 字符(防 0 窗口)
        """
        if len(text) <= target_chars:
            return text

        # 后 10% 窗口(至少 1 字符,防止 target_chars < 10 时窗口为 0)
        window = max(1, target_chars // 10)
        candidate_region = text[target_chars - window:target_chars]

        # 标点按"优先级"排序(更长的优先,避免 "." 匹到 "。" 之前的位置)
        for punct in ["。", "？", "！", ";\n", "!\n", "?\n", ".\n", ". ", "! ", "? "]:
            idx = candidate_region.rfind(punct)
            if idx >= 0:
                return text[: target_chars - window + idx + len(punct)]
        # 没找到合适句号,硬截
        return text[:target_chars].rstrip()

    # ---------- 来源分析 ----------

    @staticmethod
    def _analyze_sources(chunks: List[DocumentChunk]) -> Dict[str, int]:
        """统计每个来源贡献了多少 chunk。"""
        counter: Counter = Counter()
        for c in chunks:
            counter[c.source or "unknown"] += 1
        return dict(counter)

    # ---------- 主流程 ----------

    def assemble(
        self,
        retrieved_docs: List[InputDoc],
        question: str = "",
    ) -> Dict[str, Any]:
        """
        把检索结果组装成 LLM 上下文。

        Args:
            retrieved_docs: 来自检索器的列表(支持 LCDocument / DocumentChunk / dict 混合)
            question: 用户问题(预留,后续可用于 query-aware rerank)

        Returns:
            {
              "context_text": str,
              "metadata": {
                "total_chunks_retrieved": int,
                "unique_chunks_after_dedup": int,
                "chunks_selected": int,
                "estimated_tokens": int,
                "chunk_sources": Dict[str, int],
              },
              "selected_chunks": List[DocumentChunk]
            }
        """
        # 1. 标准化
        normalized = [self._normalize(d) for d in retrieved_docs if d is not None]
        total_retrieved = len(normalized)

        # 2. 去重
        unique_chunks = self._deduplicate(normalized)

        # 3. 多样化排序
        ranked = self._diversity_rerank(unique_chunks)

        # 4. 按 token 预算贪心选
        selected: List[DocumentChunk] = []
        used_tokens = 0
        for chunk in ranked:
            chunk_tokens = self.estimate_tokens(chunk.text)
            if used_tokens + chunk_tokens > self.max_tokens:
                continue
            selected.append(chunk)
            used_tokens += chunk_tokens
            if used_tokens >= self.max_tokens:
                break

        # 5. 构建上下文字符串(按相关性降序)
        parts: List[str] = []
        for i, chunk in enumerate(selected, 1):
            header = (
                f"【文档{i}】"
                f"PMID: {chunk.metadata.get('pmid', '?')} | "
                f"来源: {chunk.source or 'unknown'} | "
                f"相关性: {chunk.relevance_score:.3f}"
            )
            parts.append(f"{header}\n{chunk.text.strip()}")

        context_text = "\n\n".join(parts)

        # 6. 完整段落截断兜底(防 estimate_tokens 误差)
        max_chars = self.max_tokens * 4  # 中文为主的兜底
        if len(context_text) > max_chars:
            context_text = self._truncate_at_sentence(context_text, max_chars)

        final_tokens = self.estimate_tokens(context_text)

        # 7. 元数据
        context_metadata = {
            "total_chunks_retrieved": total_retrieved,
            "unique_chunks_after_dedup": len(unique_chunks),
            "chunks_selected": len(selected),
            "estimated_tokens": final_tokens,
            "chunk_sources": self._analyze_sources(selected),
        }

        return {
            "context_text": context_text,
            "metadata": context_metadata,
            "selected_chunks": selected,
        }


# ==================== 便捷函数 ====================

def assemble_context(
    retrieved_docs: List[InputDoc],
    max_tokens: int = 3000,
    **kwargs: Any,
) -> Dict[str, Any]:
    """便捷函数 — 一行调用。"""
    assembler = ContextAssembler(max_tokens=max_tokens, **kwargs)
    return assembler.assemble(retrieved_docs)


if __name__ == "__main__":
    # 自测
    sample = [
        DocumentChunk(
            text="二甲双胍是治疗2型糖尿病的一线药物,可降低心血管事件风险。",
            metadata={"pmid": "12345", "chunk_index": 0},
            relevance_score=0.92,
        ),
        DocumentChunk(
            text="二甲双胍是治疗2型糖尿病的一线药物,可降低心血管事件风险。它通过抑制肝糖输出发挥作用。",
            metadata={"pmid": "12345", "chunk_index": 1},
            relevance_score=0.85,
        ),
        DocumentChunk(
            text="一项2023年的Meta分析显示,SGLT2抑制剂对心衰患者有显著获益。",
            metadata={"pmid": "67890", "chunk_index": 0},
            relevance_score=0.78,
        ),
    ]
    out = assemble_context(sample, max_tokens=500)
    print("=== context_text ===")
    print(out["context_text"])
    print("=== metadata ===")
    for k, v in out["metadata"].items():
        print(f"  {k}: {v}")
