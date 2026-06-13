# =============================================================================
# ⚠️  DEPRECATED — 已被 stage6_retrieval_pipeline.py 取代 ⚠️
# =============================================================================
# 这个文件是阶段六的"第二版"(独立 MultiPathRetriever / MultiCriteriaReranker),
# 但项目实际跑的是 stage6_retrieval_pipeline.py(基于 multi_path_retriever.py)。
#
# 这个文件保留只为兼容历史测试 test_retrieval_pipeline.py。
# 任何新代码请用 stage6_retrieval_pipeline.py。
#
# 已知问题(参见 code review 报告):
#   P0-4: BM25 keyword_search 拿不到真实分数,直接 [1.0/(i+1)] 伪造
#   P0-5: _normalize_scores 用 min-max 归一化 BGE-reranker logits,丢绝对信号
# =============================================================================
# retrieval_pipeline.py - 阶段六:检索系统第二部分
#
# 三大组件:
#   1. MultiPathRetriever:向量 + BM25 多路检索,3 种融合策略(simple/rrf/weighted)
#   2. MultiCriteriaReranker:多准则重排(relevance + recency + authority)
#   3. RetrievalPipeline:端到端流水线(query → enhance → multi-path → rerank → top_k)
#
# 跑法:
#   from retrieval_pipeline import RetrievalPipeline
#   pipeline = RetrievalPipeline()
#   docs = pipeline.retrieve("What is ARNO?")
#
#   # 或者跑 demo
#   python retrieval_pipeline.py

import os
import math
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


# ==================== 配置常量 ====================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
VECTOR_DB_DIR = "./vector_db"
COLLECTION_NAME = "medical_papers_v4"

# 权威性:期刊影响因子近似(高 IF = 高权威)
# 数据来源:学科常见 IF 区间,本字典覆盖 9 个期刊
JOURNAL_AUTHORITY = {
    "nature": 10.0,
    "science": 10.0,
    "cell": 9.0,
    "nejm": 9.5,
    "lancet": 9.0,
    "jama": 8.5,
    "plos biology": 6.0,
    "plos medicine": 6.0,
    "bmc": 3.0,
    "default": 4.0,  # 未知期刊默认值
}

# 缺省多准则权重
DEFAULT_CRITERIA_WEIGHTS = {
    "relevance": 0.6,   # 相关性(主)
    "recency": 0.25,    # 时效性
    "authority": 0.15,  # 权威性
}


# ==================== 1. MultiPathRetriever ====================
class MultiPathRetriever:
    """
    多路检索器:向量检索 + BM25 关键词检索,支持 3 种融合策略

    Args:
        vectorstore: ChromaDB 向量库实例
        chunks: 原始 chunks 列表(用于 BM25 索引)
        fusion_strategy: 'simple' / 'rrf' / 'weighted'
    """

    def __init__(self,
                 vectorstore: Chroma,
                 chunks: List[Document],
                 fusion_strategy: str = "rrf",
                 vector_weight: float = 0.6,
                 keyword_weight: float = 0.4,
                 rrf_k: int = 60):
        if fusion_strategy not in ("simple", "rrf", "weighted"):
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")

        self.vectorstore = vectorstore
        self.chunks = chunks
        self.fusion_strategy = fusion_strategy
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        # 初始化 BM25
        print(f"[MultiPath] 初始化 BM25 索引 ({len(chunks)} chunks)...")
        self.bm25 = BM25Retriever.from_documents(chunks, k=20)

    def vector_search(self, query: str, query_embedding: List[float],
                      top_k: int = 20) -> List[Document]:
        """向量检索"""
        results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=top_k,
        )
        # results: list of (Document, relevance_score)
        return [doc for doc, _ in results]

    def keyword_search(self, query: str, top_k: int = 20) -> List[Document]:
        """BM25 关键词检索"""
        self.bm25.k = top_k
        return self.bm25.invoke(query)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max 归一化到 [0, 1]"""
        if not scores:
            return []
        s_min, s_max = min(scores), max(scores)
        if s_max - s_min < 1e-9:
            return [1.0] * len(scores)
        return [(s - s_min) / (s_max - s_min) for s in scores]

    def _doc_key(self, doc: Document) -> str:
        """统一的 doc key:优先 doc_id(PMID),其次 source,最后内容 hash"""
        md = doc.metadata or {}
        return str(md.get("doc_id") or md.get("source") or id(doc))

    def _fuse_simple(self, vec_docs: List[Document], kw_docs: List[Document]) -> List[Document]:
        """
        简单合并去重(不保留排名信息)
        - 同一 doc_id 只保留一次,优先保留向量版本
        - 返回顺序:向量在前 + 关键词新增
        """
        seen = set()
        fused = []
        for doc in vec_docs:
            key = self._doc_key(doc)
            if key not in seen:
                seen.add(key)
                fused.append(doc)
        for doc in kw_docs:
            key = self._doc_key(doc)
            if key not in seen:
                seen.add(key)
                fused.append(doc)
        return fused

    def _fuse_rrf(self, vec_docs: List[Document], kw_docs: List[Document]) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion)
        - 每个 doc 的分数 = sum(weight / (rank + k))
        - 排名靠前的 doc 分数更高
        """
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vec_docs):
            key = self._doc_key(doc)
            scores[key] += self.vector_weight / (rank + self.rrf_k)
            doc_map[key] = doc

        for rank, doc in enumerate(kw_docs):
            key = self._doc_key(doc)
            scores[key] += self.keyword_weight / (rank + self.rrf_k)
            # 如果已有,保留最早加入的(向量优先)
            if key not in doc_map:
                doc_map[key] = doc

        # 按 RRF 分数降序
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def _fuse_weighted(self, vec_docs: List[Document], kw_docs: List[Document],
                       vec_scores: List[float], kw_scores: List[float]) -> List[Document]:
        """
        加权融合
        - 归一化各自的分数
        - 同一 doc_id 加权求和
        - 按总分排序
        """
        vec_norm = self._normalize_scores(vec_scores)
        kw_norm = self._normalize_scores(kw_scores)

        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for doc, s in zip(vec_docs, vec_norm):
            key = self._doc_key(doc)
            scores[key] += self.vector_weight * s
            doc_map[key] = doc

        for doc, s in zip(kw_docs, kw_norm):
            key = self._doc_key(doc)
            scores[key] += self.keyword_weight * s
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def retrieve(self, query: str, query_embedding: List[float],
                 top_k_vector: int = 20, top_k_keyword: int = 20,
                 top_k_final: int = 10) -> List[Document]:
        """完整多路检索 + 融合"""
        # 1) 向量检索(同时拿分数)
        vec_results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding, k=top_k_vector
        )
        vec_docs = [doc for doc, _ in vec_results]
        vec_scores = [score for _, score in vec_results]

        # 2) BM25 关键词检索
        kw_docs = self.keyword_search(query, top_k=top_k_keyword)
        # BM25 拿不到分数,默认按返回顺序给递减分数
        kw_scores = [1.0 / (i + 1) for i in range(len(kw_docs))]

        # 3) 融合
        if self.fusion_strategy == "simple":
            fused = self._fuse_simple(vec_docs, kw_docs)
        elif self.fusion_strategy == "rrf":
            fused = self._fuse_rrf(vec_docs, kw_docs)
        elif self.fusion_strategy == "weighted":
            fused = self._fuse_weighted(vec_docs, kw_docs, vec_scores, kw_scores)
        else:
            raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")

        return fused[:top_k_final]


# ==================== 2. MultiCriteriaReranker ====================
class MultiCriteriaReranker:
    """
    多准则重排器:在 BGE-reranker 相关性分数基础上,
    叠加 **时效性**(recency)和 **权威性**(authority)权重。

    Score = w_relevance * relevance + w_recency * recency + w_authority * authority
    """

    def __init__(self,
                 model_name: str = RERANKER_MODEL,
                 criteria_weights: Optional[Dict[str, float]] = None,
                 current_year: Optional[int] = None):
        # 默认权重
        self.weights = criteria_weights or DEFAULT_CRITERIA_WEIGHTS.copy()
        # 校验
        for k in ("relevance", "recency", "authority"):
            if k not in self.weights:
                self.weights[k] = DEFAULT_CRITERIA_WEIGHTS[k]

        self.current_year = current_year or datetime.now().year

        # 加载 BGE reranker
        print(f"[MultiCriteriaReranker] 加载 {model_name}...")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def _recency_score(self, year_str: str) -> float:
        """时新性:近 1 年 = 1.0,每往前 1 年线性衰减 0.1"""
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            return 0.5  # 未知年份给中间分
        age = max(0, self.current_year - year)
        # 衰减 0.1/年,最低 0.1(>40 年前)
        return max(0.1, 1.0 - age * 0.1)

    def _authority_score(self, journal: str) -> float:
        """权威性:基于期刊的近似影响因子,归一化到 [0, 1]"""
        journal = (journal or "").lower().strip()
        # 模糊匹配
        for key, score in JOURNAL_AUTHORITY.items():
            if key != "default" and key in journal:
                return score / 10.0  # 归一化
        return JOURNAL_AUTHORITY["default"] / 10.0

    def rerank(self, query: str, docs: List[Document],
               top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        多准则重排
        Returns: [(doc, final_score), ...] 按 final_score 降序
        """
        if not docs:
            return []

        # 1) BGE reranker 算 relevance
        pairs = [[query, d.page_content] for d in docs]
        relevance_scores = self.model.predict(pairs, show_progress_bar=False)
        # CrossEncoder 输出 logits,需要 sigmoid 归一化到 [0, 1]
        # 用 max-min 归一化(简单)
        rel_min, rel_max = min(relevance_scores), max(relevance_scores)
        if rel_max - rel_min > 1e-9:
            rel_norm = [(s - rel_min) / (rel_max - rel_min) for s in relevance_scores]
        else:
            rel_norm = [1.0] * len(relevance_scores)

        # 2) 算 recency 和 authority
        scored = []
        for doc, rel in zip(docs, rel_norm):
            rec = self._recency_score(doc.metadata.get("year", ""))
            auth = self._authority_score(doc.metadata.get("journal", ""))
            final = (
                self.weights["relevance"] * rel +
                self.weights["recency"] * rec +
                self.weights["authority"] * auth
            )
            scored.append((doc, final, rel, rec, auth))

        # 3) 按 final_score 降序排
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score, _, _, _ in scored[:top_k]]


# ==================== 3. RetrievalPipeline(端到端) ====================
@dataclass
class RetrievalResult:
    """结构化检索结果"""
    query: str                       # 原始 query
    enhanced_query: Any = None        # QueryEnhancer 输出
    vector_query: str = ""            # 给向量的 query(已加 BGE instruction)
    keyword_query: str = ""          # 给 BM25 的 query
    filter_conditions: Dict = field(default_factory=dict)
    retrieved_docs: List[Document] = field(default_factory=list)
    reranked_docs: List[Tuple[Document, float]] = field(default_factory=list)
    fusion_strategy: str = "rrf"
    total_chunks_in_db: int = 0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0

    def summary(self) -> str:
        lines = [
            f"原始 query: {self.query}",
            f"向量 query: {self.vector_query[:80]}...",
            f"关键词 query: {self.keyword_query[:80]}...",
            f"过滤条件: {self.filter_conditions}",
            f"融合策略: {self.fusion_strategy}",
            f"召回数: {len(self.retrieved_docs)} → 重排后: {len(self.reranked_docs)}",
        ]
        if self.reranked_docs:
            lines.append("Top 结果:")
            for i, (doc, score) in enumerate(self.reranked_docs[:3], 1):
                src = doc.metadata.get("doc_id") or doc.metadata.get("source") or "?"
                title = doc.metadata.get("source_title", "")[:50]
                lines.append(f"  {i}. PMID:{src} (score: {score:.3f}) - {title}")
        return "\n".join(lines)


class RetrievalPipeline:
    """
    完整检索流水线
    1) QueryEnhancer → 增强 query + 提取过滤
    2) MultiPathRetriever → 多路召回 + 融合
    3) MultiCriteriaReranker → 多准则重排
    4) Article-level dedup → 返回 top_k
    """

    def __init__(self,
                 vector_db_dir: str = VECTOR_DB_DIR,
                 collection_name: str = COLLECTION_NAME,
                 embedding_model: str = EMBEDDING_MODEL,
                 reranker_model: str = RERANKER_MODEL,
                 fusion_strategy: str = "rrf",
                 criteria_weights: Optional[Dict[str, float]] = None,
                 enable_enhancer: bool = True):
        """
        Args:
            enable_enhancer: 是否使用上周开发的 QueryEnhancer
        """
        # 1) 加载嵌入模型
        print(f"[Pipeline] 加载嵌入模型 {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda" if self._has_cuda() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.embedding_model = embedding_model

        # 2) 加载向量库
        print(f"[Pipeline] 加载向量库 {vector_db_dir}...")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=vector_db_dir,
        )

        # 3) 加载 chunks(用于 BM25)
        chunks_path = "./output/chunks.parquet"
        if os.path.exists(chunks_path):
            import pandas as pd
            df = pd.read_parquet(chunks_path)
            self.chunks = [
                Document(page_content=row["text"], metadata={
                    "doc_id": str(row["doc_id"]),
                    "source": str(row["doc_id"]),  # 用 doc_id(PMID)做统一 source 标识
                    "source_title": str(row.get("source_title", "")),
                    "chunk_id": str(row["chunk_id"]),
                    "year": "",  # chunks.parquet 无此字段,留空让 recency fallback
                    "journal": "",  # 同上
                })
                for _, row in df.iterrows()
            ]
            print(f"[Pipeline] 加载 {len(self.chunks)} chunks 用于 BM25")
        else:
            self.chunks = []
            print(f"[Pipeline] WARNING: 找不到 {chunks_path},BM25 不可用")

        # 4) 初始化多路检索器
        self.multi_path = MultiPathRetriever(
            vectorstore=self.vectorstore,
            chunks=self.chunks,
            fusion_strategy=fusion_strategy,
        )

        # 5) 初始化多准则重排器
        self.reranker = MultiCriteriaReranker(
            model_name=reranker_model,
            criteria_weights=criteria_weights,
        )

        # 6) 初始化 query enhancer(可选)
        self.enhancer = None
        if enable_enhancer:
            try:
                from query_enhancer import QueryEnhancer
                self.enhancer = QueryEnhancer()
                print(f"[Pipeline] QueryEnhancer 已启用")
            except ImportError:
                print(f"[Pipeline] WARNING: 找不到 query_enhancer,跳过")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        端到端检索
        """
        import time
        result = RetrievalResult(query=query, fusion_strategy=self.multi_path.fusion_strategy)
        result.total_chunks_in_db = self.vectorstore._collection.count()

        # 1) Query 增强
        if self.enhancer:
            eq = self.enhancer.enhance(query)
            result.enhanced_query = eq
            result.vector_query = eq.vector_query
            result.keyword_query = eq.keyword_query
            result.filter_conditions = eq.filter_conditions
        else:
            result.vector_query = "Represent this question for searching relevant passages: " + query
            result.keyword_query = query

        # 2) 计算向量嵌入
        t0 = time.time()
        query_emb = self.embeddings.embed_query(result.vector_query)
        result.retrieval_time = time.time() - t0

        # 3) 多路检索 + 融合
        t0 = time.time()
        # 用 keyword_query 喂 BM25
        retrieved = self.multi_path.retrieve(
            query=result.keyword_query,
            query_embedding=query_emb,
            top_k_vector=20,
            top_k_keyword=20,
            top_k_final=20,  # 多召回点,给重排留空间
        )
        result.retrieval_time += time.time() - t0
        result.retrieved_docs = retrieved

        # 4) 多准则重排
        t0 = time.time()
        reranked = self.reranker.rerank(
            query=result.vector_query,
            docs=retrieved,
            top_k=top_k,
        )
        result.rerank_time = time.time() - t0
        result.reranked_docs = reranked

        return result


# ==================== Demo ====================
def run_demo():
    """跑几个典型 query,展示端到端流水线"""
    print("=" * 70)
    print("阶段六:检索系统第二部分 — Demo")
    print("=" * 70)

    # 初始化(会 load 嵌入 + rerank 模型,首次会下载)
    pipeline = RetrievalPipeline(fusion_strategy="rrf")

    test_queries = [
        "What is ARNO?",
        "ARF 蛋白激活磷脂酶 D 的机制是什么?",
        "What is the effect of metformin on cardiovascular disease?",
        "二甲双胍对心血管疾病有何影响?",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"[Q{i}] {q}")
        print("─" * 70)
        try:
            result = pipeline.retrieve(q, top_k=3)
            print(result.summary())
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_demo()
