# stage6_retrieval_pipeline.py - 阶段六:完整检索流水线
#
# 完整流程:
#   Raw Query
#     ↓
#   QueryEnhancer (阶段五: 清洗 / 实体识别 / 同义词扩展 / 元数据过滤)
#     ↓
#   MultiPathRetriever (本阶段: BM25 + 向量 + 融合)
#     ↓
#   MultiCriteriaReranker (本阶段: 相关性 / 时效性 / 权威性)
#     ↓
#   LLM Answer Generation (qwen3:8b 生成答案)
#
# 跑法:
#   python stage6_retrieval_pipeline.py
#   # 或在别的脚本中:
#   from stage6_retrieval_pipeline import RetrievalPipeline
#   pipeline = RetrievalPipeline()
#   result = pipeline.query("二甲双胍对心血管的影响")
#   print(result["answer"])
#
# 依赖:
#   pip install rank_bm25 jieba sentence-transformers torch
#   ollama serve
#   ollama pull qwen3:8b

import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from langchain_chroma import Chroma

# 本地模块
from query_enhancer import QueryEnhancer, EnhancedQuery
from multi_path_retriever import (
    MultiPathRetriever,
    MultiCriteriaReranker,
    rrf_fusion,
    weighted_fusion,
    BM25Index,
)
from rag_medical import (
    load_documents,
    split_documents,
    load_vectorstore,
    get_embeddings,
    parse_pubmed_xml,
)

# ==================== 配置 ====================
DATA_PATH = "./data/medical_papers"
VECTOR_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "medical_papers_v2_bgem3"
MODEL_NAME = "qwen3:8b"
EMBEDDING_MODEL = "bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-base"

# 检索参数
RETRIEVER_TOP_K = 30       # BM25 和向量各自召回数量
RERANKER_TOP_K = 5        # 重排序后返回给 LLM 的数量
FUSION_STRATEGY = "rrf"   # 'rrf' | 'weighted' | 'simple'
VECTOR_WEIGHT = 0.6       # 加权融合时向量权重 (BM25 = 1-VECTOR_WEIGHT)

# 多准则权重
CRITERIA_WEIGHTS = {
    "relevance": 0.60,   # BGE-reranker 相关性
    "recency":   0.25,   # 时效性(年份衰减)
    "authority": 0.15,   # 期刊权威性
}

MAX_FILES = None  # None=全量, 50=作业演示用
# =============================================


# ==================== 结果结构 ====================
@dataclass
class RetrievalResult:
    """完整检索流水线输出"""
    query: str                              # 原始查询
    enhanced: Optional[EnhancedQuery]       # 查询增强结果
    retrieved_docs: List[Document]          # 检索结果
    retrieval_time_ms: float                # 检索耗时(ms)
    fusion_strategy: str                    # 使用的融合策略
    fusion_stats: Dict = field(default_factory=dict)  # 融合统计
    answer: str = ""                       # LLM 生成答案
    generation_time_ms: float = 0.0       # LLM 生成耗时
    total_time_ms: float = 0.0            # 全流程耗时
    error: Optional[str] = None            # 错误信息

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "retrieved_count": len(self.retrieved_docs),
            "retrieval_time_ms": round(self.retrieval_time_ms, 1),
            "generation_time_ms": round(self.generation_time_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
            "fusion_strategy": self.fusion_strategy,
            "fusion_stats": self.fusion_stats,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "sources": [
                {
                    "source": d.metadata.get("source", ""),
                    "year": d.metadata.get("year", ""),
                    "journal": d.metadata.get("journal", ""),
                    "pmid": d.metadata.get("pmid", ""),
                    "preview": d.page_content[:100].replace("\n", " "),
                }
                for d in self.retrieved_docs
            ],
            "enhanced_query": self.enhanced.to_dict() if self.enhanced else {},
            "error": self.error,
        }


# ==================== 完整流水线 ====================
class RetrievalPipeline:
    """
    医疗文献检索完整流水线

    集成阶段五(QueryEnhancer) + 阶段六(MultiPathRetriever + Reranker)
    支持:
      - 多种融合策略(RRF / 加权 / 简单)
      - 多种重排序准则(相关性 / 时效性 / 权威性)
      - 多查询变体(QueryEnhancer 生成的同义词版本)
      - LLM 答案生成
      - 统计分析
    """

    def __init__(
        self,
        data_path: str = DATA_PATH,
        vector_persist_dir: str = VECTOR_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        llm_model: str = MODEL_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        rerank_model: str = RERANK_MODEL,
        fusion_strategy: str = FUSION_STRATEGY,
        vector_weight: float = VECTOR_WEIGHT,
        retriever_top_k: int = RETRIEVER_TOP_K,
        reranker_top_k: int = RERANKER_TOP_K,
        criteria_weights: Optional[Dict[str, float]] = None,
        max_files: Optional[int] = MAX_FILES,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.config = {
            "data_path": data_path,
            "vector_persist_dir": vector_persist_dir,
            "collection_name": collection_name,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
            "fusion_strategy": fusion_strategy,
            "vector_weight": vector_weight,
            "retriever_top_k": retriever_top_k,
            "reranker_top_k": reranker_top_k,
            "criteria_weights": criteria_weights or CRITERIA_WEIGHTS,
            "max_files": max_files,
            "ollama_base_url": ollama_base_url,
        }

        self.query_enhancer = QueryEnhancer()

        # 加载文档
        print("=" * 60)
        print("初始化检索流水线...")
        print("=" * 60)
        print(f"  数据路径: {data_path}")
        print(f"  向量库: {vector_persist_dir}")
        print(f"  融合策略: {fusion_strategy}")
        print(f"  多准则权重: {criteria_weights or CRITERIA_WEIGHTS}")

        raw_docs = load_documents(data_path)
        if max_files and len(raw_docs) > max_files:
            raw_docs = raw_docs[:max_files]

        chunks = split_documents(raw_docs)
        print(f"  Chunks 总数: {len(chunks)}")

        # 加载/构建向量库
        embeddings = get_embeddings()
        if os.path.exists(vector_persist_dir):
            self.vectorstore = load_vectorstore(vector_persist_dir, embeddings)
            print(f"  向量库已加载: {self.vectorstore._collection.count()} 条向量")
        else:
            from rag_medical import build_vectorstore
            self.vectorstore = build_vectorstore(chunks, vector_persist_dir, embeddings)

        self.chunks = chunks

        # 构建多路检索器
        self._build_retriever()

        # 初始化 LLM
        print(f"  连接 LLM: {llm_model} @ {ollama_base_url}")
        self.llm = OllamaLLM(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0.3,
            num_predict=1024,
        )

        print("✅ 流水线初始化完成\n")

    def _build_retriever(self):
        """构建多路检索器"""
        self.retriever = MultiPathRetriever(
            vectorstore=self.vectorstore,
            chunks=self.chunks,
            fusion_strategy=self.config["fusion_strategy"],
            vector_weight=self.config["vector_weight"],
            top_k_vector=self.config["retriever_top_k"],
            top_k_bm25=self.config["retriever_top_k"],
            reranker_top_k=self.config["reranker_top_k"],
            criteria_weights=self.config["criteria_weights"],
        )
        print(f"  多路检索器已构建")

    # ==================== 核心:单次查询 ====================
    def query(self,
               raw_query: str,
               use_enhanced: bool = True,
               generate_answer: bool = True,
               verbose: bool = True) -> RetrievalResult:
        """
        执行完整检索流程

        Args:
            raw_query: 用户原始查询
            use_enhanced: 是否使用查询增强(QueryEnhancer)
            generate_answer: 是否生成 LLM 答案
            verbose: 是否打印详细信息

        Returns:
            RetrievalResult: 包含检索结果和(可选)答案
        """
        t_start = time.perf_counter()
        result = RetrievalResult(
            query=raw_query,
            enhanced=None,
            retrieved_docs=[],
            retrieval_time_ms=0.0,
            fusion_strategy=self.config["fusion_strategy"],
        )

        try:
            # ===== Step 1: 查询增强 =====
            if use_enhanced:
                t_eq = time.perf_counter()
                enhanced = self.query_enhancer.enhance(raw_query)
                result.enhanced = enhanced
                # BM25 拿 keyword_query(cleaned + 同义词扩展,纯文本)
                retrieval_query = enhanced.keyword_query or enhanced.cleaned
                # 向量检索拿 vector_query(带 BGE instruction,无同义词拼接)
                vector_query = enhanced.vector_query
                year_filter = enhanced.filter_conditions.get("year_filter")
                journal_filter = enhanced.filter_conditions.get("journal_filter")
                t_eq_end = time.perf_counter()

                if verbose:
                    print(f"\n🔍 查询: {raw_query}")
                    print(f"  → 清洗: {enhanced.cleaned}")
                    if enhanced.entities:
                        print(f"  → 实体: {enhanced.entities}")
                    if enhanced.synonyms:
                        print(f"  → 同义词: {enhanced.synonyms[:5]}...")
                    if enhanced.filter_conditions:
                        print(f"  → 过滤: {enhanced.filter_conditions}")
                    print(f"  → 检索 query: {retrieval_query[:80]}")
                    print(f"  → 查询增强耗时: {(t_eq_end-t_eq)*1000:.1f}ms")
            else:
                retrieval_query = raw_query
                vector_query = raw_query  # 没 enhancer 就用 raw query 当 vector query
                year_filter = None
                journal_filter = None
                enhanced = None

            # ===== Step 2: 多路检索 =====
            t_ret = time.perf_counter()
            docs = self.retriever.retrieve(
                query=retrieval_query,
                year_filter=year_filter,
                journal_filter=journal_filter,
                vector_query=vector_query,
            )
            t_ret_end = time.perf_counter()
            result.retrieval_time_ms = (t_ret_end - t_ret) * 1000
            result.retrieved_docs = docs

            if verbose:
                print(f"\n  → 检索耗时: {result.retrieval_time_ms:.1f}ms")
                print(f"  → 召回文档: {len(docs)} 篇")

            # ===== Step 3: LLM 答案生成 =====
            if generate_answer and docs:
                t_gen = time.perf_counter()
                answer = self._generate_answer(raw_query, docs)
                t_gen_end = time.perf_counter()
                result.answer = answer
                result.generation_time_ms = (t_gen_end - t_gen) * 1000

                if verbose:
                    print(f"\n  → 生成耗时: {result.generation_time_ms:.1f}ms")
                    print(f"\n{'='*60}")
                    print("📖 答案:")
                    print(f"{'='*60}")
                    print(answer)
                    self._print_sources(docs)
            elif not docs:
                result.answer = "根据提供的医学文献库,未找到与该问题相关的文档。"
                if verbose:
                    print("\n  ⚠️ 未召回任何相关文档")

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"\n  ❌ 错误: {result.error}")

        t_total = time.perf_counter()
        result.total_time_ms = (t_total - t_start) * 1000

        if verbose:
            print(f"\n  ⏱️ 总耗时: {result.total_time_ms:.1f}ms")

        return result

    def _generate_answer(self, question: str, docs: List[Document]) -> str:
        """基于检索结果生成 LLM 答案。

        异常向上抛 — 由 query() 的 try/except 捕获并填到 result.error。
        之前这里 try/except 返回错误字符串,导致 result.error 永远为空,
        evaluate() 把 LLM 失败也算成"成功"了 (P2-5 修复)。
        """
        prompt = self._build_prompt(question, docs)
        return self.llm.invoke(prompt)

    def _build_prompt(self, question: str, docs: List[Document]) -> str:
        """构建医疗专用 prompt"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = (f"【文献 {i}】PMID: {meta.get('pmid','?')} | "
                      f"{meta.get('year','?')} | {meta.get('journal','?')}")
            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        prompt = f"""你是一名严谨的医学文献助手,严格基于下面提供的医学文献片段回答问题。

【回答规则】
1. 只能基于下方 context 的内容回答,不得编造任何数据、结论或参考文献
2. 引用具体内容时,使用 [1][2][3] 的格式标注来源编号
3. 如果 context 不包含答案,直接回答"根据提供的医学文献,我无法回答该问题",不要推测
4. 不给出诊断、用药剂量或治疗方案;涉及个体医疗建议时,建议咨询执业医师
5. 区分"文献报道"和"临床建议",措辞要保守、严谨

【文献片段】
{context}

【用户问题】
{question}

【回答】"""
        return prompt

    def _print_sources(self, docs: List[Document]):
        """打印引用来源"""
        if not docs:
            return
        print(f"\n📚 参考来源:")
        seen = set()
        for doc in docs:
            pmid = doc.metadata.get("pmid", "?")
            src = doc.metadata.get("source", "?")
            year = doc.metadata.get("year", "")
            journal = doc.metadata.get("journal", "")
            if pmid in seen:
                continue
            seen.add(pmid)
            line = f"  - PMID: {pmid} | {src} | {year}"
            if journal:
                line += f" | {journal}"
            print(line)
            if pmid and pmid != "?":
                print(f"    https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

    # ==================== 批量测试 ====================
    def batch_query(self,
                    queries: List[str],
                    use_enhanced: bool = True,
                    generate_answer: bool = False,
                    verbose: bool = False) -> List[RetrievalResult]:
        """
        批量查询(用于评估)
        默认不生成 LLM 答案,加快评估速度
        """
        results = []
        for i, q in enumerate(queries, 1):
            if verbose:
                print(f"\n[{i}/{len(queries)}]", end=" ")
            r = self.query(q, use_enhanced=use_enhanced,
                           generate_answer=generate_answer, verbose=False)
            results.append(r)
            if verbose:
                recall = len(r.retrieved_docs)
                t_ms = r.total_time_ms
                status = "✅" if r.error is None else "❌"
                print(f"  {status} {q[:40]}... → {recall} 篇 ({t_ms:.0f}ms)")
        return results

    def evaluate(self,
                 queries: List[str],
                 use_enhanced: bool = True) -> Dict:
        """
        评估流水线性能
        生成统计报告
        """
        print("\n" + "=" * 60)
        print("检索流水线评估")
        print("=" * 60)

        results = self.batch_query(queries, use_enhanced=use_enhanced,
                                   generate_answer=False, verbose=True)

        total = len(results)
        success = sum(1 for r in results if r.error is None)
        has_docs = sum(1 for r in results if len(r.retrieved_docs) > 0)
        avg_recall = sum(len(r.retrieved_docs) for r in results) / total if total else 0
        avg_ret_ms = sum(r.retrieval_time_ms for r in results) / total if total else 0
        avg_total_ms = sum(r.total_time_ms for r in results) / total if total else 0

        report = {
            "total_queries": total,
            "success": success,
            "failure": total - success,
            "has_documents": has_docs,
            "no_documents": total - has_docs,
            "avg_recall_per_query": round(avg_recall, 2),
            "avg_retrieval_time_ms": round(avg_ret_ms, 1),
            "avg_total_time_ms": round(avg_total_ms, 1),
            "fusion_strategy": self.config["fusion_strategy"],
            "criteria_weights": self.config["criteria_weights"],
            "per_query": [r.to_dict() for r in results],
        }

        print(f"\n{'='*60}")
        print("📊 评估报告:")
        print(f"  总查询数: {total}")
        print(f"  成功: {success} | 失败: {total-success}")
        print(f"  有召回结果: {has_docs}/{total}")
        print(f"  平均召回数/查询: {avg_recall:.1f} 篇")
        print(f"  平均检索耗时: {avg_ret_ms:.1f}ms")
        print(f"  平均总耗时: {avg_total_ms:.1f}ms")

        return report


# ==================== 对比实验 ====================
def compare_fusion_strategies(pipeline: RetrievalPipeline,
                               test_queries: List[str],
                               verbose: bool = False) -> Dict:
    """
    对比不同融合策略的检索效果

    对每个查询分别跑 RRF / 加权 / 简单三种策略,
    统计召回率、平均分数等指标。

    优化 (P2-4):只切换融合策略,不重建 BM25 索引和重排序器。
    之前每个策略都会重建检索器,导致 evaluate 慢 6-10 秒。
    """
    strategies = ["rrf", "weighted", "simple"]
    results_map = {}

    # 记住原策略,跑完恢复(避免污染 pipeline 状态)
    original_strategy = pipeline.retriever.fusion_strategy

    for strategy in strategies:
        # 直接切 fusion_strategy(不重建索引,见 MultiPathRetriever.fusion_strategy setter)
        pipeline.retriever.fusion_strategy = strategy

        r_list = pipeline.batch_query(
            test_queries,
            use_enhanced=True,
            generate_answer=False,
            verbose=False,
        )
        results_map[strategy] = r_list

        # 统计
        recall_list = [len(r.retrieved_docs) for r in r_list]
        print(f"  [{strategy}] 平均召回: {sum(recall_list)/len(recall_list):.1f} 篇, "
              f"成功: {sum(1 for r in r_list if not r.error)}/{len(r_list)}")

    # 恢复原策略
    pipeline.retriever.fusion_strategy = original_strategy

    # 对比报告
    report = {}
    for strategy, r_list in results_map.items():
        recalls = [len(r.retrieved_docs) for r in r_list]
        times = [r.retrieval_time_ms for r in r_list]
        report[strategy] = {
            "avg_recall": round(sum(recalls)/len(recalls), 2),
            "max_recall": max(recalls),
            "min_recall": min(recalls),
            "avg_time_ms": round(sum(times)/len(times), 1),
            "success_rate": round(sum(1 for r in r_list if not r.error)/len(r_list)*100, 1),
        }

    print("\n融合策略对比:")
    for s, stats in report.items():
        print(f"  {s:8s} | recall={stats['avg_recall']:4.1f} | "
              f"time={stats['avg_time_ms']:6.1f}ms | "
              f"success={stats['success_rate']:.0f}%")

    return report


# ==================== Demo / 主入口 ====================
def _get_demo_queries() -> List[str]:
    """演示用查询列表"""
    return [
        "二甲双胍对心血管疾病的影响",
        "What is the effect of metformin on cardiovascular disease?",
        "PD-1 免疫疗法近五年研究进展",
        "阿司匹林和氯吡格雷联合用药的效果",
        "EGFR mutations in lung cancer treatment",
        "高血压患者使用 ARB 治疗的效果",
        "SGLT2 inhibitors for heart failure in diabetes",
        "近三年关于 CAR-T 细胞疗法的研究",
        "二型糖尿病患者使用恩格列净对心血管结局的影响",
        "COVID-19 新冠后遗症研究进展",
    ]


def main():
    print("=" * 60)
    print("阶段六:医疗文献检索完整流水线")
    print("  MultiPathRetriever + MultiCriteriaReranker + LLM")
    print("=" * 60)

    # 初始化流水线
    pipeline = RetrievalPipeline(max_files=50)

    # 单次演示查询
    demo_queries = _get_demo_queries()

    print("\n" + "=" * 60)
    print("交互模式:输入问题,回车查询;输入 'batch' 运行批量测试")
    print("输入 'compare' 对比融合策略;输入 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n请输入问题(或命令): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见")
            break

        if user_input.lower() == "exit":
            print("👋 再见")
            break
        elif user_input.lower() == "batch":
            print("\n📊 运行批量测试...")
            report = pipeline.evaluate(demo_queries)
            # 保存报告
            report_path = "./output/stage6_evaluation_report.json"
            os.makedirs("./output", exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  报告已保存: {report_path}")
        elif user_input.lower() == "compare":
            print("\n🔬 对比融合策略...")
            compare_fusion_strategies(pipeline, demo_queries[:5])
        elif user_input:
            pipeline.query(user_input, verbose=True)


if __name__ == "__main__":
    main()
