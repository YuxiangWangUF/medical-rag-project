# multi_path_retriever.py - 阶段六 Part A: 多路检索器
#
# 功能:
#   1. BM25 索引构建 + 检索
#   2. ChromaDB 向量检索
#   3. 三种融合策略: RRF / 加权融合 / 简单合并去重
#   4. 多准则重排序器 (BGE-reranker-base + 相关性/时效性/权威性)
#
# 跑法:
#   python multi_path_retriever.py      # 独立测试用
#   from multi_path_retriever import MultiPathRetriever, MultiCriteriaReranker
#
# 依赖: rank_bm25, sentence-transformers (CrossEncoder)

import math
import jieba
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
# bge-m3 is available in Ollama 0.24.0
# ChromaDB must be rebuilt with bge-m3 (1024 dims) before use
# Run: python rebuild_chroma.py

import numpy as np

# ==================== 配置 ====================
EMBEDDING_MODEL = "bge-m3" # Ollama bge-m3 (1024 dims)
RERANK_MODEL = "BAAI/bge-reranker-base"
VECTOR_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "medical_papers_v2_bgem3"  # rebuilt with bge-m3
# =============================================


# ==================== 工具函数 ====================
import re

# 中文字符范围(CJK 基本)
_CN_RE = re.compile(r"[\u4e00-\u9fff]+")
# 标点 / 空白
_NOISE_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

# jieba 自定义词典(医学术语,jieba 默认不识别)
# 作用:让 jieba 优先把"二甲双胍"切成一个词,而不是拆成"二甲"+"胍"
_CUSTOM_DICT_LOADED = False


def _load_custom_dict():
    """懒加载:第一次用 jieba 之前注入医学术语"""
    global _CUSTOM_DICT_LOADED
    if _CUSTOM_DICT_LOADED:
        return
    _CUSTOM_DICT_LOADED = True

    custom_words = [
        # 药物
        "二甲双胍", "格华止", "阿司匹林", "氯吡格雷", "恩格列净",
        "恩格列净", "利拉鲁肽", "他汀类", "ACEI", "ARB",
        # 疾病
        "心血管疾病", "心力衰竭", "高血压", "糖尿病", "二型糖尿病",
        "心肌梗死", "冠心病", "心律失常", "房颤", "血脂异常",
        "免疫检查点", "肿瘤免疫", "靶向治疗",
        # 治疗手段
        "免疫疗法", "免疫治疗", "联合用药", "二级预防", "联合治疗",
        "靶向治疗", "基因治疗", "细胞治疗",
        # 蛋白 / 基因
        "ARNO", "ARF", "PLD", "磷脂酶", "鸟苷酸", "Sec7",
        "EGFR", "PD-1", "PD-L1", "CAR-T", "HER2", "BCR-ABL",
        "JAK", "GLP-1", "SGLT2", "TGFβ", "BMP",
    ]
    for w in custom_words:
        jieba.add_word(w, freq=10000, tag="med")


def tokenize(text: str) -> List[str]:
    """
    统一分词:同时处理中英双语,query 和 doc 用同一个函数(关键!)

    步骤:
    1. 拆出中文片段(连续 CJK) → jieba.cut → 过滤单字(2+ 才保留)
    2. 拆出英文片段(连续 [a-z0-9]) → 转小写、拆 token
    3. 过滤掉纯标点 / 空白 / 1 字 token

    之前 bug:  jieba 对混合文本会把中文按字切('二' '甲' '双' '胍'),
              然后 query 跟 doc 切出来不一致 → 永远 0 命中
    """
    _load_custom_dict()  # 懒加载医学词典
    tokens: List[str] = []
    text_lower = text.lower()

    # 1) 提取所有英文/数字 token(连续的 a-z 0-9)
    for m in re.finditer(r"[a-z0-9][a-z0-9\-]+", text_lower):
        tok = m.group()
        if len(tok) >= 2:  # 过滤 1 字英文
            tokens.append(tok)

    # 2) 提取所有中文片段,逐段 jieba 切
    for m in _CN_RE.finditer(text):
        cn_segment = m.group()
        for w in jieba.cut(cn_segment):
            w = w.strip()
            # 过滤:纯标点 / 空白 / 单字中文(避免 doc_freq 爆炸)
            if not w or _NOISE_RE.match(w) or len(w) < 2:
                continue
            tokens.append(w)

    return tokens


def _doc_key(item) -> str:
    """
    统一 doc key,用于融合去重

    优先级:pmid > source > doc_id > id(item)
    - pmid 是 PubMed ID,跨 BM25 / 向量 / 各种重排路径最稳定的标识
    - 如果都没有(单元测试 fixture),用 id 作为最后兜底
    """
    # item 可能是 Document,也可能是 tuple
    doc = item[0] if isinstance(item, tuple) else item
    md = getattr(doc, "metadata", None) or {}
    key = md.get("pmid") or md.get("source") or md.get("doc_id")
    if key:
        return f"pmid:{key}"
    return f"id:{id(doc)}"


# ==================== BM25 索引 ====================
class BM25Index:
    """
    手动实现的 BM25 索引,对比 langchain 内置版本的优势:
    - 完全可控,可输出每篇文档的原始分数
    - 支持自定义 k1, b 参数
    - 支持按年份过滤(在检索阶段)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_metadata: List[Dict] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> doc frequency
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        _load_custom_dict()  # 保证 fit 阶段也加载
        return tokenize(text)

    def fit(self, documents: List[Document]):
        """
        构建 BM25 索引

        Args:
            documents: langchain Document 列表,每个 Document.page_content 为正文,
                      metadata 包含 'source', 'year', 'journal' 等
        """
        self.corpus = [doc.page_content for doc in documents]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.doc_ids = [doc.metadata.get("source", f"doc_{i}")
                         for i, doc in enumerate(documents)]
        self.doc_metadata = [doc.metadata for doc in documents]
        N = len(self.corpus)

        # 统计 doc frequency
        self.doc_freqs = {}
        for tokens in self.tokenized_corpus:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        # 计算 IDF
        for t, df in self.doc_freqs.items():
            self.idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        # 平均文档长度
        self.avgdl = sum(len(tokens) for tokens in self.tokenized_corpus) / N if N > 0 else 0
        self._fitted = True
        print(f"  [BM25] 索引构建完成: {N} 篇文档, {len(self.idf)} 个倒排词条")

    def get_scores(self, query: str) -> np.ndarray:
        """
        计算 query 对所有文档的 BM25 分数

        Returns:
            scores: ndarray, shape (N,), 每篇文档的 BM25 分数
        """
        if not self._fitted:
            raise RuntimeError("BM25 索引未构建,请先调用 fit()")
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.corpus))

        doc_len = [len(tokens) for tokens in self.tokenized_corpus]

        for q_term in query_tokens:
            if q_term not in self.idf:
                continue
            idf = self.idf[q_term]
            for i, tokens in enumerate(self.tokenized_corpus):
                tf = tokens.count(q_term)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len[i] / self.avgdl)
                scores[i] += idf * numerator / denominator

        return scores

    def search(self, query: str, top_k: int = 20,
               year_filter: Optional[Dict] = None,
               journal_filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回数量
            year_filter:  年份过滤, 如 {"$gte": "2021"}
            journal_filter: 期刊过滤, 如 {"$in": ["Nature", "NEJM"]}

        Returns:
            List of (Document, bm25_score) sorted by score descending
        """
        scores = self.get_scores(query)
        results = []

        for i, score in enumerate(scores):
            if score <= 0:
                continue
            meta = self.doc_metadata[i]

            # 应用元数据过滤
            if year_filter:
                doc_year = meta.get("year", "")
                if doc_year:
                    if "$gte" in year_filter and doc_year < year_filter["$gte"]:
                        continue
                    if "$lte" in year_filter and doc_year > year_filter["$lte"]:
                        continue
                    if "$in" in year_filter and doc_year not in year_filter["$in"]:
                        continue
                else:
                    continue  # 没有年份信息的跳过

            if journal_filter:
                doc_journal = meta.get("journal", "").lower()
                if "$in" in journal_filter:
                    kw = [j.lower() for j in journal_filter["$in"]]
                    if not any(j in doc_journal for j in kw):
                        continue
                elif "$eq" in journal_filter:
                    if journal_filter["$eq"].lower() not in doc_journal:
                        continue

            doc = Document(page_content=self.corpus[i], metadata=meta)
            results.append((doc, float(score)))

        # 按分数降序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ==================== 融合策略 ====================
def rrf_fusion(results_list: List[List[Tuple[Any, float]]],
               k: int = 60) -> List[Tuple[Any, float]]:
    """
    Reciprocal Rank Fusion (RRF)

    核心思想:一篇文档在多个检索路径中排名越靠前,融合分数更高。
    RRF_score(d) = Σ 1/(k + rank_i(d))
    k 是平滑参数,通常取 60。k 越大,各路径权重越均衡。

    同源文档(同 pmid)会被合并,跨 BM25 / 向量两条路径的命中累加。

    Args:
        results_list: 每个元素是一路检索的结果,格式为 [(doc, score), ...]
                     其中 score 是该路的原始分数,doc 需要可哈希
        k: RRF 平滑参数
    """
    rrf_scores: Dict[str, Tuple[Any, float]] = {}

    for results in results_list:
        for rank, (item, _) in enumerate(results, 1):
            key = _doc_key(item)
            if key not in rrf_scores:
                rrf_scores[key] = (item, 0.0)
            rrf_scores[key] = (rrf_scores[key][0], rrf_scores[key][1] + 1.0 / (k + rank))

    fused = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
    return fused


def weighted_fusion(results_list: List[List[Tuple[Any, float]]],
                    weights: Optional[List[float]] = None) -> List[Tuple[Any, float]]:
    """
    加权分数融合

    公式: final_score(d) = Σ w_i * norm(score_i(d))
    其中 norm(score_i) = score_i / max_score_i (按路归一化)

    **修复 (P0-3)**:单路命中(只在 1 路出现)的 doc 不会因为另一路缺位而被稀释,
    因为我们按 doc 出现过的路径集合,只累加存在的路径,而不是按 total 路径归一化。

    Args:
        results_list: 每路检索结果 [(doc, score), ...]
        weights: 每路权重,如 [0.6, 0.4]。默认均分
    """
    if not results_list:
        return []

    n_paths = len(results_list)
    if weights is None:
        weights = [1.0 / n_paths] * n_paths
    else:
        weights = list(weights)
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    # 收集所有文档(按 pmid/source 合并)
    all_docs: Dict[str, Tuple[Any, Dict]] = {}  # key -> (doc, {path: raw_score})
    for pi, results in enumerate(results_list):
        for item, score in results:
            key = _doc_key(item)
            if key not in all_docs:
                all_docs[key] = (item, {})
            all_docs[key][1][pi] = score

    # 计算每路最大值(用于归一化)
    path_max_scores = []
    for results in results_list:
        if results:
            path_max_scores.append(max(score for _, score in results))
        else:
            path_max_scores.append(1.0)

    # 加权融合 — 只累加出现过的路径,不稀释
    fused_results = []
    for key, (doc, path_scores) in all_docs.items():
        final_score = 0.0
        for pi, raw_score in path_scores.items():
            max_s = path_max_scores[pi]
            norm_score = raw_score / max_s if max_s > 0 else 0.0
            final_score += weights[pi] * norm_score
        fused_results.append((doc, float(final_score)))

    fused_results.sort(key=lambda x: x[1], reverse=True)
    return fused_results


def simple_fusion(results_list: List[List[Tuple[Any, float]]]) -> List[Tuple[Any, float]]:
    """
    简单合并去重:将所有路的结果拼接,按原始分数排序去重

    优点:实现简单
    缺点:忽略排名信息,不同路径的结果可能有偏
    """
    seen_keys = set()
    merged = []

    for results in results_list:
        for item, score in results:
            key = _doc_key(item)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append((item, score))

    # 按分数降序
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


# ==================== 多准则重排序器 ====================
class MultiCriteriaReranker:
    """
    多准则重排序器

    综合三个维度打分:
      1. relevance (相关性)    — BGE-reranker 打分,权重最高
      2. recency (时效性)    — 年份越新分数越高,线性衰减
      3. authority (权威性)   — 顶级期刊权重更高

    融合方式: weighted_sum
      final_score = w_relevance * norm(relevance)
                  + w_recency * norm(recency)
                  + w_authority * norm(authority)
    """

    # 默认权重
    DEFAULT_WEIGHTS = {
        "relevance": 0.60,
        "recency":   0.25,
        "authority":  0.15,
    }

    # 期刊权威性权重 (越高越权威)
    # 匹配规则:每条 entry 含 (match_strings, weight)
    #   - 顺序敏感:从上到下匹配,**先列更具体的**(否则 "cell" 会吃掉 "Cell Reports")
    #   - 短缩写(<=5 字符)用单词边界匹配,长名(>=6)用子串匹配
    JOURNAL_WEIGHTS = [
        # === 顶刊 — 长全名/缩写,先列 ===
        (["new england journal of medicine", "n engl j med", "nejm"], 5.0),
        (["journal of the american medical association", "jama"], 5.0),

        # === Nature 系 — 子刊明确降权 ===
        (["nature communications"], 4.0),  # IF ≈ 17,不是 Nature 主刊
        (["nature medicine", "nat med"], 5.0),  # 顶刊
        (["nature genetics", "nature immunology", "nature cell biology",
          "nature biotechnology", "nature methods"], 4.5),
        (["nature"], 5.0),  # Nature 主刊

        # === Lancet 系 ===
        (["lancet"], 5.0),  # 主刊

        # === Science 系 ===
        (["science"], 5.0),  # 主刊(注意:有 "Science Translational Medicine" 等)

        # === Cell 系 — 关键:Cell Reports 不是 Cell ===
        (["cell reports", "cell reports medicine"], 4.0),  # IF ≈ 7-10,降权
        (["cell"], 5.0),  # Cell 主刊

        # === 主医刊 ===
        (["british medical journal", "bmj"], 4.5),
        (["circulation"], 4.5),
        (["j clin oncol", "jco"], 4.5),  # Journal of Clinical Oncology
        (["j natl cancer inst", "jnci"], 4.5),
        (["annals of oncology", "ann oncol"], 4.0),
        (["european heart journal", "european heart"], 4.0),
        (["plos medicine", "plos med"], 3.5),

        # === 兜底 ===
        (["default"], 1.0),
    ]

    # 时效性参考年份(越近越高) — 用动态年份,避免跨年失效
    @staticmethod
    def _reference_year() -> int:
        from datetime import datetime
        return datetime.now().year

    def __init__(self,
                 model_name: str = RERANK_MODEL,
                 criteria_weights: Optional[Dict[str, float]] = None,
                 current_year: Optional[int] = None):
        self.criteria_weights = criteria_weights or self.DEFAULT_WEIGHTS.copy()
        self.current_year = current_year if current_year is not None else self._reference_year()
        self.reranker = None
        self._fallback_mode = False

        print(f"加载 BGE-reranker ({model_name})...")
        try:
            self.reranker = CrossEncoder(model_name)
        except Exception as e:
            print(f"WARNING: Cannot load BGE-reranker ({type(e).__name__})")
            print("  -> Fallback: using bge-m3 embedding similarity for relevance")
            self._fallback_mode = True
            from ollama_batch_embeddings import OllamaEmbeddingsBatch
            self._emb = OllamaEmbeddingsBatch(model="bge-m3")
            _ = self._emb.embed_query("warmup")  # warmup

        print(f"多准则权重: {self.criteria_weights}")

    def _get_authority_score(self, metadata: Dict) -> float:
        """根据期刊名返回权威性分数

        匹配策略:每个 JOURNAL_WEIGHTS 条目给一组字符串,按"长名优先"排序匹配。
        短缩写(如 "nejm"、"jama")采用子串包含(因为它们就是缩写,本身就是字符串);
        长名(如 "nature"、"cell")按"独立词"匹配,防止子刊误中(见 P1-1)。
        """
        journal = metadata.get("journal", "").lower()
        if not journal:
            return self._journal_default_weight()

        for match_strings, weight in self.JOURNAL_WEIGHTS:
            if "default" in match_strings:
                continue  # 最后兜底
            for kw in match_strings:
                # 短缩写(<=5 字符)要求独立词边界(避免 "nejm" 误中 "jnejmxxx")
                # 长名(>=6 字符)允许子串包含(因为它通常就是完整名)
                if len(kw) <= 5:
                    import re as _re
                    if not _re.search(rf"\b{_re.escape(kw)}\b", journal):
                        continue
                else:
                    if kw not in journal:
                        continue
                return weight
        return self._journal_default_weight()

    @classmethod
    def _journal_default_weight(cls) -> float:
        for match_strings, weight in cls.JOURNAL_WEIGHTS:
            if "default" in match_strings:
                return weight
        return 1.0

    def _get_recency_score(self, metadata: Dict) -> float:
        """
        时效性分数:年份越近越高,线性衰减
        公式: max(0, 1 - (当前年 - doc_year) / 10)
        即:10年内的文档都有正分数,10年前=0
        """
        year_str = metadata.get("year", "")
        if not year_str:
            return 0.5  # 未知年份给中间值

        try:
            year = int(year_str)
        except ValueError:
            return 0.5

        age = self.current_year - year
        score = max(0.0, 1.0 - age / 10.0)
        return score

    def rerank(self,
               query: str,
               candidates: List[Tuple[Any, float]],
               top_n: int = 5) -> List[Tuple[Any, float]]:
        """
        多准则重排序

        Args:
            query: 查询文本
            candidates: 候选文档列表,[(doc, fusion_score), ...]
            top_n: 返回前 N 条

        Returns:
            List of (doc, final_score) sorted by final_score descending
        """
        if not candidates:
            return []

        # 1. 获取 relevance 分数 (CrossEncoder 或 embedding 回退)
        if self._fallback_mode:
            # Fallback: 用 bge-m3 embedding相似度作为 relevance
            query_emb = self._emb.embed_query(query)
            doc_texts = [doc.page_content for doc, _ in candidates]
            doc_embs = self._emb.embed_documents(doc_texts)
            import numpy as np
            q_arr = np.array(query_emb)
            relevance_scores = np.array([
                float(np.dot(q_arr, np.array(d)) / (np.linalg.norm(q_arr) * np.linalg.norm(d) + 1e-9))
                for d in doc_embs
            ])
        else:
            pairs = [[query, doc.page_content] for doc, _ in candidates]
            relevance_scores = self.reranker.predict(pairs, show_progress_bar=False)

        # 2. 收集各维度分数
        all_scores = []
        max_rel = max(relevance_scores) if relevance_scores.max() > 0 else 1.0
        max_auth = max(self._get_authority_score(c.metadata)
                       for c, _ in candidates)
        max_rec = max(self._get_recency_score(c.metadata)
                      for c, _ in candidates)

        for i, (doc_or_item, fusion_score) in enumerate(candidates):
            # 提取 Document
            if isinstance(doc_or_item, Document):
                doc = doc_or_item
                metadata = doc.metadata
            else:
                # 元组形式
                doc = doc_or_item
                metadata = getattr(doc_or_item, "metadata", {})

            rel = float(relevance_scores[i])
            rec = self._get_recency_score(metadata)
            auth = self._get_authority_score(metadata)

            # 归一化 — 三维度统一用"候选池内相对归一化"(P2-6 修复)
            # 之前 recency 用了绝对值(不归一化),导致候选都新时 recency 几乎没区分度。
            norm_rel  = rel  / max_rel  if max_rel  > 0 else 0.0
            norm_rec  = rec  / max_rec  if max_rec  > 0 else 0.0
            norm_auth = auth / max_auth if max_auth > 0 else 0.0

            # 加权求和
            w = self.criteria_weights
            final_score = (
                w["relevance"] * norm_rel +
                w["recency"]   * norm_rec +
                w["authority"] * norm_auth
            )

            all_scores.append((doc, final_score, rel, rec, auth))

        # 3. 按最终分数降序
        all_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. article 级去重(按 source)
        seen = set()
        deduped = []
        for doc, fs, rel, rec, auth in all_scores:
            src = doc.metadata.get("source", "")
            if src and src in seen:
                continue
            seen.add(src)
            deduped.append((doc, fs))

        return deduped[:top_n]


# ==================== 多路检索器 ====================
class MultiPathRetriever:
    """
    多路检索器

    检索流程:
      1. BM25 检索 (中文分词,jieba)
      2. ChromaDB 向量检索 (bge-m3)
      3. 融合策略 (RRF / 加权 / 简单)
      4. 多准则重排序 (相关性 + 时效性 + 权威性)
      5. Article 级去重

    Args:
        vectorstore: langchain Chroma 向量库
        chunks: 所有 Document chunks (用于 BM25)
        fusion_strategy: 融合策略 ('rrf' | 'weighted' | 'simple')
        vector_weight: 加权融合时向量检索的权重 (BM25 权重 = 1 - vector_weight)
        top_k_vector: 向量检索返回数量
        top_k_bm25: BM25 返回数量
        reranker_top_k: 经过 rerank 后返回数量
        criteria_weights: 多准则权重, 如 {"relevance": 0.6, "recency": 0.25, "authority": 0.15}
    """

    # 普通类属性(下划线开头是 Python "private" 约定,不是 Pydantic 的 PrivateAttr)
    # 之前用 PrivateAttr 是死代码 — 类没继承 BaseModel,这些注解不生效。

    def __init__(
        self,
        vectorstore: Chroma,
        chunks: List[Document],
        fusion_strategy: str = "rrf",
        vector_weight: float = 0.6,
        top_k_vector: int = 30,
        top_k_bm25: int = 30,
        reranker_top_k: int = 5,
        criteria_weights: Optional[Dict[str, float]] = None,
    ):
        self._vectorstore = vectorstore
        self._chunks = chunks
        self._config = {
            "fusion_strategy": fusion_strategy,
            "vector_weight": vector_weight,
            "top_k_vector": top_k_vector,
            "top_k_bm25": top_k_bm25,
            "reranker_top_k": reranker_top_k,
        }

        # 构建 BM25 索引
        print("[MultiPathRetriever] 构建 BM25 索引...")
        self._bm25 = BM25Index()
        self._bm25.fit(chunks)

        # 初始化重排序器
        self._reranker = MultiCriteriaReranker(
            criteria_weights=criteria_weights,
        )

    @property
    def fusion_strategy(self) -> str:
        return self._config["fusion_strategy"]

    @fusion_strategy.setter
    def fusion_strategy(self, value: str) -> None:
        """运行时切换融合策略,无需重建 BM25 索引和重排序器。"""
        if value not in ("rrf", "weighted", "simple"):
            raise ValueError(f"Unknown fusion_strategy: {value}")
        self._config["fusion_strategy"] = value

    def retrieve(self,
                 query: str,
                 year_filter: Optional[Dict] = None,
                 journal_filter: Optional[Dict] = None,
                 vector_query: Optional[str] = None) -> List[Document]:
        """
        执行多路检索 + 重排序

        Args:
            query: 给 BM25 用的查询(可包含同义词扩展,例如 "metformin 二甲双胍")
            year_filter: 年份过滤
            journal_filter: 期刊过滤
            vector_query: 给向量检索的查询(默认 = query)。
                **重要**:实际场景应该传带 BGE instruction 的 query,
                例如 "Represent this question for searching relevant passages: {cleaned}",
                而 BM25 那边拿 cleaned + synonyms 的纯文本版本。

        Returns:
            List of Document, 按最终分数降序
        """
        strategy = self._config["fusion_strategy"]
        kv = self._config["top_k_vector"]
        kb = self._config["top_k_bm25"]
        vw = self._config["vector_weight"]

        # 向量检索默认用同一个 query(向后兼容)
        if vector_query is None:
            vector_query = query

        # ===== 1. BM25 检索(用带同义义的 query)=====
        bm25_results = self._bm25.search(
            query,
            top_k=kb,
            year_filter=year_filter,
            journal_filter=journal_filter,
        )

        # ===== 2. 向量检索(用带 BGE instruction 的 query)=====
        vector_results = self._vectorstore.similarity_search_with_score(
            vector_query,
            k=kv,
        )
        # Chroma 返回 [(Document, distance), ...], 距离转分数
        vector_results = [
            (doc, float(1.0 / (1.0 + dist)))  # distance -> 相似度分数
            for doc, dist in vector_results
        ]

        # ===== 3. 融合 =====
        if strategy == "rrf":
            fused = rrf_fusion([bm25_results, vector_results])
        elif strategy == "weighted":
            # [BM25_weight, vector_weight]
            fused = weighted_fusion(
                [bm25_results, vector_results],
                weights=[1.0 - vw, vw],
            )
        else:  # simple
            fused = simple_fusion([bm25_results, vector_results])

        print(f"  [融合] {strategy} → {len(fused)} 条候选")

        # ===== 4. 多准则重排序 =====
        reranked = self._reranker.rerank(
            query=query,
            candidates=fused,
            top_n=self._config["reranker_top_k"],
        )

        # 打印分析
        self._print_analysis(fused, reranked, bm25_results, vector_results)

        return [doc for doc, _ in reranked]

    def _print_analysis(self,
                        fused: List,
                        reranked: List,
                        bm25_raw: List,
                        vec_raw: List):
        """打印检索分析信息。

        用 _doc_key (pmid/source 优先) 做集合,而不是 id(doc):
        rrf_fusion/weighted_fusion 在合并跨路径的同 pmid 文档时,
        新建的 Document 对象 id 会变;但 pmid 不变 — 用 _doc_key 才反映"逻辑同源"
        (P3-9 修复)。
        """
        bm25_set = {_doc_key(d) for d, _ in bm25_raw}
        vec_set = {_doc_key(d) for d, _ in vec_raw}

        # 共同召回的文档数
        overlap = len(bm25_set & vec_set)
        bm25_only = len(bm25_set - vec_set)
        vec_only = len(vec_set - bm25_set)

        print(f"\n  检索分析:")
        print(f"    BM25 召回: {len(bm25_raw)} 条")
        print(f"    向量召回: {len(vec_raw)} 条")
        print(f"    两路共同命中: {overlap} 条")
        print(f"    仅 BM25 命中: {bm25_only} 条")
        print(f"    仅向量命中: {vec_only} 条")
        print(f"    融合({self._config['fusion_strategy']})后: {len(fused)} 条候选")
        print(f"    重排序后: {len(reranked)} 条")

        # top-3 预览
        if reranked:
            print(f"    Top-3:")
            for i, (doc, score) in enumerate(reranked[:3], 1):
                src = doc.metadata.get("source", "?")
                year = doc.metadata.get("year", "?")
                journal = doc.metadata.get("journal", "?")
                preview = doc.page_content[:60].replace("\n", " ")
                print(f"      [{i}] [{src}] {year} {journal[:20]}: {preview}...")

    def set_criteria_weights(self, weights: Dict[str, float]):
        """运行时更新多准则权重(P2-7:校验 key 集合,防止 KeyError 跑跑才炸)"""
        required = {"relevance", "recency", "authority"}
        if not isinstance(weights, dict):
            raise TypeError(f"weights 必须是 dict,收到 {type(weights).__name__}")
        missing = required - set(weights.keys())
        extra = set(weights.keys()) - required
        if missing or extra:
            raise ValueError(
                f"weights 必须是 {{relevance, recency, authority}} 三个 key,"
                f"缺 {missing or '无'},多 {extra or '无'}"
            )
        self._reranker.criteria_weights = weights
        print(f"权重已更新: {weights}")


# ==================== LangChain Retriever 适配器 ====================
class MultiPathLangChainRetriever(BaseRetriever):
    """
    LangChain BaseRetriever 接口适配器
    让 MultiPathRetriever 可以直接接入 RetrievalQA 等 langchain 组件

    注意:这个类真的继承自 Pydantic 的 BaseRetriever,所以 PrivateAttr() 是对的。
    (MultiPathRetriever 是不继承 BaseModel 的普通类,那 5 个 PrivateAttr 注解是死代码 — 见 P1-3。)
    """

    _mp_retriever: object = PrivateAttr()

    def __init__(self, mp_retriever: MultiPathRetriever, **kwargs):
        super().__init__(**kwargs)
        self._mp_retriever = mp_retriever

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._mp_retriever.retrieve(query)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._mp_retriever.retrieve(query)


# ==================== 独立测试 ====================
def _demo():
    """独立测试用,跑之前确保向量库已构建"""
    from rag_medical import load_documents, split_documents, load_vectorstore, get_embeddings

    print("=" * 60)
    print("多路检索器独立测试")
    print("=" * 60)

    # 加载 chunks(用于 BM25)
    docs = load_documents("./data/medical_papers")
    chunks = split_documents(docs)

    # 加载向量库
    embeddings = get_embeddings()
    vs = load_vectorstore(VECTOR_PERSIST_DIR, embeddings)

    # 构建多路检索器
    retriever = MultiPathRetriever(
        vectorstore=vs,
        chunks=chunks,
        fusion_strategy="rrf",
        top_k_vector=30,
        top_k_bm25=30,
        reranker_top_k=5,
    )

    # 测试查询
    test_queries = [
        "二甲双胍对心血管疾病的影响",
        "What is the effect of metformin on cardiovascular disease?",
        "PD-1 免疫疗法近五年的研究进展",
    ]

    print("\n" + "=" * 60)
    for q in test_queries:
        print(f"\n🔍 查询: {q}")
        results = retriever.retrieve(q)
        print(f"   → 返回 {len(results)} 条结果\n")
        if results:
            for i, doc in enumerate(results, 1):
                print(f"   [{i}] {doc.metadata.get('source', '?')} | "
                      f"{doc.metadata.get('year', '?')} | "
                      f"{doc.metadata.get('journal', '?')[:25]}")
                print(f"       {doc.page_content[:100].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    _demo()
