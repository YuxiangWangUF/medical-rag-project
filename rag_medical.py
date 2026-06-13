# rag_medical.py v2 + rerank
# 改进点(vs v1):
#   1. XML 不再当纯文本加载,用 BeautifulSoup 解析 JATS 结构,只保留干净文本 + 元数据
#   2. 医疗专用 prompt:角色约束 + 引用编号 + 拒诊兜底 + "不知道就说不知道"
#   3. chunk_size 改用 bge 的 tokenizer 精确控制,避免中文超 512 token 被截断
#   4. 检索:BM25 + dense hybrid 粗召回(RETRIEVER_K) + bge-reranker 精排(TOP_K)
#   5. Embedding 单例 + 错误处理 + 引用来源打印
#
# 跑法:
#   pip install -r requirements.txt
#   ollama serve   # 另开一个终端
#   ollama pull qwen3:8b
#   python rag_medical.py

import os
import sys
from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from ollama_batch_embeddings import OllamaEmbeddingsBatch
# langchain 1.0+ 推荐用 langchain-chroma(独立包),不是 langchain_community
from langchain_chroma import Chroma
# langchain 1.0+:EnsembleRetriever 移到了 langchain_classic.retrievers
# BM25Retriever 还在 community(有 deprecation warning 但能用)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_classic.chains import RetrievalQA
from sentence_transformers import CrossEncoder
from pydantic import PrivateAttr

# ==================== 配置区 ====================
DATA_PATH = "./data/medical_papers"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "medical_papers_v2_bgem3"  # bge-m3 重建后的新库(1024维)
MODEL_NAME = "qwen3:8b"
EMBEDDING_MODEL = "bge-m3"  # Ollama 上的 embedding 模型(本地可用)
CHUNK_SIZE = 256            # token 数(不是字符),bge 上限 512,留 buffer
CHUNK_OVERLAP = 50          # token 数
TOP_K = 5                   # 最终返回给 LLM 的 top-k(经过 rerank 之后)
RETRIEVER_K = 40            # 粗召回阶段给 rerank 多少候选(建议 TOP_K 的 2-3 倍)
RERANK_MODEL = "BAAI/bge-reranker-base"   # 重排模型,~568MB
HYBRID_WEIGHTS = [0.7, 0.3] # [BM25, dense] 权重(BM25 主导,关键词匹配更准)
MAX_FILES = 50             # 限制解析的 XML 数量(None=全跑);作业演示 50 篇足够
# ===============================================

# ---- 简单的字符级 token 计数(替代 bge tokenizer) ----
# bge-small-zh 中文约 2字符/token; bge-m3 类似
# 用 jieba 分词更准确
import jieba
jieba.initialize()

def _token_len(text: str) -> int:
    """用 jieba 分词估算 token 数,避免按字符估算超长被截断"""
    return len(list(jieba.cut(text)))


# ==================== 1. XML 解析 ====================
def parse_pubmed_xml(file_path: str) -> Document:
    """
    解析 PubMed Central 的 JATS XML,提取结构化文本 + 元数据。
    不再用 TextLoader 把 XML 当纯文本塞进去 — 那样会把标签、命名空间、
    `&#x000fa;` 实体等噪音带进向量库,严重拖检索质量。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")

    # --- 元数据 ---
    title_tag = soup.find("article-title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    pmid_tag = soup.find("article-id", {"pub-id-type": "pmid"})
    pmid = pmid_tag.get_text(strip=True) if pmid_tag else ""

    journal_tag = soup.find("journal-title")
    journal = journal_tag.get_text(strip=True) if journal_tag else ""

    pub_date_tag = soup.find("pub-date")
    year = ""
    if pub_date_tag:
        y = pub_date_tag.find("year")
        if y:
            year = y.get_text(strip=True)

    # --- 正文 ---
    abstract_tag = soup.find("abstract")
    abstract = abstract_tag.get_text(" ", strip=True) if abstract_tag else ""

    # 解析 body 内每个 <sec>,保留章节结构
    body = soup.find("body")
    sections_text = []
    if body:
        for sec in body.find_all("sec", recursive=True):
            sec_title_tag = sec.find("title")
            sec_title = sec_title_tag.get_text(strip=True) if sec_title_tag else ""
            paragraphs = [p.get_text(" ", strip=True) for p in sec.find_all("p")]
            if not paragraphs:
                continue
            sec_body = "\n".join(paragraphs)
            sections_text.append(
                f"## {sec_title}\n{sec_body}" if sec_title else sec_body
            )

    # 拼成结构化 Markdown
    parts = []
    if title:
        parts.append(f"# {title}")
    if abstract:
        parts.append(f"## Abstract\n{abstract}")
    parts.extend(sections_text)
    full_text = "\n\n".join(parts)

    if not full_text.strip():
        return None  # 跳过空文档

    return Document(
        page_content=full_text,
        metadata={
            "pmid": pmid,
            "journal": journal,
            "year": year,
            "source": os.path.basename(file_path),
        },
    )


def load_documents(data_path: str) -> List[Document]:
    """遍历目录解析所有 .xml 为 Document"""
    xml_files = []
    for root, _, files in os.walk(data_path):
        for fname in files:
            if fname.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, fname))

    # 限制文件数,作业演示不需要跑全量
    if MAX_FILES is not None and len(xml_files) > MAX_FILES:
        print(f"⚠️  共 {len(xml_files)} 个文件,按 MAX_FILES={MAX_FILES} 限制只解析前 {MAX_FILES} 个")
        xml_files = sorted(xml_files)[:MAX_FILES]   # 排序保证可复现
    else:
        print(f"找到 {len(xml_files)} 个 XML 文件")

    docs, failed = [], []
    for i, fp in enumerate(xml_files, 1):
        try:
            doc = parse_pubmed_xml(fp)
            if doc:
                docs.append(doc)
        except Exception as e:
            failed.append((os.path.basename(fp), str(e)))
        if i % 50 == 0:
            print(f"  解析进度: {i}/{len(xml_files)}")

    print(f"成功解析 {len(docs)} 篇文献")
    if failed:
        print(f"⚠️ 解析失败 {len(failed)} 个:")
        for name, err in failed[:5]:
            print(f"   - {name}: {err[:80]}")
        if len(failed) > 5:
            print(f"   ... 还有 {len(failed) - 5} 个")
    return docs


# ==================== 2. 切分 ====================
def split_documents(documents: List[Document]) -> List[Document]:
    """按 token 切分(不是按字符)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
        length_function=_token_len,   # 关键:用 bge tokenizer 估长度
    )
    chunks = splitter.split_documents(documents)

    # 二次切分:有些段落因为没合适的 separator,会被 splitter 整个塞进 chunk
    # 超过 400 tokens 的 chunk 在 bge encode 时会被截断到 512,语义有损
    # 所以强制把它们再切一次
    safe_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""],
        length_function=_token_len,
    )
    safe_chunks = []
    overlong = 0
    for c in chunks:
        if _token_len(c.page_content) > 400:
            overlong += 1
            safe_chunks.extend(safe_splitter.split_documents([c]))
        else:
            safe_chunks.append(c)

    print(f"切分得到 {len(chunks)} 个 chunk "
          f"(其中 {overlong} 个超长做了二次切分 → {len(safe_chunks)} 个)")
    return safe_chunks


# ==================== 3. Embedding / 向量库 ====================
_EMBEDDINGS = None

def get_embeddings() -> OllamaEmbeddingsBatch:
    """单例:通过 Ollama 使用 bge-m3 embedding (支持批量 embed_documents)"""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        print(f"加载 embedding 模型 ({EMBEDDING_MODEL}) via Ollama...")
        _EMBEDDINGS = OllamaEmbeddingsBatch(model=EMBEDDING_MODEL)
        # warmup first call (~4s) so subsequent calls are fast
        _EMBEDDINGS.embed_query("warmup")
    return _EMBEDDINGS


def build_vectorstore(chunks, persist_dir: str, embeddings) -> Chroma:
    print("构建向量库...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )
    # langchain_chroma 1.0+ 已自动持久化(传了 persist_directory),不需要再调 vs.persist()
    print(f"✅ 向量库已保存: {persist_dir}")
    return vs


def load_vectorstore(persist_dir: str, embeddings) -> Chroma:
    print("加载已有向量库...")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ==================== 4. Hybrid + Rerank Retriever ====================
class BGEReranker:
    """基于 sentence-transformers CrossEncoder 的 BGE reranker。
    CrossEncoder 比 bi-encoder(bge embedding)更准:它把 query 和 doc 拼在一起
    跑一次完整 forward,能捕捉 query 和 doc 之间的细粒度交互。
    缺点:慢,只能对召回后的几十~几百个候选做精排。
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        print(f"加载 rerank 模型 ({model_name})...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_n: int = TOP_K):
        """对所有 chunk 打分,按分数降序排。
        返回值:
          - top_n=数字:返回 [Document, ...] (取前 top_n)
          - top_n=None:返回 [(score, Document), ...] (全部,让调用方自己处理)
        """
        if not docs:
            return [] if top_n is not None else []
        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        # 按 rerank 分数降序排
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        if top_n is None:
            return ranked  # 返回 (score, doc) 对,让调用方自己取 top_n + dedup
        return [doc for _, doc in ranked[:top_n]]


class HybridRerankRetriever(BaseRetriever):
    """
    三段式检索:
      1) BM25 + dense hybrid(粗召回,给 rerank 更多候选)
      2) BGE-reranker 二次重排(精排)
      3) 取 top_n 给 LLM
    """
    _ensemble: object = PrivateAttr()
    _reranker: object = PrivateAttr()
    _top_n: int = PrivateAttr()

    def __init__(self, ensemble_retriever, reranker, top_n: int = TOP_K):
        super().__init__()
        self._ensemble = ensemble_retriever
        self._reranker = reranker
        self._top_n = top_n

    def _get_relevant_documents(self, query, *, run_manager=None):
        # 1) hybrid 粗召回
        raw_chunks = self._ensemble.invoke(query)
        if not raw_chunks:
            return []
        # 2) rerank 给所有 chunk 打分(不立即取 top_n,因为后面要 article 级去重)
        scored = self._reranker.rerank(query, raw_chunks, top_n=None)
        # 3) article 级去重 — 关键:按 rerank 分数保留每个 source 分数最高的 chunk
        #    (如果按召回顺序保留,会把 rerank 分数低的留下、把分数高的丢掉)
        seen_sources = set()
        deduped_scored = []
        for score, doc in scored:
            src = doc.metadata.get("source", "")
            if not src or src in seen_sources:
                continue
            seen_sources.add(src)
            deduped_scored.append((score, doc))
        # 4) 取 top_n
        final = [doc for _, doc in deduped_scored[:self._top_n]]

        # === debug ===
        target = "PMC212319"
        in_rerank_pool = any(target in d.metadata.get("source", "") for _, d in scored)
        in_dedup = any(target in d.metadata.get("source", "") for _, d in deduped_scored)
        in_final = any(target in d.metadata.get("source", "") for d in final)
        print(f"  [debug] 召回 {len(raw_chunks)} chunk → rerank 打分 {len(scored)} → "
              f"按 article 去重 {len(deduped_scored)} 篇 → top-{self._top_n} = {len(final)}")
        print(f"  [debug] PMC212319 追踪:rerank池 {'在' if in_rerank_pool else '不在'} | "
              f"去重后 {'在' if in_dedup else '不在'} | top-{self._top_n} {'在' if in_final else '不在'}")
        if in_dedup:
            # 找到 PMC212319 的去重排名和分数
            for rank, (s, d) in enumerate(deduped_scored, 1):
                if target in d.metadata.get("source", ""):
                    print(f"  [debug] PMC212319 去重后排名: {rank}/{len(deduped_scored)}, rerank 分数: {s:.4f}")
                    break
        # === end debug ===

        return final


def build_hybrid_retriever(chunks, vectorstore, llm) -> BaseRetriever:
    """
    三段式检索:
      1) MultiQueryRetriever:LLM 把用户 query 改写成 3 个不同表述,扩召回
      2) hybrid (BM25 + dense):每个改写 query 都跑一遍粗召回
      3) bge-reranker:对所有候选精排,取 top_n

    为什么需要 query 改写:
    - 短 query + 多义专名(比如 "ARNO" 既可能是蛋白也可能是意大利人名)BM25/dense 都召回差
    - LLM 改写后多关键词,能精准命中核心文章
    """
    print(f"构建 multi-query + hybrid + rerank retriever...")
    # 内层:hybrid 粗召回
    bm25 = BM25Retriever.from_documents(chunks, k=RETRIEVER_K)
    dense = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=HYBRID_WEIGHTS,
    )
    # 外层:MultiQueryRetriever 包装 ensemble,自动改写 query
    multi_query = MultiQueryRetriever.from_llm(
        retriever=ensemble,
        llm=llm,
    )
    reranker = BGEReranker()
    return HybridRerankRetriever(
        ensemble_retriever=multi_query,
        reranker=reranker,
        top_n=TOP_K,
    )


# ==================== 5. 医疗专用 Prompt ====================
MEDICAL_PROMPT = """你是一名严谨的医学文献助手,严格基于下面提供的医学文献片段回答问题。

【回答规则】
1. 只能基于下方 context 的内容回答,不得编造任何数据、结论或参考文献
2. 引用具体内容时,使用 [1][2][3] 的格式标注来源编号(对应"文献片段"前的编号)
3. 如果 context 不包含答案,直接回答"根据提供的医学文献,我无法回答该问题",不要推测
4. 不给出诊断、用药剂量或治疗方案;涉及个体医疗建议时,建议读者咨询执业医师
5. 区分"文献报道"和"临床建议",措辞要保守、严谨

【文献片段】
{context}

【用户问题】
{question}

【回答】"""


# ==================== 6. 辅助:打印引用 ====================
def print_sources(source_docs):
    """打印引用来源(按 PMID 去重)"""
    if not source_docs:
        return
    print("\n📚 参考来源:")
    seen = set()
    for doc in source_docs:
        pmid = doc.metadata.get("pmid", "未知")
        src = doc.metadata.get("source", "未知")
        year = doc.metadata.get("year", "")
        journal = doc.metadata.get("journal", "")
        if pmid in seen:
            continue
        seen.add(pmid)
        line = f"  - PMID: {pmid} | 文件: {src}"
        if year:
            line += f" | 年份: {year}"
        if journal:
            line += f" | 期刊: {journal}"
        print(line)
        if pmid != "未知":
            print(f"    链接: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")


# ==================== 7. 主流程 ====================
def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据路径不存在: {DATA_PATH}")
        return

    # 加载并切分(每次启动都跑一次,数据量小,几秒搞定;BM25 需要原始 chunks)
    raw_docs = load_documents(DATA_PATH)
    if not raw_docs:
        print("❌ 没有解析到有效文档,请检查数据目录。")
        return
    chunks = split_documents(raw_docs)

    embeddings = get_embeddings()

    # 向量库(可复用)
    if os.path.exists(PERSIST_DIR):
        print("发现已有向量库,直接加载...")
        vectorstore = load_vectorstore(PERSIST_DIR, embeddings)
    else:
        vectorstore = build_vectorstore(chunks, PERSIST_DIR, embeddings)

    # LLM(提前到 retriever 之前,因为 MultiQueryRetriever 需要 LLM 改写 query)
    print(f"连接本地模型: {MODEL_NAME}...")
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.3, num_predict=1024)

    # Hybrid + MultiQuery + rerank retriever
    retriever = build_hybrid_retriever(chunks, vectorstore, llm)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=MEDICAL_PROMPT,
                input_variables=["context", "question"],
            )
        },
    )

    print("\n" + "=" * 55)
    print("🏥 医疗文献问答系统 v2")
    print("   - Hybrid retrieval (BM25 + dense)")
    print("   - 引用标注 + 拒诊兜底")
    print("   输入 'exit' 退出")
    print("=" * 55)

    while True:
        try:
            query = input("\n请输入您的问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见")
            break

        if query.lower() == "exit":
            print("👋 再见")
            break
        if not query:
            continue

        print("\n🔍 检索中...\n")
        try:
            result = qa_chain.invoke({"query": query})
            print(result["result"])
            print_sources(result.get("source_documents", []))
        except Exception as e:
            print(f"\n❌ 查询失败: {type(e).__name__}: {e}")
            err = str(e).lower()
            if "connection" in err or "refused" in err:
                print("💡 提示: 请先启动 Ollama 服务 (ollama serve)")
            elif "model" in err and "not found" in err:
                print(f"💡 提示: 请先拉取模型 (ollama pull {MODEL_NAME})")


if __name__ == "__main__":
    main()
