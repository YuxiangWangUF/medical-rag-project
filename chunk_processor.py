# chunk_processor.py - 阶段三:文档解析与分割
#
# 流程:
#   1. XML → DataFrame
#   2. 基础清洗
#   3. 智能分割(整体不超阈值 → 不分割;超了 → 切分)
#   4. 输出 Parquet + JSONL + stats.json
#   5. 预览 + 质量验证
#
# 跑法:
#   python chunk_processor.py

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== 配置 ====================
DATA_PATH = "./data/medical_papers"
OUTPUT_DIR = "./output"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 256            # token 上限(超过则切分)
CHUNK_OVERLAP = 50
NO_SPLIT_THRESHOLD = 300    # token 数 ≤ 此值则整体不分割
TOKENIZER_LIMIT = 512       # bge 模型硬上限
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

# ==================== 1. XML → DataFrame ====================
def parse_xml(file_path: str) -> dict:
    """解析单个 XML,提取关键字段"""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")

    def get_text(tag_name, attr=None):
        tag = soup.find(tag_name, attr) if attr else soup.find(tag_name)
        return tag.get_text(" ", strip=True) if tag else ""

    # 正文(取所有 <p>)
    body_text = " ".join(
        p.get_text(" ", strip=True) for p in soup.find_all("p")
    )

    return {
        "pmid": get_text("article-id", {"pub-id-type": "pmid"}),
        "title": get_text("article-title"),
        "abstract": get_text("abstract"),
        "journal": get_text("journal-title"),
        "year": get_text("year"),
        "body_text": body_text,
    }


def load_to_dataframe(data_path: str) -> pd.DataFrame:
    """遍历目录,把 XML 解析为 DataFrame"""
    xml_files = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, f))
    xml_files = sorted(xml_files)
    print(f"[1/5] 找到 {len(xml_files)} 个 XML 文件")

    records, failed = [], []
    for i, fp in enumerate(xml_files, 1):
        try:
            d = parse_xml(fp)
            # 用 pmid 或 文件名作为 doc_id
            doc_id = d["pmid"] if d["pmid"] else os.path.splitext(os.path.basename(fp))[0]
            records.append({
                "doc_id": doc_id,
                "pmid": d["pmid"],
                "title": d["title"],
                "abstract": d["abstract"],
                "journal": d["journal"],
                "year": d["year"],
                "body_text": d["body_text"],
                "source_file": os.path.basename(fp),
            })
        except Exception as e:
            failed.append((os.path.basename(fp), str(e)))
        if i % 500 == 0:
            print(f"    解析进度: {i}/{len(xml_files)}")

    df = pd.DataFrame(records)
    print(f"    成功解析: {len(df)} 篇,失败: {len(failed)}")
    return df


# ==================== 2. 基础清洗 ====================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2/5] 基础清洗...")
    before = len(df)

    # 1. 去除完全空白行(title 和 abstract 都为空)
    df = df.dropna(subset=["title", "abstract"], how="all")
    df = df[~((df["title"].fillna("").str.strip() == "") &
              (df["abstract"].fillna("").str.strip() == ""))]
    print(f"  去除完全空白行: {before} → {len(df)}")

    # 2. 字段填充
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["pmid"] = df["pmid"].fillna("")
    df["journal"] = df["journal"].fillna("")
    df["year"] = df["year"].fillna("")

    # 3. 编码异常清理(理论上 XML 不会有,但兜底)
    for col in ["title", "abstract", "body_text"]:
        df[col] = df[col].astype(str).str.replace("�", "", regex=False)
        df[col] = df[col].str.replace("\x00", "", regex=False)
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    print(f"  清理编码异常 + 多余空白")

    # 4. doc_id 去重(同一 pmid 重复,保留第一个)
    dup = df.duplicated(subset=["doc_id"], keep="first").sum()
    df = df.drop_duplicates(subset=["doc_id"], keep="first")
    print(f"  去重(同一 pmid 重复): {dup} 个")

    # 5. 重新索引
    df = df.reset_index(drop=True)
    print(f"  清洗后总数: {len(df)} 篇")
    return df


# ==================== 3. 智能分割 ====================
class DocumentChunker:
    """智能分割器:
    - token 数 ≤ NO_SPLIT_THRESHOLD → 整体不分割(方案 b)
    - token 数 > NO_SPLIT_THRESHOLD → RecursiveCharacterTextSplitter 切分(方案 a)
    """

    def __init__(self, model_name: str, chunk_size: int, chunk_overlap: int,
                 no_split_threshold: int = NO_SPLIT_THRESHOLD):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.no_split_threshold = no_split_threshold
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", " ", ""],
            length_function=self._count_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_document(self, doc: dict) -> list:
        """对单篇文献切分"""
        # 拼接全文:标题 + 摘要 + 正文
        full_text = f"{doc['title']}\n\n{doc['abstract']}"
        if doc.get("body_text"):
            full_text += f"\n\n{doc['body_text']}"

        total_tokens = self._count_tokens(full_text)
        doc_id = doc["doc_id"]

        # 方案 b:整体不分割
        if total_tokens <= self.no_split_threshold:
            return [{
                "chunk_id": doc_id,
                "text": full_text,
                "doc_id": doc_id,
                "chunk_index": 0,
                "total_chunks": 1,
                "source_title": doc["title"],
                "token_count": total_tokens,
                "split_method": "no_split",
            }]

        # 方案 a:智能分割
        texts = self.splitter.split_text(full_text)
        chunks = []
        for i, text in enumerate(texts):
            # chunk_id 格式: doc_id + 序号(3 位补零)
            chunk_id = f"{doc_id}_chunk_{i:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(texts),
                "source_title": doc["title"],
                "token_count": self._count_tokens(text),
                "split_method": "recursive",
            })
        return chunks


def chunk_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 所有行切分,返回 chunks DataFrame"""
    print(f"\n[3/5] 智能分割(阈值: ≤ {NO_SPLIT_THRESHOLD} token 不分割)...")
    chunker = DocumentChunker(
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"  使用 tokenizer: {EMBEDDING_MODEL}")

    all_chunks = []
    no_split_count = 0
    split_count = 0

    docs = df.to_dict("records")
    for i, doc in enumerate(docs, 1):
        chunks = chunker.chunk_document(doc)
        if chunks and chunks[0]["split_method"] == "no_split":
            no_split_count += 1
        else:
            split_count += 1
        all_chunks.extend(chunks)
        if i % 500 == 0:
            print(f"    切分进度: {i}/{len(docs)}")

    chunks_df = pd.DataFrame(all_chunks)
    print(f"  共生成 {len(chunks_df)} 个 chunk")
    print(f"  整体不分割(方案 b): {no_split_count} 篇")
    print(f"  智能分割(方案 a): {split_count} 篇")
    return chunks_df


# ==================== 4. 保存 ====================
def save_outputs(df: pd.DataFrame, chunks_df: pd.DataFrame) -> dict:
    print(f"\n[4/5] 保存结果...")

    # 1. Parquet(主格式)
    parquet_path = os.path.join(OUTPUT_DIR, "chunks.parquet")
    chunks_df.to_parquet(parquet_path, index=False)
    print(f"  ✅ Parquet: {parquet_path}")

    # 2. JSONL(可读备份)
    jsonl_path = os.path.join(OUTPUT_DIR, "chunks.jsonl")
    chunks_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"  ✅ JSONL: {jsonl_path}")

    # 3. 统计信息
    stats = {
        "processed_date": datetime.now().isoformat(),
        "data_split": "oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23",
        "original_documents": len(df),
        "total_chunks": len(chunks_df),
        "chunks_per_doc": round(len(chunks_df) / len(df), 2) if len(df) > 0 else 0,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "no_split_threshold": NO_SPLIT_THRESHOLD,
        "tokenizer_limit": TOKENIZER_LIMIT,
        "embedding_model": EMBEDDING_MODEL,
        "output_file": parquet_path,
    }
    stats_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 统计: {stats_path}")

    return stats


# ==================== 5. 预览 + 质量验证 ====================
def preview(chunks_df: pd.DataFrame, n: int = 3):
    print(f"\n[预览] 前 {n} 个 chunk:")
    for idx, row in chunks_df.head(n).iterrows():
        print(f"\n  --- chunk_id: {row['chunk_id']} ---")
        print(f"  doc_id: {row['doc_id']}")
        print(f"  chunk_index: {row['chunk_index']} / {row['total_chunks']}")
        print(f"  source_title: {row['source_title'][:80]}")
        print(f"  token_count: {row['token_count']}")
        print(f"  split_method: {row['split_method']}")
        print(f"  text 前 200 字: {row['text'][:200]}...")


def quality_check(chunks_df: pd.DataFrame) -> dict:
    print(f"\n[5/5] 质量验证...")

    report = {}

    # 1. 总块数 + 分布
    report["total_chunks"] = len(chunks_df)
    print(f"  1. 总块数: {len(chunks_df)}")

    # 2. 超过模型限制的 chunk
    over_limit = chunks_df[chunks_df["token_count"] > TOKENIZER_LIMIT]
    report["chunks_over_tokenizer_limit"] = len(over_limit)
    report["over_limit_pct"] = round(len(over_limit) / len(chunks_df) * 100, 2)
    status = "✅" if len(over_limit) == 0 else "⚠️"
    print(f"  2. {status} 超模型上限 ({TOKENIZER_LIMIT}) 的 chunk: {len(over_limit)} ({report['over_limit_pct']}%)")

    # 3. 文本质量
    empty = chunks_df[chunks_df["text"].str.strip() == ""]
    very_short = chunks_df[chunks_df["text"].str.len() < 50]
    report["empty_chunks"] = len(empty)
    report["very_short_chunks_lt50"] = len(very_short)
    print(f"  3. ✅ 空 chunk: {len(empty)}, 极短(< 50 字): {len(very_short)}")

    # 4. 包含标题的 chunk(检查首块应该包含 source_title)
    multi = chunks_df[chunks_df["total_chunks"] > 1]
    if len(multi) > 0:
        first_chunks = multi[multi["chunk_index"] == 0]
        with_title = sum(
            1 for _, r in first_chunks.iterrows()
            if r["source_title"] and r["source_title"] in r["text"]
        )
        report["first_chunks_with_title"] = with_title
        report["first_chunks_total"] = len(first_chunks)
        pct = round(with_title / len(first_chunks) * 100, 2) if len(first_chunks) > 0 else 0
        print(f"  4. ✅ 分割文献首块含标题: {with_title}/{len(first_chunks)} ({pct}%)")

    # 5. 不完整截断检查(检查 chunk 文本是否在句子中间切断)
    # 简化:检查 chunk 末尾是否以句号/问号/段落结束
    end_pattern = re.compile(r"[.。?？!！\n]$")
    proper_end = chunks_df["text"].apply(lambda t: bool(end_pattern.search(t.strip())))
    report["chunks_ending_properly"] = int(proper_end.sum())
    report["chunks_ending_pct"] = round(proper_end.sum() / len(chunks_df) * 100, 2)
    print(f"  5. ✅ 末尾正常结束(标点/换行): {proper_end.sum()}/{len(chunks_df)} ({report['chunks_ending_pct']}%)")

    # 6. token_count 分布
    report["token_count_stats"] = {
        "min": int(chunks_df["token_count"].min()),
        "max": int(chunks_df["token_count"].max()),
        "mean": round(float(chunks_df["token_count"].mean()), 1),
        "median": float(chunks_df["token_count"].median()),
        "p95": round(float(chunks_df["token_count"].quantile(0.95)), 1),
        "p99": round(float(chunks_df["token_count"].quantile(0.99)), 1),
    }
    print(f"  6. token_count 分布: min={report['token_count_stats']['min']}, "
          f"max={report['token_count_stats']['max']}, "
          f"mean={report['token_count_stats']['mean']}, "
          f"p95={report['token_count_stats']['p95']}")

    # 7. 多块分割 vs 整体不分割
    docs_total_chunks = chunks_df.groupby("doc_id")["total_chunks"].first()
    multi_count = (docs_total_chunks > 1).sum()
    single_count = (docs_total_chunks == 1).sum()
    report["multi_chunk_documents"] = int(multi_count)
    report["single_chunk_documents"] = int(single_count)
    print(f"  7. 整体不分割(1 chunk)的文献: {single_count} 篇")
    print(f"     智能分割(多 chunk)的文献: {multi_count} 篇")

    # 8. 重叠部分检查(取一个多块样本,看相邻 chunk 是否有重叠文本)
    if multi_count > 0:
        sample_doc_id = docs_total_chunks[docs_total_chunks > 1].index[0]
        sample_chunks = chunks_df[chunks_df["doc_id"] == sample_doc_id].sort_values("chunk_index")
        if len(sample_chunks) >= 2:
            text1 = sample_chunks.iloc[0]["text"]
            text2 = sample_chunks.iloc[1]["text"]
            # 找尾部/头部重叠的字符数(简化:取后 100 字符 vs 前 100 字符的最长公共子串)
            tail = text1[-200:]
            head = text2[:200]
            # 找最长公共前缀(尾部 vs 头部)
            common_len = 0
            for i in range(min(len(tail), len(head))):
                if tail[i] == head[i]:
                    common_len += 1
                else:
                    break
            # 转成 token 估算
            sample_overlap_tokens = max(0, common_len * 2 // 3)  # 1 字符 ≈ 0.66 token,反向估
            report["sample_doc_id"] = sample_doc_id
            report["sample_overlap_chars"] = common_len
            report["sample_overlap_tokens_est"] = sample_overlap_tokens
            print(f"  8. 样本 doc {sample_doc_id}: 相邻 chunk 字符重叠 ≈ {common_len} (~{sample_overlap_tokens} token)")

    return report


# ==================== 主流程 ====================
def main():
    print("=" * 70)
    print("阶段三:文档解析与分割")
    print("=" * 70)

    # 1. 加载
    df = load_to_dataframe(DATA_PATH)

    # 2. 清洗
    df = clean_dataframe(df)

    # 3. 切分
    chunks_df = chunk_dataframe(df)

    # 4. 保存
    stats = save_outputs(df, chunks_df)

    # 5. 预览
    preview(chunks_df, n=3)

    # 6. 质量验证
    quality_report = quality_check(chunks_df)

    # 7. 保存质量报告
    qr_path = os.path.join(OUTPUT_DIR, "quality_report.json")
    with open(qr_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ 质量报告: {qr_path}")

    print("\n" + "=" * 70)
    print(f"✅ 完成!共生成 {len(chunks_df)} 个 chunk,保存到 {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
