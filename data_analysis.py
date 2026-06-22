# data_analysis.py v2 - 真实数据分析
# 改用 3028 个真实 XML 文献 + bge tokenizer
# 输出"《RAG 数据分析与设计说明》"到 ./数据分析与设计说明.md
#
# 跑法:
#   python data_analysis.py

import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

# ==================== 配置 ====================
DATA_PATH = "./data/medical_papers"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
TOKENIZER_LIMIT = 512           # bge-small-zh-v1.5 实际 max_seq_length
MAX_FILES = None                # 跑多少篇做样本(None = 全量 3028)
OUTPUT_REPORT = "./数据分析与设计说明.md"
LENGTH_PLOT_PATH = "./data/token_length_distribution.png"
# ===============================================

# 字体设置(尝试兼容 Mac/Win)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("RAG 数据分析与设计说明 — 真实数据版")
print("=" * 70)
print(f"数据源: {DATA_PATH}")
print(f"Embedding 模型: {EMBEDDING_MODEL}")
print(f"模型 token 上限: {TOKENIZER_LIMIT}")
print(f"样本数: {MAX_FILES if MAX_FILES else '全量'}")
print("=" * 70)

# ---- bge tokenizer 延迟加载 ----
# 用 lazy 加载,避免模块 import 时触发网络请求(内网环境会卡)
# 第一次调用 token_length_analysis() 时才真正加载
_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        print(f"\n[首次调用] 加载 tokenizer ({EMBEDDING_MODEL})...")
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        except Exception as e:
            print(f"[警告] tokenizer 加载失败({e}),后续 token 分析会回退到字符估算")
            _TOKENIZER = False  # 标记为失败,避免反复尝试
    return _TOKENIZER if _TOKENIZER else None


# ==================== 1. 数据加载 ====================
def parse_xml(file_path: str) -> dict:
    """简化版 XML 解析,只提取数据分析需要的字段"""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")

    def get_text(tag_name, attr=None):
        tag = soup.find(tag_name, attr) if attr else soup.find(tag_name)
        return tag.get_text(" ", strip=True) if tag else ""

    return {
        "title": get_text("article-title"),
        "abstract": get_text("abstract"),
        "pmid": get_text("article-id", {"pub-id-type": "pmid"}),
        "journal": get_text("journal-title"),
        "year": get_text("year"),
        "body_text": " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")),
    }


def load_documents(data_path: str, max_files=None) -> list:
    xml_files = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, f))
    xml_files = sorted(xml_files)
    if max_files is not None and len(xml_files) > max_files:
        xml_files = xml_files[:max_files]

    print(f"\n加载 XML 文件: {len(xml_files)} 个")
    docs, failed = [], []
    for i, fp in enumerate(xml_files, 1):
        try:
            d = parse_xml(fp)
            if d["title"] or d["abstract"]:
                docs.append(d)
        except Exception as e:
            failed.append((os.path.basename(fp), str(e)))
        if i % 100 == 0:
            print(f"  解析进度: {i}/{len(xml_files)}")
    print(f"成功解析: {len(docs)} 篇")
    if failed:
        print(f"失败: {len(failed)} 个")
    return docs


# ==================== 2. 数据结构分析 ====================
def analyze_structure(data: list) -> dict:
    print("\n" + "=" * 70)
    print("1. 数据结构分析")
    print("=" * 70)
    if not data:
        return {}

    total = len(data)
    fields = ["title", "abstract", "pmid", "journal", "year", "body_text"]
    field_stats = {}
    print(f"\n总文献数: {total}")
    print(f"\n字段完整性:")
    for f in fields:
        count = sum(1 for d in data if d.get(f, "").strip())
        missing_rate = (total - count) / total * 100
        field_stats[f] = {"count": count, "missing_rate": missing_rate}
        status = "⚠️" if missing_rate > 1 else "✅"
        print(f"  {status} {f:12s}: {count:4d}/{total} ({missing_rate:5.2f}% 缺失)")

    # 清洗策略
    print(f"\n清洗策略建议:")
    if "abstract" in field_stats:
        am = field_stats["abstract"]["missing_rate"]
        if am > 1:
            print(f"  abstract 缺失率 {am:.2f}% > 1% → 建议:丢弃缺失 abstract 的文献")
        else:
            print(f"  abstract 缺失率 {am:.2f}% ≤ 1% → 建议:填充空字符串(影响小)")

    return field_stats


# ==================== 3. 基础质量分析 ====================
def quality_check(data: list) -> dict:
    print("\n" + "=" * 70)
    print("2. 基础质量分析")
    print("=" * 70)
    total = len(data)

    # 极短文本
    short_threshold = 50
    short_items = [
        i for i, d in enumerate(data)
        if len((d.get("abstract", "") + " " + d.get("title", "")).strip()) < short_threshold
    ]
    short_rate = len(short_items) / total * 100
    print(f"\n极短文本 (< {short_threshold} 字符): {len(short_items)}/{total} ({short_rate:.2f}%)")
    if short_items:
        print(f"  示例 (前 3):")
        for i in short_items[:3]:
            t = data[i].get("title", "")[:40]
            print(f"    - {t}...")

    # 编码错误
    encoding_issues = sum(
        1 for d in data
        if "�" in str(d) or "\x00" in str(d)
    )
    print(f"\n编码异常 (含 � 或 \\x00): {encoding_issues}/{total}")

    return {"short": len(short_items), "encoding": encoding_issues}


# ==================== 4. 关键字段分析 ====================
def analyze_key_fields(data: list) -> dict:
    print("\n" + "=" * 70)
    print("3. 关键字段分析(元数据过滤器)")
    print("=" * 70)
    total = len(data)
    stats = {}

    # journal
    journals = [d["journal"] for d in data if d.get("journal", "").strip()]
    stats["journal"] = len(journals)
    print(f"\njournal 字段: {len(journals)}/{total} 有值")
    if journals:
        c = Counter(journals)
        print(f"  高频期刊 top 5:")
        for j, cnt in c.most_common(5):
            print(f"    - {j}: {cnt} 篇")
        print(f"  ✅ 可作为过滤器:检索'Nature 上的文献'等")

    # pub_date / year
    years = [d["year"] for d in data if d.get("year", "").strip()]
    stats["year"] = len(years)
    print(f"\nyear 字段: {len(years)}/{total} 有值")
    if years:
        years_int = [int(y) for y in years if y.isdigit()]
        if years_int:
            print(f"  范围: {min(years_int)} - {max(years_int)}")
            print(f"  ✅ 可作为过滤器:检索'近 5 年文献'等时间过滤")

    # pmid
    pmids = [d["pmid"] for d in data if d.get("pmid", "").strip()]
    stats["pmid"] = len(pmids)
    print(f"\npmid 字段: {len(pmids)}/{total} 有值")
    if pmids:
        print(f"  示例 PMID: {pmids[0]}")
        print(f"  链接示例: https://pubmed.ncbi.nlm.nih.gov/{pmids[0]}/")
        print(f"  ✅ 可作为原文追溯链接")

    # body_text(正文)是否普遍存在
    bodies = [d["body_text"] for d in data if d.get("body_text", "").strip()]
    stats["body_text"] = len(bodies)
    print(f"\nbody_text(正文)字段: {len(bodies)}/{total} 有值")
    if bodies:
        avg_len = np.mean([len(b) for b in bodies])
        print(f"  平均长度: {avg_len:.0f} 字符")
        print(f"  ✅ 可用于 RAG 检索(不只是 abstract)")

    return stats


# ==================== 5. 领域内容理解 ====================
def analyze_domain_content(data: list) -> dict:
    print("\n" + "=" * 70)
    print("4. 领域内容理解")
    print("=" * 70)
    total = len(data)

    # 分层抽样:按 title+abstract 字符长度排
    items_with_len = sorted(
        [(len((d.get("abstract", "") + d.get("title", ""))), d) for d in data],
        key=lambda x: x[0]
    )
    n = len(items_with_len)
    print(f"\n分层抽样(按 title+abstract 字符长度):")
    samples = {}
    for p, name in [(0.1, "短"), (0.5, "中"), (0.9, "长")]:
        idx = min(int(n * p), n - 1)
        s = items_with_len[idx]
        samples[name] = s[1]
        print(f"  {name}({int(p*100)}% 分位, 字符={s[0]}):")
        print(f"    title: {s[1].get('title', '')[:80]}")
        print(f"    abstract 前 200 字: {s[1].get('abstract', '')[:200]}")

    # IMRaD 结构
    print(f"\nIMRaD 结构检测(在 abstract 中):")
    imrad_keywords = {
        'background': ['background', 'introduction', '目的'],
        'methods':    ['method', 'approach', '材料与方法', '方法'],
        'results':    ['result', 'finding', '结果'],
        'conclusion': ['conclusion', '结论'],
    }
    structures = []
    for d in data:
        text = d.get("abstract", "").lower()
        has_bg = any(kw in text for kw in imrad_keywords['background'])
        has_md = any(kw in text for kw in imrad_keywords['methods'])
        has_rs = any(kw in text for kw in imrad_keywords['results'])
        has_cl = any(kw in text for kw in imrad_keywords['conclusion'])
        score = sum([has_bg, has_md, has_rs, has_cl])
        structures.append(score)
    avg_structure = np.mean(structures)
    full_imrad = sum(1 for s in structures if s == 4)
    print(f"  平均 IMRaD 完整度: {avg_structure:.2f}/4")
    print(f"  完整(4/4)文献: {full_imrad}/{total} ({full_imrad/total*100:.1f}%)")
    print(f"  0/4(无任何结构): {sum(1 for s in structures if s == 0)}/{total}")

    # 医学术语高频词
    print(f"\n医学缩写高频词(从 title + abstract 提取大写 2+ 字母的 token):")
    all_text = " ".join(
        (d.get("title", "") + " " + d.get("abstract", ""))
        for d in data
    )
    # 抽缩写(2+ 个大写字母)
    abbreviations = re.findall(r'\b[A-Z]{2,}\b', all_text)
    # 过滤常见英文词
    common_words = {
        "THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "WERE", "BEEN",
        "HAVE", "HAS", "WAS", "WERE", "ARE", "BUT", "NOT", "ALL", "CAN",
        "USA", "DNA", "RNA", "PCR",  # PCR/DNA/RNA 不算医学专有缩写
    }
    abbr_filtered = [a for a in abbreviations if a not in common_words and len(a) >= 3]
    c = Counter(abbr_filtered)
    print(f"  缩写 top 15:")
    for a, cnt in c.most_common(15):
        print(f"    - {a}: {cnt} 次")

    return {"imrad_avg": avg_structure, "samples": samples}


# ==================== 6. 文本特征量化分析 ====================
def token_length_analysis(data: list) -> np.ndarray:
    print("\n" + "=" * 70)
    print("5. 文本特征量化分析(以 token 为单位)")
    print("=" * 70)
    print(f"使用 tokenizer: {EMBEDDING_MODEL}")
    print(f"模型 max_seq_length: {TOKENIZER_LIMIT}")

    text_lengths = []
    tokenizer = _get_tokenizer()  # lazy load,失败时回退字符估算
    for d in data:
        # 用 title + abstract + body 前 2000 字(模拟实际 RAG 处理)
        text = d.get("title", "") + " " + d.get("abstract", "")
        if d.get("body_text"):
            text += " " + d["body_text"][:2000]
        if tokenizer:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        else:
            # 回退:1 token ≈ 1.5 字符
            tokens = list(range(0, int(len(text) / 1.5)))
        text_lengths.append(len(tokens))
    text_lengths = np.array(text_lengths)

    print(f"\nToken 长度统计(title + abstract + body 2000 字):")
    print(f"  最小:  {np.min(text_lengths)}")
    print(f"  最大:  {np.max(text_lengths)}")
    print(f"  平均:  {np.mean(text_lengths):.0f}")
    print(f"  中位:  {np.median(text_lengths):.0f}")
    print(f"  90% 分位: {np.percentile(text_lengths, 90):.0f}")
    print(f"  95% 分位: {np.percentile(text_lengths, 95):.0f}")
    print(f"  99% 分位: {np.percentile(text_lengths, 99):.0f}")

    # 跟模型上限对比
    over_limit = sum(1 for l in text_lengths if l > TOKENIZER_LIMIT)
    print(f"\n超过模型上限 ({TOKENIZER_LIMIT}) 的: {over_limit}/{len(text_lengths)} ({over_limit/len(text_lengths)*100:.1f}%)")

    # 绘图
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(text_lengths, bins=30, edgecolor='black', color='steelblue')
        plt.axvline(TOKENIZER_LIMIT, color='r', linestyle='--', label=f'Limit={TOKENIZER_LIMIT}')
        plt.axvline(np.percentile(text_lengths, 95), color='orange', linestyle='--', label=f'95%分位={int(np.percentile(text_lengths, 95))}')
        plt.xlabel('Token 长度')
        plt.ylabel('频次')
        plt.title('文本 Token 长度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.boxplot(text_lengths)
        plt.axhline(TOKENIZER_LIMIT, color='r', linestyle='--', label=f'Limit={TOKENIZER_LIMIT}')
        plt.ylabel('Token 长度')
        plt.title('箱线图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("./data", exist_ok=True)
        plt.savefig(LENGTH_PLOT_PATH, dpi=150, bbox_inches='tight')
        print(f"\n长度分布图已保存: {LENGTH_PLOT_PATH}")
    except Exception as e:
        print(f"\n绘图失败: {e}")

    return text_lengths


# ==================== 7. 制定分割策略 ====================
def recommend_split_strategy(text_lengths: np.ndarray) -> dict:
    print("\n" + "=" * 70)
    print("6. 分割策略制定")
    print("=" * 70)
    p95 = np.percentile(text_lengths, 95)
    p99 = np.percentile(text_lengths, 99)
    p90 = np.percentile(text_lengths, 90)

    print(f"\n数据特征:")
    print(f"  90% 分位: {p90:.0f} tokens")
    print(f"  95% 分位: {p95:.0f} tokens")
    print(f"  99% 分位: {p99:.0f} tokens")
    print(f"  模型上限: {TOKENIZER_LIMIT} tokens")

    result = {"p95": p95, "p99": p99, "p90": p90}

    if p95 <= TOKENIZER_LIMIT:
        strategy = "整体不分割"
        print(f"\n📌 推荐策略: {strategy}")
        print(f"  理由: 95% 分位 ({p95:.0f}) ≤ {TOKENIZER_LIMIT},大部分文本无需分割")
        print(f"  方法: 不使用 Splitter,直接将'标题+摘要+正文'作为一个 Document")
        print(f"  chunk_size: {TOKENIZER_LIMIT}")
        result.update({"strategy": strategy, "chunk_size": TOKENIZER_LIMIT, "chunk_overlap": 0})
    elif p99 <= TOKENIZER_LIMIT * 2:
        chunk_size = min(TOKENIZER_LIMIT - 50, int(p95))
        chunk_overlap = int(chunk_size * 0.2)
        strategy = "重叠滑动窗口"
        print(f"\n📌 推荐策略: {strategy}")
        print(f"  理由: 存在少量长尾文档(> {TOKENIZER_LIMIT}),但 99% 分位 ({p99:.0f}) ≤ {TOKENIZER_LIMIT * 2}")
        print(f"  方法: RecursiveCharacterTextSplitter")
        print(f"  chunk_size: {chunk_size} tokens")
        print(f"  chunk_overlap: {chunk_overlap} tokens (20%)")
        result.update({"strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
    else:
        chunk_size = 300
        chunk_overlap = 50
        strategy = "按语义章节分割 + 重叠滑动窗口"
        print(f"\n📌 推荐策略: {strategy}")
        print(f"  理由: 99% 分位 ({p99:.0f}) > {TOKENIZER_LIMIT * 2},长尾过长")
        print(f"  方法: 按 BACKGROUND/METHODS/RESULTS/CONCLUSIONS 章节切分,再用 RecursiveCharacterTextSplitter 兜底")
        result.update({"strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})

    print(f"\n✅ 最终选型: {strategy}")
    return result


# ==================== 8. 主流程 ====================
def main():
    # 1. 加载
    data = load_documents(DATA_PATH, max_files=MAX_FILES)
    if not data:
        print("❌ 没有解析到数据,退出")
        return

    # 2-5. 分析
    field_stats = analyze_structure(data)
    quality_stats = quality_check(data)
    key_stats = analyze_key_fields(data)
    domain_stats = analyze_domain_content(data)

    # 6-7. Token + 策略
    lengths = token_length_analysis(data)
    strategy_info = recommend_split_strategy(lengths)

    print("\n" + "=" * 70)
    print("✅ 数据分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
