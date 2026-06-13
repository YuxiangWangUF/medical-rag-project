# Medical RAG Project · 医疗文献检索问答系统

> 基于 PubMed 医学文献的 RAG(检索增强生成)系统,本地 Ollama 驱动,零云端依赖。

## 🎯 项目目标

构建一个**本地可跑**的医疗文献问答系统:
- 从 PubMed 批量下载 XML
- 切分、向量化、入库
- 多路检索 + 多准则重排
- LLM 生成带引文的答案

## 📊 整体进度

| 阶段 | 内容 | 状态 | 测试 |
|---|---|---|---|
| 一 | 数据下载(PubMed XML) | ✅ | - |
| 二 | 数据分析(`data_analysis.py`) | ✅ | 20 |
| 三 | 文档切分(`chunk_processor.py`) | ✅ | 18 |
| 四 | 向量索引(`vector_indexer.py`) | ✅ | 16 |
| 五 | 查询增强(`query_enhancer.py`) | ✅ | 41 |
| 六·上 | 多路检索 + 重排(`multi_path_retriever.py`) | ✅ | 23 |
| 六·下 | 检索流水线(`stage6_retrieval_pipeline.py`) | ✅ | 34 |
| 七 | LLM 答案生成(集成在 stage6) | ✅ | 36 |
| **BM25 修复** | **tokenize + 医学自定义词典** | **✅** | **12** |
| **同义词 + vector_query 优化** | **BM25 拿 synonyms / 向量拿 BGE instruction** | **✅** | **+3** |
| **合计** | | **✅ 8 阶段** | **218 个测试,66s 全过** |

## 🏗️ 系统架构

```
┌─────────────┐
│  PubMed XML │  (3028 篇,作业用 50 篇)
└──────┬──────┘
       ↓
┌─────────────┐
│  chunk_processor │  → 3335 chunks
└──────┬──────┘
       ↓
┌──────────────────┐
│  vector_indexer  │  → ChromaDB (bge-m3 嵌入)
└──────┬───────────┘
       ↓
┌──────────────────────────────────────┐
│  stage6_retrieval_pipeline (主入口)   │
│                                       │
│  1. QueryEnhancer                      │
│     - 清洗 / 实体识别 / 同义词扩展     │
│     - 提取年份/期刊过滤条件            │
│                ↓                       │
│  2. MultiPathRetriever                 │
│     - 向量检索 (ChromaDB)              │
│     - BM25 关键词检索 (jieba 分词)     │
│     - RRF / 加权 / 简单融合            │
│                ↓                       │
│  3. MultiCriteriaReranker              │
│     - BGE-reranker-base 相关性         │
│     - 时效性 (年份线性衰减)            │
│     - 权威性 (期刊影响因子近似)        │
│     - 权重 0.6 / 0.25 / 0.15          │
│                ↓                       │
│  4. Ollama LLM (qwen3:8b)              │
│     - 医疗 prompt 模板                 │
│     - 引文标号 [1][2][3]               │
│     - 拒答保护 / 医学免责              │
└──────────────────────────────────────┘
       ↓
   📖 答案 + 📚 参考来源 (PubMed 链接)
```

## 🛠️ 技术栈

| 组件 | 选型 | 原因 |
|---|---|---|
| 向量嵌入 | **BAAI/bge-m3** (Ollama) | Ollama 可用,多语言支持 |
| 重排 | **BAAI/bge-reranker-base** (HuggingFace) | 业界标杆,中文友好 |
| 向量库 | **ChromaDB** | 轻量、本地持久化 |
| 关键词检索 | **BM25** (rank_bm25 + jieba) | 经典可靠,中文分词 |
| LLM | **qwen3:8b** (Ollama) | 本地可跑、中文强 |
| 框架 | **LangChain 1.x** | 流水线编排 |
| 测试 | **pytest** | 203 个测试 |

## 🚀 跑法

### 准备

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 启动 Ollama + 拉模型
ollama serve
ollama pull qwen3:8b
ollama pull bge-m3

# 3. 设置 HF 镜像(必须,bge-reranker-base 在国内下不动)
$env:HF_ENDPOINT="https://hf-mirror.com"  # PowerShell
export HF_ENDPOINT=https://hf-mirror.com   # bash
```

### 跑端到端 demo

```bash
$env:PYTHONIOENCODING="utf-8"  # Windows 必须
python stage6_retrieval_pipeline.py
# 交互模式:输入问题,回车;输入 batch 跑评估;输入 exit 退出
```

### 跑批量评估(50 query,不调 LLM,快)

```bash
python eval_50query.py
# 输出:output/stage6_eval_50query.json
```

### 跑全套测试

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:PYTHONIOENCODING="utf-8"
D:\Anaconda\envs\medical_rag\python.exe -m pytest tests/ -v
# 203 passed, ~72s
```

## 📈 评估结果(50 query 批量)

| 指标 | 数值 |
|---|---|
| 总查询 | 50 |
| 成功 | 50 (100%) |
| 失败 | 0 |
| 平均召回数 | 5.0 篇/查询 |
| 平均检索耗时 | 211.7 ms |
| 总评估耗时 | 10.6 秒 |
| Top-1 PMID 多样性 | 35/47 (74%) |
| Top-1 年份分布 | 2024 年占 11 次(recency 加权工作) |

详细 JSON 报告:`output/stage6_eval_50query.json`

## 📁 项目结构

```
medical_rag_project/
├── data/medical_papers/          # 原始 XML(3028 篇)
├── vector_db/                    # ChromaDB 持久化目录
├── output/                       # 评估报告 / 中间产物
│   ├── chunks.parquet            # 阶段三产物
│   └── stage6_eval_50query.json  # 50 query 评估结果
├── tests/                        # 203 个单元测试
│   ├── test_data_analysis.py
│   ├── test_chunk_processor.py
│   ├── test_vector_indexer.py
│   ├── test_query_enhancer.py
│   ├── test_multi_path_retriever.py     # 融合层 20 个
│   ├── test_retrieval_pipeline.py       # ⚠️ deprecated
│   ├── test_rag_medical.py
│   └── test_stage7_answer_generation.py # LLM 层 36 个
├── data_analysis.py              # 阶段二
├── chunk_processor.py            # 阶段三
├── vector_indexer.py             # 阶段四
├── query_enhancer.py             # 阶段五
├── multi_path_retriever.py       # 阶段六·上(融合层)
├── stage6_retrieval_pipeline.py  # 阶段六·下(端到端,主入口)
├── retrieval_pipeline.py         # ⚠️ DEPRECATED,保留只为兼容
├── rag_medical.py                # 工具集
├── eval_50query.py               # 50 query 评估脚本
└── README.md                     # 本文件
```

## 🔧 关键设计决策

### 1. 为什么用 bge-m3 而不是 bge-small-zh

- **bge-small-zh-v1.5 在 Ollama registry 不可用**(`ollama pull` 失败)
- **bge-m3 1.2GB,Ollama 0.24.0 可用,多语言支持更好**

### 2. 融合层用 RRF 而不是简单 concat

- RRF 保留排名信息(简单 concat 丢)
- 简单稳定,**k=60 是个常用平滑常数**

### 3. 多准则重排的相关性 + 时效性 + 权威性

- 权重 0.6/0.25/0.15
- 时效性按年份线性衰减(0.1/年)
- 权威性按 9 个常用期刊的近似影响因子

### 4. 端到端 prompt 模板

- 严格基于 context(不编造)
- 引用标号 [1][2]
- 不知道就说"未明确提及"
- 医学免责("建议咨询执业医师")

## 🐛 已知 P0 问题(已修)

1. **`multi_path_retriever.py` 融合 key 用 `id(doc)`** → 改用 `pmid` 优先,跨路径同源 doc 合并
2. **`weighted_fusion` 单路命中被稀释** → 改为只累加出现过的路径
3. **`MAX_FILES` 默认 None** → 改默认 50(全跑 3028 太慢)
4. **`rerank` 的 `for _, c in candidates` 反向** → `for c, _`
5. **`max_rec` 错调 `_get_authority_score`** → 改回 `_get_recency_score`
6. **BM25 中文 query 永远 0 命中** → 重写 `tokenize()`(中英分离)+ 注入 50+ 医学术语到 jieba 自定义词典

## 🩹 BM25 修复详解(2026-06-11)

**根因**(三层):
1. jieba 对中英混合文本**把中文按字切**("二甲双胍" → ["二甲", "胍"])
2. query 和 doc **分词不一致**,永远匹配不上
3. jieba 默认词典**没"二甲双胍"**等医学术语,即使切也切不出整词

**修复**:
- 重写 `tokenize(text)`:中英分离,英文用 `[a-z0-9-]+` regex,中文用 jieba,统一 lower + 过滤 1 字 + 过滤标点
- `_load_custom_dict()`:50+ 医学术语(`二甲双胍`、`心血管疾病`、`CAR-T`、`EGFR` 等)懒加载到 jieba
- BM25 索引 fit 阶段和 query 阶段**用同一个 tokenize**(避免不一致)

**效果**(全量 3028 篇 / 50 query):

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| BM25 召回率 | 0% | 100% | ✨ |
| 平均检索耗时 | 211.7ms | 164.5ms | **-22%** |
| 总耗时 | 10.6s | 8.2s | -23% |
| Top-1 多样性 | 74.5% | 76.6% | +2.1% |

**意义**:修复前 BM25 几乎不起作用,所有效果都靠向量检索;修复后 BM25 真的参与召回,RRF 融合才有意义,多路检索架构**名副其实**。

## 📊 LLM 答案示例

### Q: "What is ARNO protein?"

> 根据提供的医学文献,ARNO是一种包含三个结构域的蛋白质:N端的GAP结构域(GTPase-activating domain),中间的Sec7结构域(与酵母蛋白Sec7具有同源性),以及C端的pleckstrin homology结构域(PH结构域)[1]。文献指出,ARNO的催化活性(针对鸟苷酸交换的活性)位于Sec7结构域,并通过PH结构域与磷脂酰肌醇(3,4,5)-三磷酸的相互作用进行调控[1]。

📚 参考:
- PMID: 12969509 (2003) - https://pubmed.ncbi.nlm.nih.gov/12969509/

### Q: "二甲双胍对心血管疾病的影响"

> 根据提供的医学文献,我无法回答该问题。文献片段中未包含关于二甲双胍对心血管疾病影响的任何信息[1][2][3]。

*(LLM 正确拒答,符合 prompt 规则 — 召回的 5 篇不相关时,直接说"不知道")*

## 🧪 测试覆盖明细

| 测试文件 | 数量 | 覆盖 |
|---|---|---|
| test_data_analysis.py | 20 | XML 解析、字段提取、质量检查 |
| test_chunk_processor.py | 18 | 切分、token 估计 |
| test_vector_indexer.py | 16 | 嵌入、归一化、过滤 |
| test_query_enhancer.py | 41 | 清洗、实体、同义词、过滤 |
| test_multi_path_retriever.py | 23 | 融合 key / RRF / weighted / simple / vector_query |
| test_bm25_tokenize.py | 12 | BM25 命中(tokenize / 中英 / 医学词典) |
| test_retrieval_pipeline.py | 34 | ⚠️ deprecated 版的组件 |
| test_rag_medical.py | 16 | BGEReranker / HybridRetriever |
| test_stage7_answer_generation.py | 36 | LLM 集成层(端到端 mock) |
| **合计** | **218** | **~66s 全过** |

## 📝 后续可做

- [ ] 多模态(PubMed 论文的图表)
- [ ] 反馈学习(用户标注 → 排序优化)
- [ ] 医学本体(UMLS / MeSH)接入
- [ ] 多语言混合检索
- [ ] 流式 LLM 答案(token-by-token)
- [ ] Web UI(Gradio / Streamlit)
- [ ] FastAPI 服务化
- [ ] 真实数据集评估(BioASQ / TREC-COVID)
