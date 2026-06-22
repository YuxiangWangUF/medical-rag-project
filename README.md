# Medical RAG Project · 医疗文献检索问答系统

> 基于 PubMed 医学文献的 RAG(检索增强生成)系统,本地 Ollama 驱动,零云端依赖。

![Tests](https://img.shields.io/badge/tests-331%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Ollama%20qwen3%3A8b-orange)
![Stage](https://img.shields.io/badge/stage-8%2F8-success)

## 项目目标

构建一个**本地可跑**的医疗文献问答系统:
- 从 PubMed 批量下载 XML
- 切分、向量化、入库
- 多路检索 + 多准则重排
- LLM 生成带引文的答案
- 全流程可观测、可测试、可 CI

## 整体进度

| 阶段 | 内容 | 状态 | 测试 |
|---|---|---|---|
| 一 | 数据下载(PubMed XML) | 完成 | - |
| 二 | 数据分析(`data_analysis.py`) | 完成 | 20 |
| 三 | 文档切分(`chunk_processor.py`) | 完成 | 18 |
| 四 | 向量索引(`vector_indexer.py`) | 完成 | 16 |
| 五 | 查询增强(`query_enhancer.py`) | 完成 | 41 |
| 六·上 | 多路检索 + 重排(`multi_path_retriever.py`) | 完成 | 23 |
| 六·下 | 检索流水线(`stage6_retrieval_pipeline.py`) | 完成 | 34 |
| 七 | LLM 答案生成(集成在 stage6) | 完成 | 36 |
| **BM25 修复** | **tokenize + 医学自定义词典** | **完成** | **12** |
| **同义词 + vector_query 优化** | **BM25 拿 synonyms / 向量拿 BGE instruction** | **完成** | **+3** |
| 八·1 | Context Assembler + Prompt Templates | **完成** | **48** |
| 八·2 | LLM Generator + Medical Pipeline | **完成** | **62** |
| 八·3 | **并发限流 + LRU 缓存 + 流式 + metrics jsonl + TypedDict + CI** | **完成** | **+20** |
| **合计** | | **完成 8 阶段** | **331 个测试,~30s 全过** |

## 系统架构

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
│  4. MedicalGenerationPipeline(八·2)    │
│     - ContextAssembler 干净上下文      │
│     - 证据评估 → 草稿 → 审查 → 最终   │
│     - 后处理:加引用 + 加免责声明       │
└──────────────────────────────────────┘
       ↓
    答案 + 参考来源 (PubMed 链接)
       ↓
┌──────────────────────────────────────┐
│  八·3: LLM Generator                  │
│     - Semaphore 并发限流(防打爆 Ollama)│
│     - LRU 缓存(重复 query 命中)       │
│     - 流式生成(token-by-token)        │
│     - metrics jsonl 持久化            │
│     - TypedDict 类型契约              │
└──────────────────────────────────────┘
```

## 技术栈

| 组件 | 选型 | 原因 |
|---|---|---|
| 向量嵌入 | **BAAI/bge-m3** (Ollama) | Ollama 可用,多语言支持 |
| 重排 | **BAAI/bge-reranker-base** (HuggingFace) | 业界标杆,中文友好 |
| 向量库 | **ChromaDB** | 轻量、本地持久化 |
| 关键词检索 | **BM25** (rank_bm25 + jieba) | 经典可靠,中文分词 |
| LLM | **qwen3:8b** (Ollama) | 本地可跑、中文强 |
| 框架 | **LangChain 1.x** | 流水线编排 |
| 测试 | **pytest** + fixtures + conftest | 331 个测试 |
| CI | **GitHub Actions**(ubuntu + windows × py3.10/3.12) | 矩阵测试 |
| 类型 | **TypedDict** | 零开销的 dict 类型契约 |

## 跑法

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
# 真实 LLM 模式
$env:PYTHONIOENCODING="utf-8"  # Windows 必须
python stage6_retrieval_pipeline.py
# 交互模式:输入问题,回车;输入 batch 跑评估;输入 exit 退出

# 无 LLM 模式(给无 Ollama 环境 / CI 用)— mock LLM 验证流水线结构
python stage8_e2e_demo.py --no-llm --query "测试问题"

# 只跑一个 query,持久化 metrics
python stage8_e2e_demo.py --query "二甲双胍对心血管的影响" --metrics-jsonl ./metrics.jsonl
```

### CLI 参数(stage8_e2e_demo.py)

| 参数 | 作用 |
|---|---|
| `--no-llm` | 跳过真实 LLM,用 mock 验证流水线 |
| `--query` / `-q` | 只跑这个 query(默认跑全部 3 个) |
| `--metrics-jsonl <path>` | 持久化 metrics 到 jsonl |
| `--model <name>` | 指定 Ollama 模型(默认 qwen3:8b) |
| `--no-review` | 跳过批判性审查阶段(加速) |

### 跑批量评估(50 query,不调 LLM,快)

```bash
python eval_50query.py
# 输出:output/stage6_eval_50query.json
```

### 跑全套测试

```bash
# 设置离线(避免 HF 自动拉模型)
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:PYTHONIOENCODING="utf-8"

D:\Anaconda\envs\medical_rag\python.exe -m pytest tests/ -v
# 331 passed, ~30s
```

## 评估结果(50 query 批量)

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

## 项目结构

```
medical_rag_project/
├── data/medical_papers/          # 原始 XML(3028 篇)
├── chroma_db/                    # ChromaDB 持久化目录
├── output/                       # 评估报告 / 中间产物
│   ├── chunks.parquet            # 阶段三产物
│   └── stage6_eval_50query.json  # 50 query 评估结果
├── tests/                        # 331 个单元测试
│   ├── conftest.py               # 共享 fixtures(八·3 新增)
│   ├── test_data_analysis.py     # 20
│   ├── test_chunk_processor.py   # 18
│   ├── test_vector_indexer.py    # 16
│   ├── test_query_enhancer.py    # 41
│   ├── test_multi_path_retriever.py     # 23
│   ├── test_bm25_tokenize.py     # 12
│   ├── test_stage6_retrieval_pipeline.py  # 38
│   ├── test_stage8_end_to_end.py # 6
│   ├── test_context_assembler.py # 28
│   ├── test_prompt_templates.py  # 20
│   ├── test_llm_generator.py     # 37 (八·2 + 八·3 新增)
│   ├── test_medical_pipeline.py  # 25 (八·2 + 八·3 新增)
│   ├── test_stage8_cli.py        # 10 (八·3 新增 — CLI 参数)
│   ├── test_conftest.py          # 10 (八·3 新增 — fixtures 自身)
│   └── test_types_typed.py       # 8  (八·3 新增 — TypedDict)
├── data_analysis.py              # 阶段二
├── chunk_processor.py            # 阶段三
├── vector_indexer.py             # 阶段四
├── query_enhancer.py             # 阶段五
├── multi_path_retriever.py       # 阶段六·上(融合层)
├── stage6_retrieval_pipeline.py  # 阶段六·下(端到端,主入口)
├── retrieval_pipeline.py         # DEPRECATED
├── rag_medical.py                # 工具集
├── context_assembler.py          # 八·1 上下文组装
├── prompt_templates.py           # 八·1 prompt 模板(防幻觉强化)
├── llm_generator.py              # 八·2 + 八·3 LLM 生成器
│                                  #     - Semaphore 并发限流
│                                  #     - LRU 缓存(磁盘持久化)
│                                  #     - 流式生成
│                                  #     - allow_no_llm 离线模式
├── medical_generation_pipeline.py # 八·2 + 八·3 医学生成流水线
│                                  #     - 6 阶段:context → eval → gen → review → final → postprocess
│                                  #     - metrics_path jsonl 持久化
├── stage8_e2e_demo.py            # 八·2 + 八·3 端到端 demo(--no-llm 支持)
├── types_typed.py                # 八·3 TypedDict 类型定义
├── eval_50query.py               # 50 query 评估脚本
├── pytest.ini                    # pytest 配置(默认 env vars)
├── requirements.txt
├── README.md                     # 本文件
└── .github/workflows/tests.yml   # CI 矩阵(ubuntu/windows × py3.10/3.12)
```

## 关键设计决策

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

### 4. 端到端 prompt 模板(防幻觉强化)

- 严格基于 context(不编造)
- **严禁编造未在 context 出现的 PMID / 试验名 / 数据**
- **引用必须是 context 里真实出现的 PMID**
- 不知道就说"现有文献未明确支持"
- 医学免责("建议咨询执业医师")

### 5. 八·3 工程优化

- **Semaphore 并发限流**:防止 batch 调用时把 Ollama 打爆
- **LRU 缓存**:相同 query 复用结果,默认 32 条容量,可持久化到磁盘
- **流式生成**:长答案支持 token-by-token 输出,前端可做打字机效果
- **metrics jsonl 持久化**:每次 run 追加一条记录到 jsonl,便于事后分析
- **TypedDict 类型契约**:`CacheStats` / `PipelineMetricsRecord` 等
- **`--no-llm` CLI 模式**:给无 Ollama 环境 / CI 跑流水线结构验证
- **GitHub Actions CI 矩阵**:ubuntu + windows × py3.10 + py3.12

## 已知 P0 问题(已修)

1. **`multi_path_retriever.py` 融合 key 用 `id(doc)`** → 改用 `pmid` 优先,跨路径同源 doc 合并
2. **`weighted_fusion` 单路命中被稀释** → 改为只累加出现过的路径
3. **`MAX_FILES` 默认 None** → 改默认 50(全跑 3028 太慢)
4. **`rerank` 的 `for _, c in candidates` 反向** → `for c, _`
5. **`max_rec` 错调 `_get_authority_score`** → 改回 `_get_recency_score`
6. **BM25 中文 query 永远 0 命中** → 重写 `tokenize()`(中英分离)+ 注入 50+ 医学术语到 jieba 自定义词典
7. **`test_data_analysis` 联网卡死** → 改 lazy tokenizer 加载 + fallback 到字符估算
8. **LLM 幻觉** → prompt 加固:严禁编造 PMID / 试验 / 数据,引用必须真实
9. **`.gitignore` 缺 metrics / cache** → 补全 Stage 8 产物
10. **`_filter_by_evaluation` 把"2016年"当 PMID** → 改用严格正则 `PMID[:\s=]+(\d{4,9})`

## BM25 修复详解(2026-06-11)

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
| BM25 召回率 | 0% | 100% | |
| 平均检索耗时 | 211.7ms | 164.5ms | **-22%** |
| 总耗时 | 10.6s | 8.2s | -23% |
| Top-1 多样性 | 74.5% | 76.6% | +2.1% |

**意义**:修复前 BM25 几乎不起作用,所有效果都靠向量检索;修复后 BM25 真的参与召回,RRF 融合才有意义,多路检索架构**名副其实**。

## LLM 答案示例

### Q: "What is ARNO protein?"

> 根据提供的医学文献,ARNO是一种包含三个结构域的蛋白质:N端的GAP结构域(GTPase-activating domain),中间的Sec7结构域(与酵母蛋白Sec7具有同源性),以及C端的pleckstrin homology结构域(PH结构域)[1]。文献指出,ARNO的催化活性(针对鸟苷酸交换的活性)位于Sec7结构域,并通过PH结构域与磷脂酰肌醇(3,4,5)-三磷酸的相互作用进行调控[1]。

 参考:
- PMID: 12969509 (2003) - https://pubmed.ncbi.nlm.nih.gov/12969509/

### Q: "二甲双胍对心血管疾病的影响"

> 根据提供的医学文献,我无法回答该问题。文献片段中未包含关于二甲双胍对心血管疾病影响的任何信息[1][2][3]。

*(LLM 正确拒答,符合 prompt 规则 — 召回的 5 篇不相关时,直接说"不知道")*

## 测试覆盖明细

| 测试文件 | 数量 | 覆盖 |
|---|---|---|
| test_data_analysis.py | 20 | XML 解析、字段提取、质量检查 |
| test_chunk_processor.py | 18 | 切分、token 估计 |
| test_vector_indexer.py | 16 | 嵌入、归一化、过滤 |
| test_query_enhancer.py | 41 | 清洗、实体、同义词、过滤 |
| test_multi_path_retriever.py | 23 | 融合 key / RRF / weighted / simple / vector_query |
| test_bm25_tokenize.py | 12 | BM25 命中(tokenize / 中英 / 医学词典) |
| test_retrieval_pipeline.py | 34 | deprecated 版的组件 |
| test_rag_medical.py | 16 | BGEReranker / HybridRetriever |
| test_stage6_retrieval_pipeline.py | 38 | 端到端 retrieval 流水线 |
| test_stage8_end_to_end.py | 6 | 八 e2e |
| test_context_assembler.py | 28 | 八·1 上下文组装 |
| test_prompt_templates.py | 20 | 八·1 prompt 模板 |
| test_llm_generator.py | 37 | 八·2 + 八·3(extract_json / 接口 / 并发 / LRU / 流式 / allow_no_llm) |
| test_medical_pipeline.py | 25 | 八·2 + 八·3(辅助方法 / run / metrics 持久化) |
| test_stage8_cli.py | 10 | 八·3 CLI 参数(--no-llm / --query / --metrics-jsonl) |
| test_conftest.py | 10 | 八·3 fixtures 自身测试 |
| test_types_typed.py | 8 | 八·3 TypedDict 定义 + 使用 |
| **合计** | **331** | **~30s 全过** |

## CI 工作流

`.github/workflows/tests.yml` 跑矩阵测试:
- **OS**: ubuntu-latest + windows-latest
- **Python**: 3.10 + 3.12
- **步骤**:
  1. install deps
  2. `pytest tests/ -v --tb=short`
  3. `python stage8_e2e_demo.py --no-llm`(验证 CLI)
  4. (可选)上传 coverage 到 Codecov

跑法:推 main / 开 PR / 手动 dispatch 都会触发。

## 后续可做

- [ ] 多模态(PubMed 论文的图表)
- [ ] 反馈学习(用户标注 → 排序优化)
- [ ] 医学本体(UMLS / MeSH)接入
- [ ] 多语言混合检索
- [x] 流式 LLM 答案(token-by-token)— 八·3 完成
- [ ] Web UI(Gradio / Streamlit)
- [ ] FastAPI 服务化
- [ ] 真实数据集评估(BioASQ / TREC-COVID)
- [ ] Prometheus metrics 导出(从 jsonl → 实时)