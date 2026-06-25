"""
Stage 8 Part 2.4: 端到端 e2e demo — 用真实 Ollama qwen3:8b 跑通整条流水线

用法:
    python stage8_e2e_demo.py                  # 跑全部 query
    python stage8_e2e_demo.py --no-llm         # 跳过 LLM,只验证流水线结构
    python stage8_e2e_demo.py --query "xxx"    # 只跑自定义 query(默认 3 个)
    python stage8_e2e_demo.py --metrics-jsonl  # 持久化 metrics 到 jsonl

前提(默认模式):
    - Ollama 服务在 http://localhost:11434 运行
    - qwen3:8b 模型已下载

行为:
    - 跑 3 个测试 query
    - 每个 query 输出最终答案 + 中间阶段产物 + 关键指标
    - 写日志到 stage8_e2e_log.txt
    - 打印汇总报告
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from context_assembler import DocumentChunk
from medical_generation_pipeline import MedicalGenerationPipeline


# ==================== 测试数据 ====================

# 模拟 3 篇检索结果(用真实风格的医学内容)
SAMPLE_CHUNKS = [
    DocumentChunk(
        text=(
            "二甲双胍是治疗2型糖尿病的一线药物。UKPDS 34 研究显示,"
            "在超重 2 型糖尿病患者中,二甲双胍可降低心肌梗死风险达 39% (p=0.01),"
            "全因死亡率降低 36%。"
        ),
        metadata={"pmid": "12345", "year": "1998", "journal": "Lancet",
                  "title": "UKPDS 34: Effect of intensive blood-glucose control"},
        relevance_score=0.92,
    ),
    DocumentChunk(
        text=(
            "二甲双胍心血管获益的机制可能涉及改善胰岛素抵抗、降低体重、改善血脂谱。"
            "但 CAMERA 研究在非糖尿病心血管患者中未观察到显著获益,"
            "提示获益可能仅限于糖尿病合并心血管风险人群。"
        ),
        metadata={"pmid": "23456", "year": "2016", "journal": "Diabetes Care",
                  "title": "CAMERA: Metformin in non-diabetic cardiovascular patients"},
        relevance_score=0.85,
    ),
    DocumentChunk(
        text=(
            "SGLT2 抑制剂(恩格列净、达格列净、卡格列净)在多项大型 RCT 中显示"
            "对心衰患者有显著心血管获益。EMPA-REG OUTCOME 试验中,"
            "恩格列净使心血管死亡风险降低 38%,全因死亡率降低 32%。"
        ),
        metadata={"pmid": "67890", "year": "2015", "journal": "NEJM",
                  "title": "EMPA-REG OUTCOME"},
        relevance_score=0.78,
    ),
    DocumentChunk(
        text=(
            "GLP-1 受体激动剂(司美格鲁肽、利拉鲁肽)在 LEADER、SUSTAIN-6 等试验中"
            "显示心血管获益,主要机制为减缓动脉粥样硬化进程。"
            "司美格鲁肽可使主要不良心血管事件风险降低 21%。"
        ),
        metadata={"pmid": "11111", "year": "2016", "journal": "NEJM",
                  "title": "LEADER trial"},
        relevance_score=0.65,
    ),
]


# 3 个测试 query
TEST_QUERIES = [
    "二甲双胍对心血管疾病有什么影响?",
    "SGLT2 抑制剂适合什么样的患者?",
    "GLP-1 受体激动剂和 SGLT2 抑制剂哪个更适合心血管保护?",
]


# ==================== 主函数 ====================

def parse_args(argv: list = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="医疗 RAG 流水线端到端 demo",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="跳过 LLM 调用(给无 Ollama 环境或 CI 用)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="只跑这个 query(默认跑全部 3 个)",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=str,
        default=None,
        help="持久化 metrics 到这个 jsonl 文件",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:8b",
        help="LLM 模型名(默认 qwen3:8b)",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="跳过批判性审查阶段(加速)",
    )
    return parser.parse_args(argv)


def main(argv: list = None) -> int:
    args = parse_args(argv)

    # 日志配置:同时输出到文件和控制台
    log_path = Path(__file__).parent / "stage8_e2e_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("e2e_demo")

    logger.info("=" * 70)
    logger.info("Stage 8 Part 2.4 端到端 e2e demo")
    logger.info(f"日志写入: {log_path}")
    logger.info(f"模式: {'--no-llm(无 LLM)' if args.no_llm else '真实 LLM'}")
    logger.info("=" * 70)

    # 决定 query 列表
    queries = [args.query] if args.query else TEST_QUERIES

    # --no-llm 模式:用 mock 替代 LLM,只验证流水线结构
    if args.no_llm:
        return _run_no_llm_mode(queries, logger, args)

    # 正常模式:初始化 pipeline
    try:
        pipeline = MedicalGenerationPipeline(
            llm_model=args.model,
            enable_review=not args.no_review,
            metrics_path=args.metrics_jsonl,
        )
    except ConnectionError as e:
        logger.error(f"无法初始化 pipeline: {e}")
        return 1

    # 跑 3 个 query
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"[Query {i}/{len(TEST_QUERIES)}] {query}")
        logger.info("=" * 70)

        t0 = time.time()
        try:
            result = pipeline.run(query=query, retrieved_docs=SAMPLE_CHUNKS)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Pipeline 异常: {e}", exc_info=True)
            continue
        elapsed = time.time() - t0

        # 输出关键指标
        m = result.generation_metrics
        logger.info(f"  总耗时: {m.total_time_seconds:.2f}s")
        logger.info(f"  答案长度: {len(result.answer)} 字符")
        logger.info(f"  Token 总数: {sum(m.token_counts.values())}")
        logger.info(
            f"  各阶段耗时: "
            + ", ".join(f"{k}={v:.2f}s" for k, v in m.stage_times.items())
        )
        logger.info(
            f"  各阶段成功: "
            + ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in m.stage_success.items())
        )
        logger.info(f"  引用来源 ({len(result.sources)} 条):")
        for s in result.sources:
            logger.info(
                f"    - PMID:{s['pmid']} "
                f"({s.get('year', '?')}) "
                f"relevance={s['relevance_score']}"
            )

        logger.info("")
        logger.info("  【最终答案】")
        for line in result.answer.split("\n"):
            logger.info(f"    {line}")

        results.append({
            "query": query,
            "elapsed": elapsed,
            "metrics": m,
            "answer": result.answer,
            "sources": result.sources,
        })

    # 汇总报告
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 汇总报告")
    logger.info("=" * 70)
    logger.info(f"  总 query 数: {len(results)}")
    if results:
        avg_time = sum(r["elapsed"] for r in results) / len(results)
        total_tokens = sum(
            sum(r["metrics"].token_counts.values()) for r in results
        )
        avg_answer_len = sum(len(r["answer"]) for r in results) / len(results)
        logger.info(f"  平均耗时: {avg_time:.2f}s")
        logger.info(f"  平均答案长度: {avg_answer_len:.0f} 字符")
        logger.info(f"  总 token 数: {total_tokens}")
        # 各 query 单独统计
        logger.info("")
        logger.info("  各 query 详细:")
        for i, r in enumerate(results, 1):
            q = r["query"][:30] + ("..." if len(r["query"]) > 30 else "")
            logger.info(
                f"    [{i}] {q}  | 耗时 {r['elapsed']:.1f}s  | "
                f"答案 {len(r['answer'])} 字  | "
                f"引用 {len(r['sources'])} 条"
            )
        # 阶段成功率
        all_stages = set()
        for r in results:
            all_stages.update(r["metrics"].stage_success.keys())
        logger.info("")
        for stage in sorted(all_stages):
            successes = sum(
                1 for r in results
                if r["metrics"].stage_success.get(stage, False)
            )
            rate = successes / len(results) * 100
            logger.info(f"  阶段 {stage}: 成功率 {rate:.0f}% ({successes}/{len(results)})")

    logger.info("")
    logger.info(f"✅ Demo 完成,详细日志: {log_path}")
    return 0


def _run_no_llm_mode(
    queries: list, logger: logging.Logger, args: argparse.Namespace,
) -> int:
    """
    --no-llm 模式 — 不连真实 Ollama,用 mock LLM 验证流水线结构。
    适合:
    - CI 环境跑测试
    - 没有 GPU/Ollama 的开发机
    - 想快速验证 context assembler / 后处理 / 引用格式化是否正常
    """
    from unittest.mock import MagicMock
    from llm_generator import GenerationResult

    logger.info(">>> 启用 --no-llm 模式 — 用 mock LLM 验证流水线 <<<")

    # mock LLMGenerator
    mock_llm = MagicMock()
    counter = {"n": 0}

    def fake_generate(user_prompt: str = "", system_prompt: str = "", config=None):
        n = counter["n"]
        counter["n"] += 1
        order = ["eval", "gen", "review", "final"]
        stage = order[n] if n < len(order) else "extra"
        texts = {
            "eval": (
                "【文档1】\n- 相关性:高\n- 证据等级:1b\n"
                "- 可用性:可作为直接证据\nPMID:12345\n"
            ),
            "gen": f"[MOCK] 草稿答案 #{n}。",
            "review": "整体评级:A。\n### 修订建议:\n- 保持现有结构",
            "final": f"[MOCK] 最终答案 #{n} [PMID:12345]。",
        }
        return GenerationResult(text=texts.get(stage, "[MOCK]"), success=True)

    mock_llm.generate.side_effect = fake_generate
    # 让 cache_stats 返回合理结构
    mock_llm.cache_stats = MagicMock(return_value={"size": 0, "max_size": 32, "utilization": 0.0})

    # 构造 pipeline,但替换 llm
    try:
        # 用 MagicMock 类代替 LLMGenerator,跳过真实连接
        with __import__("unittest.mock").mock.patch(
            "medical_generation_pipeline.LLMGenerator", return_value=mock_llm,
        ):
            pipeline = MedicalGenerationPipeline(
                llm_model=args.model,
                enable_review=not args.no_review,
                metrics_path=args.metrics_jsonl,
            )
    except Exception as e:  # noqa: BLE001
        logger.error(f"无法初始化 mock pipeline: {e}", exc_info=True)
        return 1

    results = []
    for i, query in enumerate(queries, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"[Query {i}/{len(queries)}] {query}")
        logger.info("=" * 70)

        t0 = time.time()
        try:
            result = pipeline.run(query=query, retrieved_docs=SAMPLE_CHUNKS)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Pipeline 异常: {e}", exc_info=True)
            continue
        elapsed = time.time() - t0

        logger.info(f"  耗时: {elapsed:.2f}s")
        logger.info(f"  引用数: {len(result.sources)}")
        logger.info(f"  答案长度: {len(result.answer)} 字符")
        logger.info(f"  含'重要提示': {'是' if '重要提示' in result.answer else '否'}")
        logger.info(f"  含 PMID 引用: {'是' if 'PMID:12345' in result.answer else '否'}")

        results.append({
            "query": query,
            "elapsed": elapsed,
            "answer_length": len(result.answer),
        })

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ --no-llm 模式完成")
    logger.info(f"  总 query 数: {len(results)}")
    if results:
        logger.info(f"  平均耗时: {sum(r['elapsed'] for r in results)/len(results):.3f}s")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())