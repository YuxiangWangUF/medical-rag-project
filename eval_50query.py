"""
跑 50 query 评估,统计多路检索效果

- 不用 LLM 生成(快)
- 输出 recall / latency / top-1 命中率
- 保存 JSON 报告到 ./output/stage6_eval.json
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, '.')

from stage6_retrieval_pipeline import RetrievalPipeline

# 50 个测试 query,覆盖中英双语 + 多个医学主题
EVAL_QUERIES = [
    # 英文 (20)
    "What is ARNO protein?",
    "ARF protein activation of phospholipase D",
    "PD-1 inhibitor immunotherapy cancer",
    "EGFR mutation lung cancer treatment",
    "metformin cardiovascular disease",
    "SGLT2 inhibitor heart failure diabetes",
    "aspirin clopidogrel combination therapy",
    "hypertension ARB treatment efficacy",
    "CAR-T cell therapy recent advances",
    "ACE inhibitor diabetic kidney disease",
    "mRNA vaccine cancer therapy",
    "HER2-positive breast cancer trastuzumab",
    "JAK inhibitor rheumatoid arthritis",
    "statin therapy primary prevention",
    "checkpoint inhibitor melanoma",
    "anticoagulation atrial fibrillation",
    "GLP-1 agonist weight loss mechanism",
    "leukemia targeted therapy BCR-ABL",
    "antibody-drug conjugate solid tumor",
    "tumor microenvironment immunotherapy",

    # 中文 (20)
    "二甲双胍对心血管疾病的影响",
    "PD-1 免疫疗法的最新进展",
    "阿司匹林在心血管疾病一级预防中的作用",
    "EGFR 突变肺癌靶向治疗",
    "高血压患者的 ARB 类药物治疗",
    "SGLT2 抑制剂在心力衰竭中的应用",
    "近五年 CAR-T 治疗血液肿瘤研究",
    "GLP-1 受体激动剂减肥机制",
    "二型糖尿病合并心血管疾病用药",
    "高血压合并糖尿病的联合用药",
    "近三年免疫检查点抑制剂研究",
    "急性心肌梗死的抗凝治疗",
    "新冠后遗症的长期影响",
    "EGFR-TKI 耐药机制",
    "ACEI 与 ARB 的区别",
    "糖尿病肾病的治疗策略",
    "急性白血病靶向治疗",
    "冠心病的二级预防",
    "肿瘤免疫治疗耐药机制",
    "新型口服抗凝药比较",

    # 长尾 / 难 query (10)
    "ARNO Sec7 domain structure",
    "mitochondrial dysfunction aging",
    "neuroinflammation Alzheimer disease",
    "gut microbiome obesity",
    "CRISPR gene therapy sickle cell",
    "circadian rhythm sleep disorder",
    "antibiotic resistance MRSA",
    "long COVID cognitive symptoms",
    "人工智能辅助医学诊断",
    "基因编辑伦理问题",
]


def main():
    print("=" * 60)
    print("阶段六:50 query 评估")
    print("=" * 60)

    pipeline = RetrievalPipeline(max_files=50)

    t0 = time.time()
    results = pipeline.batch_query(EVAL_QUERIES, generate_answer=False, verbose=True)
    total_time = time.time() - t0

    # 统计
    total = len(results)
    success = sum(1 for r in results if r.error is None)
    has_docs = sum(1 for r in results if len(r.retrieved_docs) > 0)
    recall_counts = [len(r.retrieved_docs) for r in results]
    ret_times = [r.retrieval_time_ms for r in results]

    report = {
        "total_queries": total,
        "success": success,
        "failure": total - success,
        "has_documents": has_docs,
        "no_documents": total - has_docs,
        "avg_recall": round(sum(recall_counts) / total, 2),
        "min_recall": min(recall_counts),
        "max_recall": max(recall_counts),
        "avg_retrieval_time_ms": round(sum(ret_times) / total, 1),
        "total_eval_time_s": round(total_time, 1),
        "fusion_strategy": "rrf",
        "per_query": [
            {
                "query": r.query,
                "retrieved_count": len(r.retrieved_docs),
                "retrieval_time_ms": round(r.retrieval_time_ms, 1),
                "top1_pmid": r.retrieved_docs[0].metadata.get("pmid") if r.retrieved_docs else None,
                "top1_year": r.retrieved_docs[0].metadata.get("year") if r.retrieved_docs else None,
                "error": r.error,
            }
            for r in results
        ],
    }

    # 保存
    output_path = Path("./output/stage6_eval_full_3028.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("📊 评估报告")
    print("=" * 60)
    print(f"总查询数: {total}")
    print(f"成功: {success} | 失败: {total - success}")
    print(f"有召回: {has_docs} | 无召回: {total - has_docs}")
    print(f"平均召回数/查询: {report['avg_recall']} 篇")
    print(f"召回范围: [{report['min_recall']}, {report['max_recall']}]")
    print(f"平均检索耗时: {report['avg_retrieval_time_ms']}ms")
    print(f"总评估耗时: {report['total_eval_time_s']}s")
    print(f"\n报告保存到: {output_path}")


if __name__ == "__main__":
    main()
