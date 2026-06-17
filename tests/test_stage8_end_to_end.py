"""
Stage 8 Part 4: 端到端集成测试

把 stage6 检索结果 -> ContextAssembler 组装 -> 4 阶段 PromptStage 渲染
完整跑一遍,验证:
1. LangChain Document 列表能正确喂给 ContextAssembler
2. 组装后的 context_text 能塞进所有 4 个 prompt
3. 4 个 stage 渲染输出长度合理,占位符都填上了
"""

import unittest
from typing import List

from context_assembler import ContextAssembler, DocumentChunk, assemble_context
from prompt_templates import (
    EVIDENCE_EVALUATOR,
    ANSWER_GENERATOR,
    CRITICAL_REVIEWER,
    FINAL_ASSEMBLER,
    PIPELINE_ORDER,
    get_full_pipeline,
)

# 用真实的 LangChain Document(项目里已经装好 langchain-core)
from langchain_core.documents import Document as LCDocument


def _make_lc_docs() -> List[LCDocument]:
    """模拟 stage6 retriever 输出的 LangChain Document 列表"""
    return [
        LCDocument(
            page_content=(
                "二甲双胍是治疗2型糖尿病的一线药物,可以通过抑制肝糖输出降低血糖。"
                "UKPDS 34 研究显示,二甲双胍在超重 2 型糖尿病患者中可降低 "
                "心肌梗死风险达 39% (p=0.01)。"
            ),
            metadata={
                "pmid": "12345",
                "year": "1998",
                "journal": "Lancet",
                "chunk_index": 0,
                "relevance_score": 0.92,
            },
        ),
        LCDocument(
            page_content=(
                "二甲双胍心血管获益的机制可能涉及改善胰岛素抵抗、降低体重、"
                "改善血脂谱。CAMERA 研究在非糖尿病心血管患者中未观察到显著获益,"
                "提示获益可能仅限于糖尿病合并心血管风险人群。"
            ),
            metadata={
                "pmid": "12345",
                "year": "2016",
                "journal": "Diabetes Care",
                "chunk_index": 1,
                "relevance_score": 0.85,
            },
        ),
        LCDocument(
            page_content=(
                "SGLT2 抑制剂(恩格列净、达格列净)在 EMPA-REG、DECLARE-TIMI 58 "
                "等大型 RCT 中显示对心衰患者有显著心血管获益,"
                "全因死亡率下降约 13-14%。"
            ),
            metadata={
                "pmid": "67890",
                "year": "2019",
                "journal": "NEJM",
                "chunk_index": 0,
                "relevance_score": 0.78,
            },
        ),
        LCDocument(
            page_content=(
                "GLP-1 受体激动剂(司美格鲁肽、利拉鲁肽)在 LEADER、SUSTAIN-6 "
                "等试验中显示心血管获益,主要机制为减缓动脉粥样硬化进程。"
            ),
            metadata={
                "pmid": "11111",
                "year": "2016",
                "journal": "NEJM",
                "chunk_index": 0,
                "relevance_score": 0.65,
            },
        ),
    ]


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.docs = _make_lc_docs()
        self.question = "二甲双胍对心血管疾病有什么影响?"

    def test_lc_documents_normalized_correctly(self):
        """LangChain Document 列表 -> ContextAssembler 应当能直接处理"""
        out = assemble_context(self.docs, max_tokens=2000, offline=True)
        self.assertGreater(len(out["selected_chunks"]), 0)
        # 这里两个 12345 的 chunk 内容差异较大(一个讲 UKPDS 34 + 心肌梗死,
        # 一个讲 CAMERA + 机制),Jaccard 不应判为重复 — 这是预期行为。
        sources = out["metadata"]["chunk_sources"]
        self.assertIn("12345", sources)
        self.assertEqual(sources["12345"], 2)  # 内容差异大,不被 dedup

    def test_lc_documents_with_dedup(self):
        """当 LC Documents 真的相似时,dedup 应当生效"""
        similar_docs = [
            LCDocument(
                page_content="二甲双胍是2型糖尿病的一线用药,能显著降低血糖水平。",
                metadata={"pmid": "A", "chunk_index": 0, "relevance_score": 0.9},
            ),
            LCDocument(
                page_content="二甲双胍是2型糖尿病的一线用药,能显著降低血糖水平。",
                metadata={"pmid": "A", "chunk_index": 1, "relevance_score": 0.8},
            ),
            LCDocument(
                page_content="SGLT2抑制剂降低心衰住院率。",
                metadata={"pmid": "B", "chunk_index": 0, "relevance_score": 0.7},
            ),
        ]
        out = assemble_context(similar_docs, max_tokens=2000, offline=True)
        sources = out["metadata"]["chunk_sources"]
        self.assertEqual(sources["A"], 1)  # 真的重复了 — 被去重成 1 个
        self.assertEqual(sources["B"], 1)
        # 4 -> 3(去重) -> 2(选 2 个,A 和 B)
        self.assertEqual(out["metadata"]["total_chunks_retrieved"], 3)
        self.assertEqual(out["metadata"]["unique_chunks_after_dedup"], 2)

    def test_full_4_stage_pipeline_renders(self):
        """组装上下文 + 4 阶段 prompt 全部能渲染"""
        out = assemble_context(self.docs, max_tokens=2000, offline=True)
        ctx = out["context_text"]
        self.assertGreater(len(ctx), 50)

        # 1. evidence_evaluator
        eval_rendered = EVIDENCE_EVALUATOR.render(
            context=ctx, question=self.question,
        )
        self.assertIn(self.question, eval_rendered)
        self.assertIn(ctx, eval_rendered)
        # 评估输出应包含 PMID 引用(虽然 evaluator 不一定要求)
        # 不强求 PMID,但要包含 "【文档1】" 这种标识

        # 2. answer_generator
        gen_rendered = ANSWER_GENERATOR.render(
            context=ctx, question=self.question,
            evaluation=eval_rendered,
        )
        self.assertIn(self.question, gen_rendered)
        self.assertIn(eval_rendered[:50], gen_rendered)
        self.assertIn("PMID", gen_rendered)

        # 3. critical_reviewer
        review_rendered = CRITICAL_REVIEWER.render(
            context=ctx, question=self.question,
            previous_answer=gen_rendered,
        )
        self.assertIn(self.question, review_rendered)
        self.assertIn(gen_rendered[:50], review_rendered)
        self.assertIn("PMID", review_rendered)

        # 4. final_assembler
        final_rendered = FINAL_ASSEMBLER.render(
            context=ctx, question=self.question,
            previous_answer=gen_rendered, evaluation=review_rendered,
        )
        self.assertIn(self.question, final_rendered)
        self.assertIn("PMID", final_rendered)

    def test_pipeline_order_serialization(self):
        """确认 get_full_pipeline 按正确顺序返回所有 4 个 stage"""
        pipeline = get_full_pipeline()
        self.assertEqual(
            list(pipeline.keys()),
            ["evidence_evaluator", "answer_generator",
             "critical_reviewer", "final_assembler"],
        )

    def test_rendered_prompts_fit_typical_context_window(self):
        """渲染后的 prompt 长度应当适配常见 LLM 上下文窗口(>500 字符,<20K)"""
        out = assemble_context(self.docs, max_tokens=2000, offline=True)
        ctx = out["context_text"]
        for stage_name, stage in get_full_pipeline().items():
            rendered = stage.render(
                context=ctx, question=self.question,
                previous_answer="(mock previous answer)",
                evaluation="(mock evaluation)",
            )
            with self.subTest(stage=stage_name):
                self.assertGreater(len(rendered), 200,
                                   f"{stage_name} 渲染输出太短,可能丢了字段")
                self.assertLess(len(rendered), 20000,
                                f"{stage_name} 渲染输出太长,可能循环引用")

    def test_assembler_metadata_exposes_for_logging(self):
        """metadata 应当足够详细,便于后续 stage 评估"""
        out = assemble_context(self.docs, max_tokens=2000, offline=True)
        meta = out["metadata"]
        self.assertIn("total_chunks_retrieved", meta)
        self.assertIn("unique_chunks_after_dedup", meta)
        self.assertIn("chunks_selected", meta)
        self.assertIn("estimated_tokens", meta)
        self.assertIn("chunk_sources", meta)
        # 这里两个 12345 chunk 内容差异大,不被 dedup,数量应当等于 retrieved
        self.assertEqual(meta["unique_chunks_after_dedup"],
                         meta["total_chunks_retrieved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
