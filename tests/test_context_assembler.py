"""
Stage 8 Part 3 — Tests for context_assembler.py
"""

import unittest
from typing import List

from context_assembler import (
    ContextAssembler,
    DocumentChunk,
    assemble_context,
)


def _make_chunks() -> List[DocumentChunk]:
    return [
        DocumentChunk(
            text="二甲双胍是治疗2型糖尿病的一线药物,可以通过抑制肝糖输出降低血糖。",
            metadata={"pmid": "12345", "chunk_index": 0},
            relevance_score=0.92,
            source="12345",
            chunk_id="12345#0",
        ),
        # 与上一条几乎完全重复 — 应当被去重
        DocumentChunk(
            text="二甲双胍是治疗2型糖尿病的一线药物,可以通过抑制肝糖输出降低血糖。",
            metadata={"pmid": "12345", "chunk_index": 1},
            relevance_score=0.85,
        ),
        # 不同来源
        DocumentChunk(
            text="SGLT2抑制剂对心衰患者有显著获益,2023年Meta分析显示全因死亡率下降13%。",
            metadata={"pmid": "67890", "chunk_index": 0},
            relevance_score=0.78,
        ),
        # 又一个不同来源
        DocumentChunk(
            text="GLP-1受体激动剂(如司美格鲁肽)在减重和心血管保护方面显示多效性。",
            metadata={"pmid": "11111", "chunk_index": 0},
            relevance_score=0.65,
        ),
    ]


class TestDocumentChunk(unittest.TestCase):
    def test_post_init_autofills_source(self):
        c = DocumentChunk(text="x", metadata={"pmid": "99999", "chunk_index": 3})
        self.assertEqual(c.source, "99999")
        self.assertEqual(c.chunk_id, "99999#3")

    def test_post_init_keeps_explicit(self):
        c = DocumentChunk(
            text="x", metadata={"pmid": "99999"},
            source="custom", chunk_id="custom_id",
        )
        self.assertEqual(c.source, "custom")
        self.assertEqual(c.chunk_id, "custom_id")

    def test_dict_normalization(self):
        d = {
            "text": "Some text",
            "metadata": {"pmid": "11111", "chunk_index": 0},
            "relevance_score": 0.5,
        }
        c = ContextAssembler._normalize(d)
        self.assertIsInstance(c, DocumentChunk)
        self.assertEqual(c.text, "Some text")
        self.assertEqual(c.source, "11111")
        self.assertEqual(c.chunk_id, "11111#0")


class TestJaccard(unittest.TestCase):
    def test_identical(self):
        a = ContextAssembler._tokenize_for_jaccard("二甲双胍治疗糖尿病")
        b = ContextAssembler._tokenize_for_jaccard("二甲双胍治疗糖尿病")
        self.assertAlmostEqual(ContextAssembler.jaccard(a, b), 1.0)

    def test_disjoint(self):
        a = ContextAssembler._tokenize_for_jaccard("二甲双胍")
        b = ContextAssembler._tokenize_for_jaccard("阿司匹林")
        self.assertEqual(ContextAssembler.jaccard(a, b), 0.0)

    def test_partial_overlap(self):
        a = ContextAssembler._tokenize_for_jaccard("二甲双胍治疗糖尿病心血管")
        b = ContextAssembler._tokenize_for_jaccard("二甲双胍治疗高血压")
        sim = ContextAssembler.jaccard(a, b)
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)


class TestDeduplication(unittest.TestCase):
    def test_removes_near_duplicates(self):
        a = ContextAssembler(max_tokens=5000, offline=True)
        chunks = _make_chunks()
        unique = a._deduplicate(chunks)
        # 1 与 2 几乎相同,应去重一个 — 4 -> 3
        self.assertEqual(len(unique), 3)

    def test_keeps_higher_relevance(self):
        a = ContextAssembler(max_tokens=5000, offline=True)
        chunks = _make_chunks()
        unique = a._deduplicate(chunks)
        # 留下来的应当是 0.92 分的那个(12345#0),不是 0.85 的
        kept_pmid = unique[0].metadata["pmid"]
        self.assertEqual(kept_pmid, "12345")


class TestDiversityRerank(unittest.TestCase):
    def test_penalty_for_same_source(self):
        a = ContextAssembler(max_tokens=5000, diversity_penalty=0.2, max_per_source=2, offline=True)
        chunks = [
            DocumentChunk(text="a", metadata={"pmid": "A"}, relevance_score=0.9),
            DocumentChunk(text="b", metadata={"pmid": "A"}, relevance_score=0.9),
            DocumentChunk(text="c", metadata={"pmid": "A"}, relevance_score=0.9),
            DocumentChunk(text="d", metadata={"pmid": "B"}, relevance_score=0.9),
        ]
        ranked = a._diversity_rerank(chunks)
        # 第 1 个 A 的 adjusted=0.9(全),第 2 个 A=0.72,第 3 个 A=0.576(>max_per_source=2 实际 0)
        # B 的 adjusted=0.9。第二个 A(0.72)和 B(0.9)排序,显然 B 排前 — 但
        # 第 1 个 A 已经在 source_count 累加,B 是 cnt=0, B 排第一 — 测试稳定
        # 但 max_per_source=2 限制下,排序只看分数不看 cnt,故 0.9 的 A 跟 0.9 的 B 并列
        # 期望:头两位一个是 A 一个是 B,具体哪个是 stable 排序决定
        top_two_pmids = {ranked[0].metadata["pmid"], ranked[1].metadata["pmid"]}
        self.assertEqual(top_two_pmids, {"A", "B"})

    def test_penalty_actually_reduces_score(self):
        a = ContextAssembler(max_tokens=5000, diversity_penalty=0.3, max_per_source=10, offline=True)
        # 同源第二个分数应明显低于第一个
        c1 = DocumentChunk(text="x", metadata={"pmid": "A"}, relevance_score=1.0)
        c2 = DocumentChunk(text="y", metadata={"pmid": "A"}, relevance_score=1.0)
        ranked = a._diversity_rerank([c1, c2])
        # c2 的 adjusted = 1.0 * 0.7 = 0.7, 排序应当 c1 在前
        # 但分数 c1 调整后也是 1.0, 同样 c2=0.7, c1 应当在前
        self.assertIs(ranked[0], c1)
        self.assertIs(ranked[1], c2)

    def test_max_per_source_limit(self):
        a = ContextAssembler(max_per_source=1, diversity_penalty=0.0, offline=True)
        chunks = [
            DocumentChunk(text="a", metadata={"pmid": "A"}, relevance_score=0.9),
            DocumentChunk(text="b", metadata={"pmid": "A"}, relevance_score=0.8),
        ]
        ranked = a._diversity_rerank(chunks)
        # max_per_source=1,第二个的 adjusted = 0,排在最后
        self.assertEqual(ranked[0].metadata["pmid"], "A")
        # 但其实两个都还是会被返回,只是 0 分 — 测试 max_per_source 路径
        self.assertEqual(len(ranked), 2)


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        a = ContextAssembler(offline=True)
        self.assertEqual(a.estimate_tokens(""), 0)

    def test_chinese_text(self):
        a = ContextAssembler(offline=True)
        # 100 个中文字符 -> 约 100 token (中文按字数)
        text = "二甲双胍" * 50
        self.assertGreater(a.estimate_tokens(text), 50)

    def test_english_text(self):
        a = ContextAssembler(offline=True)
        # 100 个英文词 -> tokenizer 失败时 1 token ≈ 1.5 字符
        text = "metformin " * 100
        est = a.estimate_tokens(text)
        self.assertGreater(est, 0)


class TestTruncateAtSentence(unittest.TestCase):
    """验证"后 10% 窗口"截断行为 — 老师明确要求按字面实现。"""

    def test_shorter_than_limit(self):
        """text 比 target_chars 短 → 直接返回原文"""
        text = "短文本。短。"
        self.assertEqual(
            ContextAssembler._truncate_at_sentence(text, 100),
            text,
        )

    def test_truncate_at_period_in_last_10pct(self):
        """句末标点在后 10% 窗口内 → 在标点处截断(关键场景)

        target_chars=100, 窗口 = 10 字符(text[90:100])。
        句末 ". " 在 text[95..96] 范围内(窗口内)。
        应该截到 "A"*95 + ". " (97 字符),而不是硬截到 100。
        """
        text = "A" * 95 + ". " + "B" * 100  # 总长 197
        out = ContextAssembler._truncate_at_sentence(text, 100)
        self.assertTrue(out.endswith(". "), f"应截到 '. ' 处,实际末尾: {out[-10:]!r}")
        self.assertEqual(out, "A" * 95 + ". ")

    def test_truncate_hard_when_no_punct_in_window(self):
        """后 10% 窗口内没有标点 → 硬截(不向后搜索)"""
        # target_chars=100, 窗口 = text[90:100]
        # 整个 text 没有标点 → 硬截
        text = "A" * 200
        out = ContextAssembler._truncate_at_sentence(text, 100)
        self.assertEqual(out, "A" * 100)

    def test_truncate_punct_before_window_ignored(self):
        """标点在窗口之前 → 不被找到(不会越过窗口去取)"""
        # target_chars=100, 窗口 = text[90:100]
        # 标点在 text[50](远在窗口之前)
        text = "A" * 50 + "." + "B" * 200
        out = ContextAssembler._truncate_at_sentence(text, 100)
        # 标点位置 50 不会被找到,因为窗口只看 90-100
        self.assertEqual(out, "A" * 50 + "." + "B" * 49)  # 硬截到 100

    def test_truncate_chinese_period_in_10pct(self):
        """中文句号 "。" 在后 10% 窗口内 → 截到句号

        target_chars=30, 窗口 = 3 字符(text[27:30])。
        把 "。" 放在 text[28](窗口内),验证能找到。
        """
        # 27 个 A + "后" + "。" + "更多内容" = 27 + 1 + 1 + 4 = 33
        text = "A" * 27 + "后" + "。" + "更多内容"
        out = ContextAssembler._truncate_at_sentence(text, 30)
        # 窗口=text[27:30]="后。X",找到 "。" at offset 1
        # return text[:30-3+1+1] = text[:29] = "A"*27 + "后" + "。" = 29 字符
        self.assertTrue(out.endswith("。"), f"应截到 '。' 处,实际末尾: {out[-5:]!r}")
        self.assertEqual(len(out), 29)

    def test_truncate_window_floor_to_1(self):
        """target_chars < 10 时,窗口至少 1 字符(防 0 窗口死循环)"""
        # target_chars=5, 窗口 = max(1, 5//10) = 1
        text = "AAAA.BBBB"  # "." 在 text[4]
        out = ContextAssembler._truncate_at_sentence(text, 5)
        # 窗口 = 1,候选区 = text[4:5] = "."
        # 找到 "." → return text[:4 + 0 + 1] = text[:5] = "AAAA."
        self.assertEqual(out, "AAAA.")

    def test_truncate_legacy_compat(self):
        """保留旧测试场景(防止回归)"""
        text = "第一句。第二句比较长,后面还有很多内容。第三句非常长" + "x" * 200
        out = ContextAssembler._truncate_at_sentence(text, 20)
        # 截断后不应超过 20 字符太多
        self.assertLessEqual(len(out), 30)
        # 应当在某个句末标点结束("第一句。第二句比较长,后面还有很多内容。" 中
        # 的 "。" 在位置 19,正好在 target_chars=20 的 2 字符窗口内)
        self.assertIn(out[-1], "。.!?？!\n;")


class TestAssemble(unittest.TestCase):
    def test_basic_assemble(self):
        out = assemble_context(_make_chunks(), max_tokens=500)
        self.assertIn("context_text", out)
        self.assertIn("metadata", out)
        self.assertIn("selected_chunks", out)

    def test_metadata_shape(self):
        out = assemble_context(_make_chunks(), max_tokens=500)
        meta = out["metadata"]
        for key in (
            "total_chunks_retrieved",
            "unique_chunks_after_dedup",
            "chunks_selected",
            "estimated_tokens",
            "chunk_sources",
        ):
            self.assertIn(key, meta)

    def test_token_budget_respected(self):
        # 制造大量 chunk,看 selected 是否受 token 限制
        chunks = [
            DocumentChunk(
                text="x" * 200,  # 200 字符
                metadata={"pmid": f"P{i:05d}", "chunk_index": 0},
                relevance_score=1.0 - i * 0.001,
            )
            for i in range(50)
        ]
        out = assemble_context(chunks, max_tokens=100)
        # 100 token 预算,每个 chunk 估 ~50 token(中英混合),所以至多选 ~2-3 个
        self.assertLessEqual(len(out["selected_chunks"]), 5)

    def test_dict_input_works(self):
        d = {"text": "test", "metadata": {"pmid": "T1", "chunk_index": 0},
             "relevance_score": 0.5}
        out = assemble_context([d], max_tokens=100)
        self.assertEqual(len(out["selected_chunks"]), 1)
        self.assertIn("T1", out["context_text"])

    def test_relevance_ordering(self):
        out = assemble_context(_make_chunks(), max_tokens=1000)
        scores = [c.relevance_score for c in out["selected_chunks"]]
        # 选中的 chunk 应当按分数降序
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_handles_empty_input(self):
        out = assemble_context([], max_tokens=100)
        self.assertEqual(out["context_text"], "")
        self.assertEqual(out["metadata"]["total_chunks_retrieved"], 0)
        self.assertEqual(out["metadata"]["chunks_selected"], 0)


class TestAnalyzeSources(unittest.TestCase):
    def test_counts_correctly(self):
        chunks = [
            DocumentChunk(text="a", metadata={"pmid": "A"}, relevance_score=0.9),
            DocumentChunk(text="b", metadata={"pmid": "A"}, relevance_score=0.8),
            DocumentChunk(text="c", metadata={"pmid": "B"}, relevance_score=0.7),
        ]
        counts = ContextAssembler._analyze_sources(chunks)
        self.assertEqual(counts, {"A": 2, "B": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
