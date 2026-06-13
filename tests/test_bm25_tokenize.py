"""
BM25 命中测试:验证 tokenize 修复后,BM25 能真的命中

跑法:
    D:\Anaconda\envs\medical_rag\python.exe -m pytest tests/test_bm25_tokenize.py -v
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multi_path_retriever import tokenize, BM25Index


# ==================== tokenize 单元测试 ====================
class TestTokenize:
    """新 tokenize:中英分离,各走各的 tokenizer"""

    def test_01_pure_english(self):
        """纯英文按空格切,转小写"""
        toks = tokenize("This is a paper about ARF protein")
        assert "this" in toks
        assert "is" in toks
        assert "paper" in toks
        assert "arf" in toks  # 转小写
        assert "protein" in toks

    def test_02_pure_chinese(self):
        """纯中文 jieba 切,保留整词(>=2 字)"""
        toks = tokenize("二甲双胍对心血管疾病有影响")
        # jieba + 自定义医学词典应切出:'二甲双胍' '对' '心血管疾病' '有' '影响'
        # 我们过滤单字,保留 2+ 字词
        assert "二甲双胍" in toks  # 整词(自定义词典生效)
        assert "心血管疾病" in toks
        assert "影响" in toks
        # 单字被过滤
        assert "对" not in toks
        assert "有" not in toks

    def test_03_mixed_consistency(self):
        """**关键**:query 和 doc 切出来一致"""
        # 用自定义词典后,二甲双胍应作为整体
        doc_tokens = tokenize("Metformin (二甲双胍) reduces cardiovascular risk.")
        q_tokens = tokenize("二甲双胍 心血管疾病")
        # 共有 token
        assert "二甲双胍" in doc_tokens
        assert "二甲双胍" in q_tokens
        assert "心血管疾病" in q_tokens

    def test_04_chinese_single_char_filtered(self):
        """中文单字过滤"""
        toks = tokenize("心 肝 脾 肺 肾")
        # 单字都过滤
        assert toks == []

    def test_05_english_short_tokens_filtered(self):
        """1 字英文过滤(2 字起)"""
        toks = tokenize("I am a big fan of X")
        # 'I', 'a', 'X' 都是 1 字 → 过滤
        # 'am', 'big', 'fan', 'of' 2+ → 保留
        assert "am" in toks
        assert "big" in toks
        assert "fan" in toks
        assert "of" in toks
        assert "i" not in toks
        assert "a" not in toks

    def test_06_punctuation_filtered(self):
        """标点 / 空白 / 噪声过滤"""
        toks = tokenize("hello, world! how are you?")
        # ',' '!' '?' 都是噪声
        assert "," not in toks
        assert "!" not in toks
        assert "?" not in toks
        # 但单词保留
        assert "hello" in toks
        assert "world" in toks
        assert "how" in toks
        assert "are" in toks
        assert "you" in toks

    def test_07_hyphenated_terms(self):
        """PD-1 / CAR-T 等带连字符的术语保留"""
        toks = tokenize("PD-1 immunotherapy CAR-T therapy")
        assert "pd-1" in toks
        assert "immunotherapy" in toks
        assert "car-t" in toks
        assert "therapy" in toks

    def test_08_lowercase_consistency(self):
        """全部转小写"""
        toks = tokenize("EGFR Mutation Lung Cancer")
        assert "egfr" in toks
        assert "mutation" in toks
        assert "lung" in toks
        assert "cancer" in toks


# ==================== BM25 命中测试 ====================
class TestBM25Hits:
    """**关键测试**:tokenize 修复后 BM25 能命中"""

    def _build_index(self) -> BM25Index:
        """构造 5 篇 doc 的 mini BM25"""
        docs = [
            Document(
                page_content="ARNO contains three domains: Sec7, PH, and coiled-coil. ARNO is a GEF.",
                metadata={"pmid": "1", "year": "2003"},
            ),
            Document(
                page_content="ARF proteins activate Phospholipase D (PLD) at the plasma membrane.",
                metadata={"pmid": "2", "year": "2003"},
            ),
            Document(
                page_content="Metformin reduces cardiovascular risk in diabetic patients.",
                metadata={"pmid": "3", "year": "2005"},
            ),
            Document(
                page_content="二甲双胍降低2型糖尿病患者的心血管疾病风险",
                metadata={"pmid": "4", "year": "2010"},
            ),
            Document(
                page_content="PD-1 inhibitors enhance T cell mediated tumor rejection in cancer immunotherapy.",
                metadata={"pmid": "5", "year": "2024"},
            ),
        ]
        idx = BM25Index()
        idx.fit(docs)
        return idx

    def test_09_english_query_hits_relevant_doc(self):
        """英文 query 命中相关 doc"""
        idx = self._build_index()
        results = idx.search("ARNO protein", top_k=2)
        assert len(results) > 0
        # 排名第 1 应该是 ARNO 那篇
        assert "ARNO" in results[0][0].page_content

    def test_10_chinese_query_hits_chinese_doc(self):
        """**关键修复** — 中文 query 命中中文 doc"""
        idx = self._build_index()
        # 这个 query 之前永远 0 命中
        results = idx.search("二甲双胍 心血管", top_k=2)
        assert len(results) > 0
        # top-1 应该是"二甲双胍降低..."那篇
        assert "二甲双胍" in results[0][0].page_content or "心血管" in results[0][0].page_content

    def test_11_mixed_query_does_not_explode(self):
        """中英混合 query 不会让所有 doc 分数爆掉"""
        idx = self._build_index()
        scores = idx.get_scores("PD-1 immunotherapy cancer")
        # 旧 bug 修复后,scores 不应该全部接近 0(说明有命中)
        assert scores.max() > 0.5

    def test_12_query_doc_consistency_for_chinese(self):
        """**关键**:query 和 doc 的中英 token 一致"""
        idx = self._build_index()
        # doc 4 含 "二甲双胍",query 也是 "二甲双胍" → 应该共享 token
        results = idx.search("二甲双胍", top_k=3)
        if results:
            # top-1 应该是 doc 4
            assert "4" in results[0][0].metadata["pmid"]
