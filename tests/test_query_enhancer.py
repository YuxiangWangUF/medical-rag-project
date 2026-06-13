# tests/test_query_enhancer.py
#
# QueryEnhancer 单元测试
# 跑法:
#   pip install pytest
#   pytest tests/test_query_enhancer.py -v
#
# 覆盖:
#   - 基础清洗
#   - 实体识别(中英文)
#   - 同义词扩展(中英文)
#   - PD-1 不误触 PD
#   - 中文子串包含
#   - 时间过滤(中英文)
#   - 期刊过滤(中英文)
#   - 综合 enhance 流程

import os
import sys
import pytest

# 让测试能找到 query_enhancer 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_enhancer import (
    QueryEnhancer, EnhancedQuery,
    MEDICAL_SYNONYMS, MEDICAL_PATTERNS,
)


@pytest.fixture
def enhancer():
    """每个测试一个干净实例"""
    return QueryEnhancer()


# ==================== 基础清洗 ====================
class TestClean:
    def test_strip_whitespace(self, enhancer):
        assert enhancer.clean("  hello  ").strip() == "hello"

    def test_normalize_quotes(self, enhancer):
        # 中文/全角引号 → ASCII 引号
        assert enhancer.clean('"hello"') == '"hello"'
        assert enhancer.clean("'hello'") == "'hello'"

    def test_collapse_multiple_spaces(self, enhancer):
        assert enhancer.clean("hello    world") == "hello world"


# ==================== 同义词扩展(英文) ====================
class TestExpandSynonymsEnglish:
    def test_mi_expands(self, enhancer):
        result = enhancer.expand_synonyms("What is MI?")
        assert "myocardial infarction" in result
        assert "heart attack" in result

    def test_cvd_expands(self, enhancer):
        result = enhancer.expand_synonyms("risk of CVD")
        assert "cardiovascular disease" in result

    def test_t2dm_expands(self, enhancer):
        result = enhancer.expand_synonyms("T2DM patients")
        assert "type 2 diabetes mellitus" in result
        assert "type 2 diabetes" in result

    def test_pd1_does_not_trigger_pd(self, enhancer):
        """PD-1 不应该误触发 PD(帕金森)"""
        result = enhancer.expand_synonyms("PD-1 immunotherapy")
        # 不应包含 parkinson 相关
        assert not any("parkinson" in s.lower() for s in result)
        # 应包含 pdcd1 (PD-1 的标准名)
        assert "pdcd1" in result
        assert "programmed cell death protein 1" in result

    def test_pd_alone_triggers_parkinson(self, enhancer):
        """单独的 'PD' 应该触发 parkinson"""
        result = enhancer.expand_synonyms("PD treatment")
        assert any("parkinson" in s.lower() for s in result)

    def test_no_match_returns_empty(self, enhancer):
        result = enhancer.expand_synonyms("hello world")
        assert result == []


# ==================== 同义词扩展(中文) ====================
class TestExpandSynonymsChinese:
    def test_chinese_substring(self, enhancer):
        result = enhancer.expand_synonyms("心梗病人")
        assert "心肌梗死" in result
        assert "myocardial infarction" in result

    def test_chinese_drug(self, enhancer):
        result = enhancer.expand_synonyms("二甲双胍")
        assert "metformin" in result
        assert "格华止" in result

    def test_chinese_no_match(self, enhancer):
        # 没有医学实体的中文
        result = enhancer.expand_synonyms("今天天气怎么样")
        assert result == []


# ==================== 实体识别(英文) ====================
class TestExtractEntitiesEnglish:
    def test_drug_recognized(self, enhancer):
        entities = enhancer.extract_entities("Does metformin work?")
        assert "drug" in entities
        assert "metformin" in entities["drug"]

    def test_disease_recognized(self, enhancer):
        entities = enhancer.extract_entities("diabetes treatment")
        assert "disease" in entities

    def test_gene_protein_recognized(self, enhancer):
        entities = enhancer.extract_entities("ARF protein")
        assert "gene_protein" in entities

    def test_no_entity(self, enhancer):
        entities = enhancer.extract_entities("hello world")
        # 至少 anatomy 之类不会匹配
        assert "drug" not in entities
        assert "disease" not in entities


# ==================== 实体识别(中文) ====================
class TestExtractEntitiesChinese:
    def test_chinese_drug(self, enhancer):
        entities = enhancer.extract_entities("服用阿司匹林")
        assert "drug_cn" in entities
        assert "阿司匹林" in entities["drug_cn"]

    def test_chinese_disease(self, enhancer):
        entities = enhancer.extract_entities("糖尿病治疗")
        assert "disease_cn" in entities
        assert "糖尿病" in entities["disease_cn"]

    def test_chinese_anatomy(self, enhancer):
        entities = enhancer.extract_entities("心脏搭桥")
        assert "anatomy_cn" in entities


# ==================== 时间过滤 ====================
class TestExtractFiltersYear:
    def test_recent_5_years_english(self, enhancer):
        # current_year = 2026, 5 年前 = 2021
        filters = enhancer.extract_filters("Recent 5 years on PD-1", current_year=2026)
        assert "year" in filters
        assert filters["year"]["$gte"] == "2021"

    def test_since_2020(self, enhancer):
        filters = enhancer.extract_filters("Studies since 2020", current_year=2026)
        assert filters["year"]["$gte"] == "2020"

    def test_year_specific(self, enhancer):
        filters = enhancer.extract_filters("Research in 2019 on diabetes", current_year=2026)
        assert filters["year"]["$gte"] == "2019"

    def test_no_year(self, enhancer):
        filters = enhancer.extract_filters("What is ARNO?", current_year=2026)
        assert "year" not in filters

    def test_chinese_recent_5_years(self, enhancer):
        filters = enhancer.extract_filters("近五年关于 PD-1", current_year=2026)
        assert filters["year"]["$gte"] == "2021"

    def test_chinese_since_2020(self, enhancer):
        filters = enhancer.extract_filters("2020年以来关于新冠", current_year=2026)
        assert filters["year"]["$gte"] == "2020"


# ==================== 期刊过滤 ====================
class TestExtractFiltersJournal:
    def test_nature(self, enhancer):
        filters = enhancer.extract_filters("What did Nature publish?")
        assert filters.get("journal_keyword") == "nature"

    def test_nejm(self, enhancer):
        filters = enhancer.extract_filters("NEJM papers on COVID")
        assert filters.get("journal_keyword") == "nejm"

    def test_no_journal(self, enhancer):
        filters = enhancer.extract_filters("What is ARNO?")
        assert "journal_keyword" not in filters

    def test_combined_year_and_journal(self, enhancer):
        filters = enhancer.extract_filters("Recent NEJM papers on COVID", current_year=2026)
        assert filters["year"]["$gte"] == "2021"
        assert filters["journal_keyword"] == "nejm"


# ==================== 综合 enhance 流程 ====================
class TestEnhance:
    def test_english_enhance(self, enhancer):
        eq = enhancer.enhance("Does metformin reduce CVD?")
        assert "metformin" in eq.entities.get("drug", [])
        assert "cardiovascular disease" in eq.synonyms
        assert eq.vector_query.startswith("Represent this question")
        assert "metformin" in eq.keyword_query
        assert "cardiovascular disease" in eq.keyword_query

    def test_chinese_enhance(self, enhancer):
        eq = enhancer.enhance("心梗病人能不能用阿司匹林?")
        assert "阿司匹林" in eq.entities.get("drug_cn", [])
        assert "aspirin" in eq.synonyms
        assert "心肌梗死" in eq.synonyms
        assert "myocardial infarction" in eq.synonyms

    def test_query_variants_generated(self, enhancer):
        eq = enhancer.enhance("What is MI?")
        # 至少原始 + 同义词追加版
        assert len(eq.query_variants) >= 2

    def test_empty_query_handled(self, enhancer):
        """空 query 不应该崩"""
        eq = enhancer.enhance("")
        # 应该返回空结果,不抛错
        assert eq.original == ""
        assert eq.synonyms == []

    def test_whitespace_query(self, enhancer):
        eq = enhancer.enhance("   ")
        # 清洗后是空字符串
        assert eq.cleaned.strip() == ""

    def test_enhancement_log_present(self, enhancer):
        eq = enhancer.enhance("What is MI?")
        assert len(eq.enhancement_log) >= 5
        # 应该记录每一步
        assert any("[1]" in line for line in eq.enhancement_log)
        assert any("[7]" in line for line in eq.enhancement_log)


# ==================== 性能:预编译 ====================
class TestPerformance:
    def test_precompilation_happens(self, enhancer):
        """确认 __init__ 里预编译了"""
        assert hasattr(enhancer, "_compiled_patterns")
        assert hasattr(enhancer, "_compiled_synonyms_en")
        assert hasattr(enhancer, "_cn_candidates")
        assert len(enhancer._compiled_patterns) > 0
        assert len(enhancer._compiled_synonyms_en) > 0
        assert len(enhancer._cn_candidates) > 0

    def test_second_enhance_faster_than_first(self, enhancer):
        """第二次 enhance 应该跟第一次差不多快(无明显缓存,但预编译避免 re.compile 开销)"""
        import time
        t0 = time.time()
        enhancer.enhance("What is MI?")
        first = time.time() - t0

        t0 = time.time()
        enhancer.enhance("What is MI?")
        second = time.time() - t0

        # 第二次不会显著慢(预编译后无重复开销)
        assert second < first * 2 + 0.05  # 给 50ms 容差


# ==================== 健壮性 ====================
class TestRobustness:
    def test_unicode_query(self, enhancer):
        eq = enhancer.enhance("二甲双胍对心血管疾病有何影响?")
        assert eq.original == "二甲双胍对心血管疾病有何影响?"

    def test_query_with_numbers(self, enhancer):
        eq = enhancer.enhance("Metformin 500mg dosage for T2DM")
        assert "metformin" in eq.entities.get("drug", [])

    def test_query_with_punctuation(self, enhancer):
        eq = enhancer.enhance("What is MI?! Why??")
        assert "myocardial infarction" in eq.synonyms

    def test_to_dict_serialization(self, enhancer):
        """结果可以序列化为 dict"""
        eq = enhancer.enhance("What is MI?")
        d = eq.to_dict()
        assert "original" in d
        assert "synonyms" in d
        assert isinstance(d["synonyms"], list)
