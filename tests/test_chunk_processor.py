# tests/test_chunk_processor.py
#
# 阶段三:文档解析与分割 — 单元测试
# 跑法:
#   pytest tests/test_chunk_processor.py -v
#
# 覆盖:
#   - parse_xml:从 XML 提取字段
#   - DocumentChunker:切分策略(整体不分割 vs 切分)
#   - 二次切分兜底
#   - chunk_id 唯一性
#   - token_count 准确

import os
import sys
import tempfile
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunk_processor import (
    parse_xml, load_to_dataframe, clean_dataframe,
    DocumentChunker, chunk_dataframe, save_outputs,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, NO_SPLIT_THRESHOLD,
)


# ==================== 真实 XML 样本(简化版) ====================
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:mml="http://www.w3.org/1998/Math/MathML">
  <front>
    <journal-meta>
      <journal-title>Test Journal</journal-title>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="pmid">12345678</article-id>
      <article-id pub-id-type="publisher-id">pub-1</article-id>
      <title-group>
        <article-title>Test Article on ARF Protein</article-title>
      </title-group>
    </article-meta>
    <abstract>
      <p>This is a test abstract about ARF and PLD. ARF mediates activation.</p>
    </abstract>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Background info on ARF protein and its role in cells.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>We used standard molecular biology methods.</p>
    </sec>
  </body>
</article>"""


@pytest.fixture
def sample_xml_file(tmp_path):
    """写一个临时 XML 文件,返回路径"""
    p = tmp_path / "test.xml"
    p.write_text(SAMPLE_XML, encoding="utf-8")
    return str(p)


# ==================== parse_xml ====================
class TestParseXml:
    def test_extract_pmid(self, sample_xml_file):
        d = parse_xml(sample_xml_file)
        assert d["pmid"] == "12345678"

    def test_extract_title(self, sample_xml_file):
        d = parse_xml(sample_xml_file)
        assert d["title"] == "Test Article on ARF Protein"

    def test_extract_abstract(self, sample_xml_file):
        d = parse_xml(sample_xml_file)
        assert "ARF and PLD" in d["abstract"]
        assert "ARF mediates activation" in d["abstract"]

    def test_extract_body_text(self, sample_xml_file):
        d = parse_xml(sample_xml_file)
        # body 应包含所有 <p> 标签的文本
        assert "Background info" in d["body_text"]
        assert "standard molecular biology" in d["body_text"]

    def test_missing_pmid_returns_empty(self, tmp_path):
        """没有 pmid 时应返回空字符串"""
        xml = """<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>No PMID</article-title></title-group>
    </article-meta>
  </front>
</article>"""
        p = tmp_path / "no_pmid.xml"
        p.write_text(xml, encoding="utf-8")
        d = parse_xml(str(p))
        assert d["pmid"] == ""
        assert d["title"] == "No PMID"

    def test_handles_xml_entities(self, tmp_path):
        """XML 实体应被 BeautifulSoup 自动解码"""
        xml = """<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmid">99999</article-id>
      <title-group>
        <article-title>Raul&apos;s study of ARF</article-title>
      </title-group>
    </article-meta>
  </front>
</article>"""
        p = tmp_path / "entity.xml"
        p.write_text(xml, encoding="utf-8")
        d = parse_xml(str(p))
        # BeautifulSoup 应该解码 &apos; -> '
        assert "Raul" in d["title"]
        assert "'" in d["title"]


# ==================== DocumentChunker:切分逻辑 ====================
class TestDocumentChunker:
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(
            model_name=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            no_split_threshold=NO_SPLIT_THRESHOLD,
        )

    @pytest.fixture
    def short_doc(self):
        """token 数 < 阈值的短文档"""
        return {
            "doc_id": "short_doc",
            "title": "Short Title",
            "abstract": "This is a very short abstract. " * 5,  # ~30 tokens
            "body_text": "",
        }

    @pytest.fixture
    def long_doc(self):
        """token 数 > 阈值的长文档"""
        return {
            "doc_id": "long_doc",
            "title": "Long Title",
            "abstract": "This is a long abstract. " * 200,  # ~1200 tokens
            "body_text": "More body text. " * 200,  # ~600 tokens
        }

    def test_short_doc_not_split(self, chunker, short_doc):
        """短文档(< 300 token)整体不分割"""
        chunks = chunker.chunk_document(short_doc)
        assert len(chunks) == 1
        assert chunks[0]["split_method"] == "no_split"
        assert chunks[0]["chunk_id"] == "short_doc"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["total_chunks"] == 1
        assert chunks[0]["token_count"] < NO_SPLIT_THRESHOLD

    def test_long_doc_split(self, chunker, long_doc):
        """长文档(> 300 token)递归切分"""
        chunks = chunker.chunk_document(long_doc)
        assert len(chunks) > 1
        assert chunks[0]["split_method"] == "recursive"
        # 每个 chunk 都有 unique chunk_id
        ids = [c["chunk_id"] for c in chunks]
        assert len(set(ids)) == len(ids)
        # chunk_index 是顺序的
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_id_format(self, chunker, long_doc):
        """chunk_id 格式: {doc_id}_chunk_{idx:03d}"""
        chunks = chunker.chunk_document(long_doc)
        for i, c in enumerate(chunks):
            assert c["chunk_id"] == f"long_doc_chunk_{i:03d}"

    def test_text_contains_title_and_abstract(self, chunker, short_doc):
        """切分后的 text 至少包含 title + abstract"""
        chunks = chunker.chunk_document(short_doc)
        assert "Short Title" in chunks[0]["text"]
        assert "very short abstract" in chunks[0]["text"]

    def test_no_overlap_when_not_split(self, chunker, short_doc):
        """整体不分割时,chunks 列表只有 1 个"""
        chunks = chunker.chunk_document(short_doc)
        assert len(chunks) == 1
        # 不应该有 overlap 的概念(只有一个 chunk)
        assert chunks[0]["total_chunks"] == 1

    def test_empty_body_text(self, chunker):
        """只有 title + abstract 的文档也能处理"""
        doc = {
            "doc_id": "no_body",
            "title": "Only Title",
            "abstract": "Only abstract content here.",
            "body_text": "",
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        # 即使只有 abstract 也应保留
        full_text = " ".join(c["text"] for c in chunks)
        assert "Only Title" in full_text
        assert "Only abstract" in full_text


# ==================== token 长度准确性 ====================
class TestTokenLength:
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(
            model_name=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def test_token_count_includes_title(self, chunker):
        doc = {
            "doc_id": "test",
            "title": "This is a test title",
            "abstract": "",
            "body_text": "",
        }
        chunks = chunker.chunk_document(doc)
        # token_count 应包含 title 的 token
        assert chunks[0]["token_count"] > 0

    def test_token_count_reasonable(self, chunker):
        """50 字符的文本大约 30-60 tokens(英文)"""
        text = "This is a test sentence with some words."  # ~10 words, ~10-15 tokens
        doc = {
            "doc_id": "test",
            "title": text,
            "abstract": "",
            "body_text": "",
        }
        chunks = chunker.chunk_document(doc)
        assert 5 <= chunks[0]["token_count"] <= 30


# ==================== 集成测试:小批量 ====================
class TestIntegration:
    def test_parse_and_chunk_pipeline(self, tmp_path, chunker=None):
        """端到端:parse XML → chunk → 验证"""
        if chunker is None:
            chunker = DocumentChunker(
                model_name=EMBEDDING_MODEL,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        # 写多个 XML
        pmids = ["11111111", "22222222", "33333333"]
        for pmid in pmids:
            xml = f"""<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmid">{pmid}</article-id>
      <title-group><article-title>Article {pmid}</article-title></title-group>
    </article-meta>
    <abstract><p>Abstract {pmid}. {'Long content. ' * 100}</p></abstract>
  </front>
</article>"""
            (tmp_path / f"{pmid}.xml").write_text(xml, encoding="utf-8")

        # parse + chunk(parse_xml 返回 dict,需要补 doc_id)
        from chunk_processor import parse_xml
        all_chunks = []
        for xml_file in tmp_path.glob("*.xml"):
            doc = parse_xml(str(xml_file))
            if doc:
                # parse_xml 没加 doc_id,补一下
                doc["doc_id"] = doc.get("pmid") or os.path.splitext(xml_file.name)[0]
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)

        # 验证
        assert len(all_chunks) > 0
        # 每个 chunk 都有 unique chunk_id(因为 doc_id 不一样)
        chunk_ids = [c["chunk_id"] for c in all_chunks]
        assert len(set(chunk_ids)) == len(chunk_ids)
        # 至少 3 个不同 doc_id
        doc_ids = {c["doc_id"] for c in all_chunks}
        assert len(doc_ids) >= 3

    def test_long_doc_chunks_have_overlap(self):
        """相邻 chunks 应该有 overlap(20 token 重叠)"""
        chunker = DocumentChunker(
            model_name=EMBEDDING_MODEL,
            chunk_size=100,  # 用小 chunk_size 让切分明显
            chunk_overlap=20,
            no_split_threshold=300,
        )
        body = " ".join([f"word{i}" for i in range(500)])
        doc = {
            "doc_id": "overlap_test",
            "title": "Overlap Test",
            "abstract": "",
            "body_text": body,
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2
        # chunk 0 可能只是 title,跳过;测 chunk 1 和 chunk 2 之间的 overlap
        for i in range(1, len(chunks) - 1):
            c1_words = chunks[i]["text"].split()
            c2_words = chunks[i + 1]["text"].split()
            # 至少应该有 1 个 word 重复
            overlap = set(c1_words[-20:]) & set(c2_words[:20])
            if overlap:
                assert len(overlap) >= 1, f"overlap 太小: {overlap}"
                return  # 找到一对有 overlap 的就过
        assert False, "任意相邻 chunks 都没有 overlap"


# ==================== 健壮性 ====================
class TestRobustness:
    def test_malformed_xml_handled(self, tmp_path):
        """畸形 XML 不应该让整个流程崩"""
        xml = """<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmid">12345</article-id>
    <!-- 注释未关闭 -->
"""
        p = tmp_path / "malformed.xml"
        p.write_text(xml, encoding="utf-8")
        # 应该有异常或返回 None,不挂
        try:
            d = parse_xml(str(p))
            # 如果没抛错,可能返回部分数据
        except Exception:
            # 抛错也行,但不能挂
            pass

    def test_empty_document(self):
        """空文档(无 title/abstract)不应崩"""
        doc = {
            "doc_id": "empty",
            "title": "",
            "abstract": "",
            "body_text": "",
        }
        chunker = DocumentChunker(
            model_name=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        # 应能处理(可能返回 1 个空 chunk)
        try:
            chunks = chunker.chunk_document(doc)
            # 不抛错就算过
        except Exception as e:
            pytest.fail(f"空文档不应该抛错: {e}")
