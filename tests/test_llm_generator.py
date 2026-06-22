"""
Stage 8 Part 2.3 — Tests for llm_generator.py

覆盖:
1. extract_json 容错(markdown/单引号/Python风格/多余逗号等)
2. LLMGenerator 接口(用 mock 替代真实 HTTP,快且稳定)
3. 批量生成(顺序/并发)
4. 重试机制
"""

import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_generator import (
    GenerationConfig,
    GenerationResult,
    LLMGenerator,
    extract_json,
    quick_generate,
)


# ==================== extract_json ====================

class TestExtractJson(unittest.TestCase):
    """JSON 提取的 6 种边界 case"""

    def test_markdown_fence(self):
        text = '```json\n{"a": 1, "b": 2}\n```'
        self.assertEqual(extract_json(text), {"a": 1, "b": 2})

    def test_surrounding_garbage(self):
        text = '前缀废话 {"a": 1, "b": 2} 后缀废话'
        self.assertEqual(extract_json(text), {"a": 1, "b": 2})

    def test_trailing_comma_object(self):
        text = '{"a": 1, "b": 2,}'
        # 应当处理掉末尾多余的逗号
        self.assertEqual(extract_json(text), {"a": 1, "b": 2})

    def test_trailing_comma_array(self):
        text = '{"a": [1, 2, 3,]}'
        self.assertEqual(extract_json(text), {"a": [1, 2, 3]})

    def test_single_quotes(self):
        text = "{'a': 1, 'b': 2}"
        # 单引号替换为双引号后应当解析
        self.assertEqual(extract_json(text), {"a": 1, "b": 2})

    def test_python_boolean_none(self):
        text = '```json\n{"a": True, "b": None, "c": False}\n```'
        self.assertEqual(extract_json(text), {"a": True, "b": None, "c": False})

    def test_chinese_keys(self):
        text = '```json\n{"副作用": ["恶心", "腹泻"], "概率": 0.1}\n```'
        out = extract_json(text)
        self.assertEqual(out["副作用"], ["恶心", "腹泻"])
        self.assertEqual(out["概率"], 0.1)

    def test_no_json(self):
        self.assertIsNone(extract_json("没有任何 JSON 内容的纯文本"))
        self.assertIsNone(extract_json(""))
        self.assertIsNone(extract_json(None))

    def test_nested_object(self):
        text = '{"outer": {"inner": [1, 2, {"x": "y"}]}}'
        out = extract_json(text)
        self.assertEqual(out["outer"]["inner"][2]["x"], "y")


# ==================== GenerationConfig ====================

class TestGenerationConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GenerationConfig()
        self.assertEqual(cfg.temperature, 0.3)
        self.assertEqual(cfg.max_tokens, 1200)
        self.assertFalse(cfg.json_mode)

    def test_custom_values(self):
        cfg = GenerationConfig(temperature=0.7, max_tokens=2000, json_mode=True)
        self.assertEqual(cfg.temperature, 0.7)
        self.assertEqual(cfg.max_tokens, 2000)
        self.assertTrue(cfg.json_mode)


# ==================== GenerationResult ====================

class TestGenerationResult(unittest.TestCase):
    def test_defaults(self):
        r = GenerationResult(text="hi")
        self.assertEqual(r.text, "hi")
        self.assertTrue(r.success)
        self.assertIsNone(r.parsed_json)
        self.assertEqual(r.retries, 0)
        self.assertEqual(r.elapsed_ms, 0.0)


# ==================== LLMGenerator 接口 (mock HTTP) ====================

def _mock_response(text: str, prompt_tokens: int = 10, response_tokens: int = 20):
    """构造一个 mock 的 requests.Response"""
    resp = MagicMock()
    resp.json.return_value = {
        "message": {"content": text},
        "prompt_eval_count": prompt_tokens,
        "eval_count": response_tokens,
    }
    resp.raise_for_status = MagicMock()
    return resp


def _mock_tags_response():
    """构造 /api/tags 的 mock 响应"""
    resp = MagicMock()
    resp.json.return_value = {
        "models": [
            {"name": "qwen3:8b"},
            {"name": "bge-m3:latest"},
        ]
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestLLMGenerator(unittest.TestCase):

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_basic(self, mock_post, mock_get):
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("hello world", 5, 3)

        gen = LLMGenerator(model_name="qwen3:8b")
        result = gen.generate(user_prompt="say hi")

        self.assertTrue(result.success)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.prompt_tokens, 5)
        self.assertEqual(result.response_tokens, 3)
        self.assertIsInstance(result.elapsed_ms, float)
        # mock 模式下 elapsed_ms 可能为 0,但应当是 float 类型
        # 真实环境下一定 > 0
        # 确认 POST 到了正确 URL
        call_args = mock_post.call_args
        self.assertIn("/api/chat", call_args[0][0])

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_with_system_prompt(self, mock_post, mock_get):
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("ok")

        gen = LLMGenerator()
        result = gen.generate(
            user_prompt="Q",
            system_prompt="You are a doctor",
        )

        # payload 应当包含 system 消息
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "You are a doctor")
        self.assertEqual(payload["messages"][1]["role"], "user")

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_json_mode(self, mock_post, mock_get):
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response('```json\n{"a": 1}\n```')

        gen = LLMGenerator()
        result = gen.generate(
            user_prompt="give json",
            config=GenerationConfig(json_mode=True),
        )

        # 应当解析出 JSON
        self.assertIsNotNone(result.parsed_json)
        self.assertEqual(result.parsed_json["a"], 1)
        # payload 应当包含 format=json
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload.get("format"), "json")

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_retry_on_failure(self, mock_post, mock_get):
        """网络异常时应当重试,最后成功"""
        import requests
        mock_get.return_value = _mock_tags_response()
        # 第一次抛异常,第二次成功
        mock_post.side_effect = [
            requests.ConnectionError("net err"),
            _mock_response("retry success"),
        ]

        gen = LLMGenerator()
        result = gen.generate(
            user_prompt="x",
            config=GenerationConfig(max_retries=2, retry_delay=0.01),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.text, "retry success")
        self.assertEqual(result.retries, 1)
        self.assertEqual(mock_post.call_count, 2)

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_returns_failure_when_all_retries_fail(self, mock_post, mock_get):
        import requests
        mock_get.return_value = _mock_tags_response()
        mock_post.side_effect = requests.ConnectionError("always fail")

        gen = LLMGenerator()
        result = gen.generate(
            user_prompt="x",
            config=GenerationConfig(max_retries=1, retry_delay=0.01),
        )
        self.assertFalse(result.success)
        self.assertIn("always fail", result.error)
        self.assertEqual(result.retries, 2)  # 初始 + 1 重试

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_empty_user_prompt(self, mock_post, mock_get):
        mock_get.return_value = _mock_tags_response()
        gen = LLMGenerator()
        result = gen.generate(user_prompt="")
        self.assertFalse(result.success)
        self.assertIn("user_prompt", result.error)
        # 不应当发起请求
        mock_post.assert_not_called()

    @patch("llm_generator.requests.get")
    def test_connection_test_warns_on_missing_model(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {"models": []}, raise_for_status=lambda: None)
        with self.assertRaises(ConnectionError) as cm:
            LLMGenerator(model_name="nonexistent:99b")
        self.assertIn("nonexistent:99b", str(cm.exception))

    @patch("llm_generator.requests.get")
    def test_connection_test_warns_on_ollama_down(self, mock_get):
        import requests
        mock_get.side_effect = requests.ConnectionError("no ollama")
        with self.assertRaises(ConnectionError) as cm:
            LLMGenerator()
        self.assertIn("ollama serve", str(cm.exception))


class TestBatchGenerate(unittest.TestCase):
    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_sequential_batch(self, mock_post, mock_get):
        mock_get.return_value = _mock_tags_response()
        mock_post.side_effect = [
            _mock_response("response 1"),
            _mock_response("response 2"),
            _mock_response("response 3"),
        ]

        gen = LLMGenerator()
        results = gen.batch_generate(
            user_prompts=["q1", "q2", "q3"],
            parallel=False,
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].text, "response 1")
        self.assertEqual(results[1].text, "response 2")
        self.assertEqual(results[2].text, "response 3")

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_parallel_batch(self, mock_post, mock_get):
        mock_get.return_value = _mock_tags_response()
        mock_post.side_effect = [_mock_response(f"r{i}") for i in range(5)]

        gen = LLMGenerator()
        results = gen.batch_generate(
            user_prompts=[f"q{i}" for i in range(5)],
            parallel=True,
            max_workers=3,
        )
        self.assertEqual(len(results), 5)
        # 顺序可能因并发而乱,但每个 result.text 应当等于 "rN"
        texts = sorted([r.text for r in results])
        self.assertEqual(texts, [f"r{i}" for i in range(5)])

    def test_batch_input_length_mismatch(self):
        with patch("llm_generator.requests.get", return_value=_mock_tags_response()):
            gen = LLMGenerator()
            with self.assertRaises(AssertionError):
                gen.batch_generate(
                    user_prompts=["q1", "q2"],
                    system_prompts=["only one"],
                )


# ==================== 并发限流 (Semaphore) ====================

class TestSemaphore(unittest.TestCase):
    """Semaphore 真的限制了最大并发请求数"""

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_semaphore_limits_concurrent(self, mock_post, mock_get):
        """
        启动 N 个并发线程,Semaphore 应当把同时在飞的请求限制在 max_concurrent。
        这里 mock 的 POST 会阻塞 50ms,允许我们观测并发峰值。
        """
        import threading
        mock_get.return_value = _mock_tags_response()

        in_flight = 0
        peak_in_flight = 0
        lock = threading.Lock()

        def slow_post(*args, **kwargs):
            nonlocal in_flight, peak_in_flight
            with lock:
                in_flight += 1
                peak_in_flight = max(peak_in_flight, in_flight)
            time.sleep(0.05)
            with lock:
                in_flight -= 1
            return _mock_response("ok")

        mock_post.side_effect = slow_post

        max_concurrent = 3
        gen = LLMGenerator(max_concurrent=max_concurrent)

        # 启动 10 个并发请求,Semaphore 应当把峰值压到 max_concurrent
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [
                ex.submit(gen.generate, user_prompt=f"q{i}")
                for i in range(10)
            ]
            for f in as_completed(futures):
                f.result()

        self.assertLessEqual(
            peak_in_flight, max_concurrent,
            f"Semaphore 没生效:peak={peak_in_flight}, expected <= {max_concurrent}",
        )
        # 同时所有请求应当都成功了
        self.assertEqual(peak_in_flight >= 1, True)


# ==================== LRU 缓存 ====================

class TestLRUCache(unittest.TestCase):
    """LRU 缓存的命中、淘汰、key 区分"""

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_cache_hit_skips_http(self, mock_post, mock_get):
        """相同 prompt 第二次调用时,POST 计数不增加(从缓存取)"""
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("cached answer")

        gen = LLMGenerator(cache_size=8)
        r1 = gen.generate(user_prompt="hello")
        r2 = gen.generate(user_prompt="hello")

        # 两次调用返回同一段文本
        self.assertEqual(r1.text, "cached answer")
        self.assertEqual(r2.text, "cached answer")
        # POST 只应当被调用一次
        self.assertEqual(mock_post.call_count, 1)

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_cache_miss_different_prompts(self, mock_post, mock_get):
        """不同 prompt 应当都打 HTTP(不被错误缓存)"""
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("x")

        gen = LLMGenerator(cache_size=8)
        gen.generate(user_prompt="q1")
        gen.generate(user_prompt="q2")
        gen.generate(user_prompt="q3")

        self.assertEqual(mock_post.call_count, 3)

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_cache_eviction_at_capacity(self, mock_post, mock_get):
        """超出 cache_size 时应当淘汰最久未用的"""
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("y")

        cache_size = 3
        gen = LLMGenerator(cache_size=cache_size)
        # 写入 5 个不同 prompt
        for i in range(5):
            gen.generate(user_prompt=f"q{i}")

        # 缓存大小应当正好等于上限
        stats = gen.cache_stats()
        self.assertEqual(stats["size"], cache_size)
        self.assertEqual(stats["max_size"], cache_size)

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_cache_key_includes_config(self, mock_post, mock_get):
        """不同 config 的相同 prompt 不应当命中缓存"""
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("z")

        gen = LLMGenerator(cache_size=8)
        gen.generate(user_prompt="hi", config=GenerationConfig(temperature=0.1))
        gen.generate(user_prompt="hi", config=GenerationConfig(temperature=0.9))

        # 两次 POST(温度不同 → 不同 key → 不命中)
        self.assertEqual(mock_post.call_count, 2)

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_cache_clear(self, mock_post, mock_get):
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("w")

        gen = LLMGenerator(cache_size=8)
        gen.generate(user_prompt="p1")
        gen.generate(user_prompt="p2")
        self.assertEqual(gen.cache_stats()["size"], 2)

        n = gen.cache_clear()
        self.assertEqual(n, 2)
        self.assertEqual(gen.cache_stats()["size"], 0)

    def test_cache_key_is_deterministic(self):
        """相同输入应当产生相同 key"""
        cfg = GenerationConfig(temperature=0.5, max_tokens=100)
        k1 = LLMGenerator._cache_key("a", "b", cfg)
        k2 = LLMGenerator._cache_key("a", "b", cfg)
        self.assertEqual(k1, k2)
        # 长度应是 sha256 hex (64 chars)
        self.assertEqual(len(k1), 64)

    def test_cache_key_differs_on_config(self):
        cfg1 = GenerationConfig(temperature=0.1)
        cfg2 = GenerationConfig(temperature=0.9)
        self.assertNotEqual(
            LLMGenerator._cache_key("a", "b", cfg1),
            LLMGenerator._cache_key("a", "b", cfg2),
        )

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_cache_persistence_roundtrip(self, mock_post, mock_get):
        """写入磁盘 → 从磁盘加载 — 应当能恢复缓存内容"""
        import tempfile
        mock_get.return_value = _mock_tags_response()
        mock_post.return_value = _mock_response("persisted")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = str(Path(tmpdir) / "cache.jsonl")

            gen1 = LLMGenerator(cache_size=8, cache_file=cache_file)
            gen1.generate(user_prompt="persisted_query")

            # 模拟重启:新实例读磁盘缓存
            gen2 = LLMGenerator(cache_size=8, cache_file=cache_file)
            gen2.generate(user_prompt="persisted_query")

            # 第二次没打 HTTP → 缓存命中
            self.assertEqual(mock_post.call_count, 1)


# ==================== 流式生成 ====================

class TestStreaming(unittest.TestCase):

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_stream_yields_chunks(self, mock_post, mock_get):
        """流式应当逐块产出内容"""
        mock_get.return_value = _mock_tags_response()

        # 模拟 Ollama 流式响应 — 每行一个 JSON
        stream_lines = [
            json.dumps({"message": {"content": "你好"}, "done": False}),
            json.dumps({"message": {"content": "世界"}, "done": False}),
            json.dumps({"message": {"content": "!"}, "done": True}),
        ]

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = stream_lines
        mock_post.return_value = resp

        gen = LLMGenerator()
        chunks = list(gen.generate_stream(user_prompt="hi"))

        self.assertEqual(chunks, ["你好", "世界", "!"])

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_stream_handles_empty(self, mock_post, mock_get):
        """空 user_prompt 时 yield 空字符串,不应当发起 HTTP"""
        mock_get.return_value = _mock_tags_response()
        gen = LLMGenerator()
        chunks = list(gen.generate_stream(user_prompt=""))
        self.assertEqual(chunks, [""])
        mock_post.assert_not_called()

    @patch("llm_generator.requests.get")
    @patch("llm_generator.requests.post")
    def test_generate_stream_handles_connection_error(self, mock_post, mock_get):
        """网络异常时静默退出,不再抛"""
        import requests as _req
        mock_get.return_value = _mock_tags_response()
        mock_post.side_effect = _req.ConnectionError("net err")

        gen = LLMGenerator()
        chunks = list(gen.generate_stream(user_prompt="x"))
        # 不抛异常,也不产出任何内容
        self.assertEqual(chunks, [])


# ==================== allow_no_llm 模式 ====================

class TestAllowNoLlm(unittest.TestCase):

    @patch("llm_generator.requests.get")
    def test_allow_no_llm_skips_connection_check(self, mock_get):
        """allow_no_llm=True 时不应当调用 /api/tags"""
        # 没有 mock side_effect,但 GET 应当不被调用
        gen = LLMGenerator(model_name="any:model", allow_no_llm=True)
        # 关键断言:GET 没被调用过
        mock_get.assert_not_called()

    @patch("llm_generator.requests.get")
    def test_default_still_checks_connection(self, mock_get):
        """默认行为:启动时仍然测连通性"""
        mock_get.return_value = _mock_tags_response()
        gen = LLMGenerator(model_name="qwen3:8b")
        # 默认应当调了 GET(/api/tags)
        mock_get.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)