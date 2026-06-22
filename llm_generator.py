"""
Stage 8 Part 2.1: 本地 LLM 集成与生成 (LLMGenerator)

负责把 Ollama 上的本地 LLM 包装成统一接口,支持:
1. 单次生成(带 system_prompt + temperature + max_tokens)
2. JSON 模式生成(自动提取 JSON,容忍格式瑕疵)
3. 批量生成(顺序/并发两模式)
4. 完整的错误处理 + 重试机制
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

import requests

from types_typed import CacheStats


# ==================== 数据类 ====================

@dataclass
class GenerationConfig:
    """生成配置 — 控制 LLM 行为"""
    temperature: float = 0.3
    max_tokens: int = 1200
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    # JSON 模式:True 时强制 LLM 输出合法 JSON(模型支持的话)
    json_mode: bool = False
    # 重试策略
    max_retries: int = 2
    retry_delay: float = 1.0  # 秒


@dataclass
class GenerationResult:
    """单次生成结果"""
    text: str
    parsed_json: Optional[Dict[str, Any]] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    response_tokens: int = 0
    retries: int = 0
    success: bool = True
    error: Optional[str] = None


# ==================== JSON 提取工具 ====================

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 输出中提取 JSON,容忍常见的格式问题:
    - markdown 代码块 ```json ... ```
    - 前后有废话
    - 末尾多余的逗号
    - 单引号(部分小模型会用)
    """
    if not text:
        return None

    # 1. 优先匹配 markdown 代码块
    m = _JSON_FENCE_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip()

    # 2. 找第一个 { 到最后一个 } 的范围
    if "{" in candidate:
        start = candidate.index("{")
        end = candidate.rfind("}")
        if end > start:
            candidate = candidate[start:end + 1]

    # 3. 修正常见格式问题
    candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')
    candidate = candidate.replace("\u2018", "'").replace("\u2019", "'")
    # 去掉尾部多余的逗号: "a":1,  →  "a":1
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # 把 Python 风格的 None/True/False 替换成 JSON 的 null/true/false
    # 不用 \b(中文边界不可靠),改用前后非字母
    candidate = re.sub(r"(?<![A-Za-z])None(?![A-Za-z])", "null", candidate)
    candidate = re.sub(r"(?<![A-Za-z])True(?![A-Za-z])", "true", candidate)
    candidate = re.sub(r"(?<![A-Za-z])False(?![A-Za-z])", "false", candidate)

    # 4. 尝试解析
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 5. 单引号兜底
    if "'" in candidate and '"' not in candidate:
        try:
            return json.loads(candidate.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    return None


# ==================== LLM 生成器 ====================

class LLMGenerator:
    """
    包装本地 Ollama 服务,提供统一的生成接口。

    使用示例:
        gen = LLMGenerator(model_name="qwen3:8b")
        result = gen.generate(
            system_prompt="你是医生",
            user_prompt="二甲双胍的作用?",
            config=GenerationConfig(temperature=0.2, json_mode=True),
        )
        if result.parsed_json:
            print(result.parsed_json)
        else:
            print(result.text)
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120  # 秒

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        # 并发限流:最多同时 N 个请求在飞(防止把 Ollama 打爆)
        max_concurrent: int = 4,
        # 简单 LRU 缓存:避免相同 query 重复调用
        cache_size: int = 32,
        cache_file: Optional[str] = None,  # 持久化缓存到文件,跨进程共享
        allow_no_llm: bool = False,  # 跳过启动时的连通性测试(给离线/测试用)
    ) -> None:
        """
        Args:
            model_name: Ollama 模型名称(如 qwen3:8b、llama3:8b)
            base_url: Ollama 服务地址,默认 http://localhost:11434
            timeout: 单次请求超时秒数,默认 120(大模型需要更久)
            max_concurrent: 最大并发请求数(并发限流)
            cache_size: LRU 缓存容量
            cache_file: 缓存文件路径,None 不持久化
            allow_no_llm: True 时跳过启动时的 Ollama 连通性测试,
                          适合离线场景(--no-llm 模式 / CI)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.max_concurrent = int(max_concurrent)
        self.cache_size = int(cache_size)
        self.cache_file = cache_file

        # 并发限流信号量
        self._semaphore = threading.Semaphore(max_concurrent)

        # LRU 缓存 — OrderedDict 实现(move_to_end + popitem(last=False))
        self._cache: "OrderedDict[str, GenerationResult]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self.allow_no_llm = bool(allow_no_llm)
        if cache_file:
            self._load_cache_from_disk()

        # 启动时测试连通性,失败给出明确提示
        # allow_no_llm 模式跳过 — 给无 LLM 环境(测试/离线)用
        if not self.allow_no_llm:
            self._test_connection()

    # ---------- 连接测试 ----------

    def _test_connection(self) -> None:
        """检查 Ollama 服务 + 模型是否可用,不可用时给出明确报错。"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(
                f"无法连接 Ollama 服务({self.base_url}): {e}\n"
                f"请先启动: ollama serve"
            ) from e

        # 检查模型是否存在
        try:
            data = r.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            target_base = self.model_name.split(":")[0]
            if target_base not in models:
                available = ", ".join(models) or "(空)"
                raise ConnectionError(
                    f"Ollama 上没有模型 '{self.model_name}'。"
                    f"可用: {available}。请先: ollama pull {self.model_name}"
                )
        except (ValueError, KeyError):
            # tags 返回异常不影响主流程
            pass

    # ---------- 单次生成 ----------

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        调用本地 LLM 生成文本。

        Args:
            user_prompt: 用户侧 prompt(必填)
            system_prompt: 系统提示词(可选,角色设定)
            config: 生成配置,None 时用默认

        Returns:
            GenerationResult: 包含 text / parsed_json / 耗时 / token 计数等
        """
        if not user_prompt:
            return GenerationResult(
                text="", success=False, error="user_prompt 不能为空",
            )

        if config is None:
            config = GenerationConfig()

        # 1. 缓存查找 — 命中直接返回,跳过 LLM
        cache_key = self._cache_key(user_prompt, system_prompt, config)
        cached = self._cache_get(cache_key)
        if cached is not None:
            # 标记这次是从缓存拿的(便于调试)
            return cached

        # 拼成 messages 格式
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # 如果是 JSON 模式,追加 JSON 格式要求
        # 没 system_prompt 时塞到 user_prompt 末尾,确保 LLM 知道要输出 JSON
        if config.json_mode:
            json_instruction = (
                "\n\n**重要**:你的回复必须是合法的 JSON 格式,"
                "用 ```json ... ``` 代码块包裹,不要加任何额外说明。"
            )
            if system_prompt:
                messages[0]["content"] += json_instruction
            else:
                messages[-1]["content"] += json_instruction

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
                "repeat_penalty": config.repeat_penalty,
            },
        }
        if config.json_mode:
            payload["format"] = "json"

        # 带重试的请求 + 并发限流
        result = self._do_generate(payload, config)

        # 2. 缓存写入(成功才缓存,失败结果不入缓存)
        if result.success:
            self._cache_put(cache_key, result)

        return result

    def _do_generate(
        self, payload: Dict[str, Any], config: GenerationConfig,
    ) -> GenerationResult:
        """执行实际的 HTTP 请求(信号量保护 + 重试)"""
        retries = 0
        last_error: Optional[str] = None
        while retries <= config.max_retries:
            t0 = time.time()
            # 信号量保护 — 限制最大并发,防止把 Ollama 打爆
            with self._semaphore:
                try:
                    resp = requests.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    raw = resp.json()
                    elapsed_ms = (time.time() - t0) * 1000

                    text = raw.get("message", {}).get("content", "")
                    prompt_tokens = raw.get("prompt_eval_count", 0)
                    response_tokens = raw.get("eval_count", 0)

                    parsed = None
                    if config.json_mode:
                        parsed = extract_json(text)

                    return GenerationResult(
                        text=text,
                        parsed_json=parsed,
                        raw_response=raw,
                        elapsed_ms=elapsed_ms,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        retries=retries,
                        success=True,
                    )
                except requests.RequestException as e:
                    last_error = f"{type(e).__name__}: {e}"
                    retries += 1
                    if retries <= config.max_retries:
                        time.sleep(config.retry_delay * retries)
                    continue

        return GenerationResult(
            text="",
            raw_response={},
            retries=retries,
            success=False,
            error=last_error or "未知错误",
        )

    # ---------- 流式生成 ----------

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """
        流式生成 — 增量返回 LLM token,适合长答案 + 前端打字机效果。

        用法:
            for chunk in gen.generate_stream(user_prompt, system_prompt):
                print(chunk, end="", flush=True)

        注意:流式模式不写入 LRU 缓存(内容是分块的,无法保证完整)
        """
        if not user_prompt:
            yield ""
            return
        if config is None:
            config = GenerationConfig()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,  # 关键:开启流式
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
                "repeat_penalty": config.repeat_penalty,
            },
        }

        # 流式请求 — 用 stream=True + iter_lines
        with self._semaphore:
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                )
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Ollama 流式每行一个 JSON,含 message.content
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    # done=True 表示生成结束
                    if chunk.get("done"):
                        break
            except requests.RequestException:
                # 流式失败时不抛异常 — 调用方可能已经把部分内容用了
                return

    # ---------- LRU 缓存 ----------

    @staticmethod
    def _cache_key(
        user_prompt: str, system_prompt: str, config: GenerationConfig,
    ) -> str:
        """
        生成缓存 key — 用 (prompt + config) 的 sha256。

        同一 prompt + 不同 config 会得到不同 key(避免温度差异导致缓存污染)。
        """
        # config 字段稳定序列化(只取影响输出的字段)
        sig = (
            f"u={user_prompt}\n"
            f"s={system_prompt}\n"
            f"t={config.temperature}\n"
            f"m={config.max_tokens}\n"
            f"p={config.top_p}\n"
            f"r={config.repeat_penalty}\n"
            f"j={int(config.json_mode)}"
        )
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[GenerationResult]:
        """线程安全的 LRU 读取 — 命中时更新访问顺序"""
        with self._cache_lock:
            if key not in self._cache:
                return None
            # OrderedDict.move_to_end 是 LRU 命中的标准做法
            self._cache.move_to_end(key)
            return self._cache[key]

    def _cache_put(self, key: str, result: GenerationResult) -> None:
        """线程安全的 LRU 写入 — 超容量时驱逐最久未用的"""
        with self._cache_lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            # 超出容量时驱逐最旧的(OrderedDict 头部就是最久未用的)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        # 持久化(可选)— 异步刷新到磁盘
        if self.cache_file:
            self._save_cache_to_disk()

    def cache_clear(self) -> int:
        """清空缓存,返回清掉的条数"""
        with self._cache_lock:
            n = len(self._cache)
            self._cache.clear()
        if self.cache_file and Path(self.cache_file).exists():
            try:
                Path(self.cache_file).unlink()
            except OSError:
                pass
        return n

    def cache_stats(self) -> CacheStats:
        """返回缓存统计信息(用于调试 / metrics)"""
        with self._cache_lock:
            return CacheStats(
                size=len(self._cache),
                max_size=self.cache_size,
                utilization=round(len(self._cache) / max(1, self.cache_size), 3),
            )

    # ---------- 缓存持久化 ----------

    def _load_cache_from_disk(self) -> None:
        """
        从磁盘加载持久化缓存 — 启动时调用。

        格式:JSON Lines,每行一个 key + GenerationResult(转 dict)
        """
        if not self.cache_file:
            return
        path = Path(self.cache_file)
        if not path.exists():
            return
        loaded = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        key = entry["key"]
                        result_dict = entry["result"]
                        # 把 dict 转回 GenerationResult
                        result = GenerationResult(**result_dict)
                        self._cache[key] = result
                        loaded += 1
                        if loaded >= self.cache_size:
                            # 超过容量,丢多余(简单截断,不驱逐老条目)
                            break
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # 单条记录损坏不影响整体
                        continue
        except OSError:
            # 文件读不了就不加载,不影响主流程
            return

    def _save_cache_to_disk(self) -> None:
        """把当前缓存写入磁盘(JSON Lines 格式)— 简单覆盖"""
        if not self.cache_file:
            return
        path = Path(self.cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as f:
                for key, result in self._cache.items():
                    # GenerationResult 是 dataclass,用 asdict 转
                    entry = {
                        "key": key,
                        "result": self._result_to_dict(result),
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # 写失败不影响主流程
            pass

    @staticmethod
    def _result_to_dict(result: GenerationResult) -> Dict[str, Any]:
        """GenerationResult → dict(JSON 可序列化)"""
        return {
            "text": result.text,
            "parsed_json": result.parsed_json,
            "raw_response": result.raw_response,
            "elapsed_ms": result.elapsed_ms,
            "prompt_tokens": result.prompt_tokens,
            "response_tokens": result.response_tokens,
            "retries": result.retries,
            "success": result.success,
            "error": result.error,
        }

    # ---------- 批量生成 ----------

    def batch_generate(
        self,
        user_prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        configs: Optional[List[GenerationConfig]] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> List[GenerationResult]:
        """
        批量生成。

        Args:
            user_prompts: 用户 prompt 列表
            system_prompts: 系统 prompt 列表,长度应等于 user_prompts
            configs: 每个 prompt 的配置,长度应等于 user_prompts
            parallel: True 时并发,False 时顺序(后者更稳,适合长任务)
            max_workers: 并发数

        Returns:
            List[GenerationResult]: 与输入顺序一一对应
        """
        n = len(user_prompts)
        if system_prompts is None:
            system_prompts = [""] * n
        if configs is None:
            configs = [None] * n
        assert len(system_prompts) == n, "system_prompts 长度必须等于 user_prompts"
        assert len(configs) == n, "configs 长度必须等于 user_prompts"

        if not parallel or n == 1:
            # 顺序模式 — 稳,适合长 prompt 或显存紧的情况
            return [
                self.generate(up, sp, cfg)
                for up, sp, cfg in zip(user_prompts, system_prompts, configs)
            ]

        # 并发模式 — 适合短 prompt 大量并发
        results: List[Optional[GenerationResult]] = [None] * n
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(self.generate, up, sp, cfg): i
                for i, (up, sp, cfg) in enumerate(zip(user_prompts, system_prompts, configs))
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[idx] = GenerationResult(
                        text="", success=False, error=f"并发异常: {e}",
                    )
        return [r if r is not None else GenerationResult(text="", success=False) for r in results]


# ==================== 便捷函数 ====================

def quick_generate(
    user_prompt: str,
    system_prompt: str = "",
    model_name: str = "qwen3:8b",
    temperature: float = 0.3,
    max_tokens: int = 1200,
    json_mode: bool = False,
) -> GenerationResult:
    """一行调用的便捷函数。"""
    gen = LLMGenerator(model_name=model_name)
    cfg = GenerationConfig(
        temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
    )
    return gen.generate(user_prompt, system_prompt, cfg)


# ==================== 自测 ====================

if __name__ == "__main__":
    print("=== LLMGenerator 自测 ===\n")

    gen = LLMGenerator(model_name="qwen3:8b")

    # Test 1: 普通生成
    print("--- Test 1: 普通生成 ---")
    result = gen.generate(
        user_prompt="一句话介绍二甲双胍,不超过 30 字。",
        system_prompt="你是一名医学助手,回答简洁。",
    )
    print(f"  成功: {result.success}")
    print(f"  耗时: {result.elapsed_ms:.0f}ms")
    print(f"  输出: {result.text.strip()[:100]}")

    # Test 2: JSON 模式
    print("\n--- Test 2: JSON 模式 ---")
    result2 = gen.generate(
        user_prompt='输出二甲双胍的 3 个常见副作用,字段:side_effects (数组)',
        system_prompt="你是一名医学助手",
        config=GenerationConfig(json_mode=True, temperature=0.1),
    )
    print(f"  成功: {result2.success}")
    print(f"  解析 JSON: {result2.parsed_json}")

    # Test 3: extract_json 边界 case
    print("\n--- Test 3: extract_json 容错 ---")
    test_cases = [
        '```json\n{"a": 1, "b": 2}\n```',
        '前缀废话 {"a": 1} 后缀废话',
        '{"a": 1,}',  # 多余逗号
        "{'a': 1}",   # 单引号
    ]
    for tc in test_cases:
        out = extract_json(tc)
        print(f"  {tc[:30]}... → {out}")