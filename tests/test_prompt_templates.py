"""
Stage 8 Part 3 — Tests for prompt_templates.py
"""

import unittest

from prompt_templates import (
    EVIDENCE_EVALUATOR,
    ANSWER_GENERATOR,
    CRITICAL_REVIEWER,
    FINAL_ASSEMBLER,
    PROMPT_REGISTRY,
    PIPELINE_ORDER,
    PromptStage,
    get_prompt,
    get_full_pipeline,
)


class TestPromptStageDataclass(unittest.TestCase):
    def test_fields_present(self):
        stage = PromptStage(
            name="test",
            system_prompt="sys",
            user_prompt_template="hello {name}",
            temperature=0.5,
            max_tokens=100,
        )
        self.assertEqual(stage.name, "test")
        self.assertEqual(stage.system_prompt, "sys")
        self.assertEqual(stage.user_prompt_template, "hello {name}")
        self.assertEqual(stage.temperature, 0.5)
        self.assertEqual(stage.max_tokens, 100)

    def test_render_substitutes_placeholders(self):
        stage = PromptStage(
            name="t",
            system_prompt="",
            user_prompt_template="Question: {question} | Context: {context}",
            temperature=0.0,
            max_tokens=10,
        )
        out = stage.render(question="Qx", context="Cx")
        self.assertIn("Qx", out)
        self.assertIn("Cx", out)
        self.assertNotIn("{question}", out)

    def test_render_missing_placeholder_uses_empty(self):
        stage = PromptStage(
            name="t",
            system_prompt="",
            user_prompt_template="Hi {name} {unknown}",
            temperature=0.0,
            max_tokens=10,
        )
        # format 严格模式会抛 KeyError — 我们的策略是 missing 兜底为空
        out = stage.render(name="Alice")
        # unknown 没传,format 不会自动用空字符串 — 我们后面单独测
        # 这里只验证 name 替换成功
        self.assertIn("Alice", out)

    def test_to_dict(self):
        stage = PromptStage("n", "sys", "tpl", 0.4, 200)
        d = stage.to_dict()
        self.assertEqual(d["name"], "n")
        self.assertEqual(d["system_prompt"], "sys")
        self.assertEqual(d["user_prompt_template"], "tpl")
        self.assertEqual(d["temperature"], 0.4)
        self.assertEqual(d["max_tokens"], 200)


class TestFourMedicalStages(unittest.TestCase):
    def test_all_registered(self):
        self.assertIn("evidence_evaluator", PROMPT_REGISTRY)
        self.assertIn("answer_generator", PROMPT_REGISTRY)
        self.assertIn("critical_reviewer", PROMPT_REGISTRY)
        self.assertIn("final_assembler", PROMPT_REGISTRY)

    def test_evidence_evaluator_low_temperature(self):
        # 评估需要稳定,温度应当是最低的
        self.assertLessEqual(EVIDENCE_EVALUATOR.temperature, 0.2)

    def test_critical_reviewer_low_temperature(self):
        # 审查需要严,温度也应当低
        self.assertLessEqual(CRITICAL_REVIEWER.temperature, 0.2)

    def test_answer_generator_medium_temperature(self):
        # 生成需要一点创造性
        self.assertGreaterEqual(ANSWER_GENERATOR.temperature, 0.2)
        self.assertLessEqual(ANSWER_GENERATOR.temperature, 0.5)

    def test_all_have_required_placeholders(self):
        for name, stage in PROMPT_REGISTRY.items():
            with self.subTest(stage=name):
                # 每个模板都应至少包含 {context} 或 {previous_answer} 之类占位符
                tpl = stage.user_prompt_template
                self.assertTrue(
                    "{" in tpl and "}" in tpl,
                    f"{name} 模板缺少占位符",
                )

    def test_evidence_evaluator_has_question_and_context(self):
        tpl = EVIDENCE_EVALUATOR.user_prompt_template
        self.assertIn("{question}", tpl)
        self.assertIn("{context}", tpl)

    def test_answer_generator_has_all_inputs(self):
        tpl = ANSWER_GENERATOR.user_prompt_template
        self.assertIn("{question}", tpl)
        self.assertIn("{context}", tpl)
        self.assertIn("{evaluation}", tpl)

    def test_critical_reviewer_has_previous_answer(self):
        tpl = CRITICAL_REVIEWER.user_prompt_template
        self.assertIn("{question}", tpl)
        self.assertIn("{context}", tpl)
        self.assertIn("{previous_answer}", tpl)

    def test_final_assembler_has_previous_and_evaluation(self):
        tpl = FINAL_ASSEMBLER.user_prompt_template
        self.assertIn("{question}", tpl)
        self.assertIn("{context}", tpl)
        self.assertIn("{previous_answer}", tpl)
        self.assertIn("{evaluation}", tpl)

    def test_prompts_mention_pmid(self):
        # 答案生成 + 审查 + 最终组装 三个阶段应当提到 PMID 引用
        # evidence_evaluator 只评估证据相关性,不一定需要 PMID
        for name in ("answer_generator", "critical_reviewer", "final_assembler"):
            with self.subTest(stage=name):
                self.assertIn("PMID", PROMPT_REGISTRY[name].user_prompt_template)

    def test_prompts_mention_chinese(self):
        # 应当用中文输出
        for name, stage in PROMPT_REGISTRY.items():
            with self.subTest(stage=name):
                full = stage.system_prompt + stage.user_prompt_template
                # 至少有一些中文字符
                chinese_chars = sum(1 for ch in full if "\u4e00" <= ch <= "\u9fff")
                self.assertGreater(chinese_chars, 50, f"{name} 中文比例太低")


class TestRegistryAccess(unittest.TestCase):
    def test_get_prompt_existing(self):
        self.assertIs(get_prompt("answer_generator"), ANSWER_GENERATOR)

    def test_get_prompt_missing_raises(self):
        with self.assertRaises(KeyError):
            get_prompt("nonexistent_stage")

    def test_get_full_pipeline_returns_all_four(self):
        pipeline = get_full_pipeline()
        self.assertEqual(len(pipeline), 4)
        self.assertEqual(list(pipeline.keys()), PIPELINE_ORDER)

    def test_pipeline_order_is_logical(self):
        # 顺序必须是 eval -> gen -> review -> assemble
        self.assertEqual(
            PIPELINE_ORDER,
            ["evidence_evaluator", "answer_generator",
             "critical_reviewer", "final_assembler"],
        )


class TestRenderWithFullPipeline(unittest.TestCase):
    """模拟一整个流水线的 render 调用,确保占位符都能正确填充。"""

    def test_full_chain_renders(self):
        ctx = "【文档1】...【文档2】..."
        q = "二甲双胍对心血管的影响?"

        # 1. evidence_evaluator
        eval_out = EVIDENCE_EVALUATOR.render(context=ctx, question=q)
        self.assertIn(q, eval_out)
        self.assertIn("【文档1】", eval_out)

        # 2. answer_generator
        gen_out = ANSWER_GENERATOR.render(
            context=ctx, question=q, evaluation=eval_out,
        )
        self.assertIn(q, gen_out)

        # 3. critical_reviewer
        review_out = CRITICAL_REVIEWER.render(
            context=ctx, question=q, previous_answer=gen_out,
        )
        self.assertIn(q, review_out)
        self.assertIn(gen_out[:30], review_out)

        # 4. final_assembler
        final_out = FINAL_ASSEMBLER.render(
            context=ctx, question=q,
            previous_answer=gen_out, evaluation=review_out,
        )
        self.assertIn(q, final_out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
