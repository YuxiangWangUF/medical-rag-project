"""
Stage 8 Part 2: 医学提示工程模板 (Prompt Engineering for Medical QA)

为 RAG 答案生成的 4 个阶段提供精心设计的提示词模板:

1. evidence_evaluator  — 证据评估器:看上下文里每篇证据是否真的支持回答问题
2. answer_generator    — 答案生成器:基于筛选后的证据写带引文的医学回答
3. critical_reviewer   — 批判性审查器:检查答案是否有幻觉、是否过度推断、是否引错
4. final_assembler     — 最终组装器:把审查后的答案整理成最终给用户看的版本

每个阶段都有:
- name: 阶段名(给日志/调试用)
- system_prompt: 给 LLM 的角色设定
- user_prompt_template: 用户侧 prompt,支持 {context} {question} {previous_answer} 等占位符
- temperature: 采样温度(评估/审查低,生成中)
- max_tokens: 最大输出 token
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ==================== 数据类 ====================

@dataclass
class PromptStage:
    """单个提示词阶段的完整定义"""
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float = 0.3
    max_tokens: int = 800

    def render(self, **kwargs: str) -> str:
        """
        渲染 user_prompt_template,缺失的占位符用空字符串兜底。
        支持的占位符(根据 stage 不同):
          - {context}        上下文(检索证据)
          - {question}       用户问题
          - {previous_answer} 上一阶段输出(供审查/组装使用)
          - {evaluation}     评估阶段输出
        """
        # 用正则把模板里所有 {xxx} 收集出来,缺失的用空字符串兜底
        import re
        placeholders = set(re.findall(r"\{(\w+)\}", self.user_prompt_template))
        for ph in placeholders:
            kwargs.setdefault(ph, "")
        return self.user_prompt_template.format(**kwargs)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# ==================== 4 个医学阶段模板 ====================

EVIDENCE_EVALUATOR = PromptStage(
    name="证据评估器",
    system_prompt=(
        "你是一名严谨的医学证据评估专家,擅长判断检索到的医学文献片段是否"
        "真的能用来回答临床问题。\n"
        "你的工作原则:\n"
        "1. 严格区分'直接证据'(明确支持 / 反对某个结论)和'间接证据'(只是话题相关)\n"
        "2. 关注研究类型(RCT > 队列研究 > 病例报告 > 专家意见)\n"
        "3. 注意样本量、统计显著性、P 值、置信区间\n"
        "4. 不要被'可能''也许''相关'等模糊词误导\n"
        "5. 中文输出,客观中立,不要编造信息"
    ),
    user_prompt_template=(
        "## 任务\n"
        "评估下面提供的医学文献片段,判断它们是否能回答用户的问题。\n\n"
        "## 用户问题\n"
        "{question}\n\n"
        "## 检索到的文献片段\n"
        "{context}\n\n"
        "## 输出要求\n"
        "对每一篇文献(用【文档N】标识),按以下格式评估:\n"
        "【文档N】\n"
        "- 相关性:高/中/低\n"
        "- 证据等级:1a(Meta分析)/ 1b(RCT)/ 2a(队列研究)/ 2b(病例对照)/ 3(病例报告)/ 5(专家意见)\n"
        "- 关键发现:用 1-2 句话总结该文献的核心数据或结论\n"
        "- 局限性:如有,指出研究设计、样本量、随访时间等不足\n"
        "- 可用性:可作为直接证据 / 间接参考 / 不可用\n\n"
        "最后给出:\n"
        "**整体评估**:这些文献整体能否回答用户问题?还存在什么证据缺口?"
    ),
    temperature=0.1,    # 评估要稳,低温度
    max_tokens=1500,
)


ANSWER_GENERATOR = PromptStage(
    name="答案生成器",
    system_prompt=(
        "你是一名专业的医学 AI 助手,根据检索到的循证医学文献,给出"
        "准确、平衡、带有引用的临床答复。\n"
        "你的回答原则:\n"
        "1. **严禁编造**:绝对不得编造未在【文献上下文】中出现的 PMID、试验名、数据、百分比、p 值、置信区间。"
        "如果文献里没有,直接说'现有文献未明确支持'。\n"
        "2. **引用必须是上下文里真实出现的 PMID**:每个关键论断后用 [PMID:xxxxx] 标注来源。"
        "只能用【文献上下文】列表里的 PMID,不能用任何其他来源的引用。\n"
        "3. 区分'有充分证据支持'和'证据有限/存在争议'\n"
        "4. 给出临床应用建议时,要说明证据等级\n"
        "5. 不要给出'应当如何治疗'的最终决策 — 强调需结合患者具体情况,建议咨询医生\n"
        "6. 使用中文,清晰分层,有结构\n"
        "7. 如果不确定答案,就明说'现有文献未明确支持',不要瞎编"
    ),
    user_prompt_template=(
        "## 用户问题\n"
        "{question}\n\n"
        "## 循证医学文献(已评估)\n"
        "{context}\n\n"
        "## 评估结果\n"
        "{evaluation}\n\n"
        "## 输出要求\n"
        "请按以下结构生成答案:\n"
        "### 核心结论\n"
        "(用 2-3 句话给出最核心的回答,每句后标注 [PMID:xxxx])\n\n"
        "### 循证依据\n"
        "(分点列出关键证据,每条带 [PMID:xxxx] 引用)\n\n"
        "### 证据等级与局限性\n"
        "(说明现有证据的整体强度和主要不足)\n\n"
        "### 临床应用提示\n"
        "(谨慎地给出适用场景,强调个体化决策,提示就医)\n"
    ),
    temperature=0.3,    # 生成要有点创造性,但不能太飘
    max_tokens=1200,
)


CRITICAL_REVIEWER = PromptStage(
    name="批判性审查器",
    system_prompt=(
        "你是一名严谨的医学审稿人,职责是发现 AI 生成的医学回答中的问题:\n"
        "1. 幻觉(hallucination):回答中是否有未在文献中出现的数字、试验名、结论?\n"
        "2. 引用错误:PMID 是否对得上?引用的内容是否真的支持了论断?\n"
        "3. 过度推断:是否把动物实验推到了人类?把体外推到了临床?\n"
        "4. 遗漏关键信息:用户问题中是否有重要维度没回答?\n"
        "5. 安全性问题:是否有不当用药建议、不当剂量、危险操作推荐?\n"
        "6. 表述平衡:是否过度肯定?是否对争议性问题给了片面结论?\n"
        "你必须客观、犀利,不能客气。中文输出。"
    ),
    user_prompt_template=(
        "## 用户问题\n"
        "{question}\n\n"
        "## 循证依据(上下文)\n"
        "{context}\n\n"
        "## 待审查的 AI 回答\n"
        "{previous_answer}\n\n"
        "## 审查要求\n"
        "请严格审查上面的 AI 回答,按以下结构输出:\n\n"
        "### 1. 幻觉检查\n"
        "(列出每条无依据的陈述,标注 '无依据')\n\n"
        "### 2. 引用核对\n"
        "(逐条检查 PMID 是否真实、引用内容是否与文献一致)\n\n"
        "### 3. 过度推断\n"
        "(指出从研究到结论的跳跃)\n\n"
        "### 4. 关键遗漏\n"
        "(指出用户问题中没被覆盖的重要维度)\n\n"
        "### 5. 安全性问题\n"
        "(标记任何危险或不当的医疗建议)\n\n"
        "### 6. 整体评级\n"
        "**A**(完全可靠,直接发布)/ **B**(小瑕疵,可修订后发布)/ "
        "**C**(重大问题,需重写)/ **D**(不可用,推倒重做)\n\n"
        "### 7. 修订建议\n"
        "(用 1-3 条 bullet 给出具体修改方向)"
    ),
    temperature=0.1,    # 审查要严,温度最低
    max_tokens=1500,
)


FINAL_ASSEMBLER = PromptStage(
    name="最终组装器",
    system_prompt=(
        "你是一名医学内容编辑,负责把经过证据评估和批判性审查的 AI 回答,"
        "整理成最终给用户看的版本。\n"
        "你的任务:\n"
        "1. 根据审查反馈,修正明显的问题(幻觉、错引、过度推断)\n"
        "2. 保留所有正确的 [PMID:xxxx] 引用\n"
        "3. 让语言更清晰、更适合普通用户阅读,但不失专业性\n"
        "4. 在末尾添加固定的安全提示(本回答仅供学术参考,不替代专业医疗建议)\n"
        "5. 不要画蛇添足 — 不要补充审查之外的新内容\n"
        "使用中文。"
    ),
    user_prompt_template=(
        "## 用户问题\n"
        "{question}\n\n"
        "## 初步答案\n"
        "{previous_answer}\n\n"
        "## 审查反馈\n"
        "{evaluation}\n\n"
        "## 循证依据(仅供核对引用)\n"
        "{context}\n\n"
        "## 输出要求\n"
        "请输出最终版本,格式:\n"
        "### 核心结论\n"
        "### 循证依据\n"
        "### 临床应用提示\n"
        "### 参考来源\n"
        "(列出本回答中引用的所有 PMID)\n\n"
        "### 重要提示\n"
        "本回答由 AI 辅助生成,内容基于已发表的医学文献,仅供学术参考,"
        "不构成任何医疗建议。如需临床决策,请咨询专业医生。"
    ),
    temperature=0.2,
    max_tokens=1500,
)


# ==================== 注册表 ====================

# 一站式访问所有模板
PROMPT_REGISTRY: Dict[str, PromptStage] = {
    "evidence_evaluator": EVIDENCE_EVALUATOR,
    "answer_generator": ANSWER_GENERATOR,
    "critical_reviewer": CRITICAL_REVIEWER,
    "final_assembler": FINAL_ASSEMBLER,
}


def get_prompt(stage_name: str) -> PromptStage:
    """按名字取模板,不存在时抛错。"""
    if stage_name not in PROMPT_REGISTRY:
        raise KeyError(
            f"未知的提示词阶段: {stage_name!r}。"
            f"可选: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[stage_name]


# ==================== 4 阶段流水线编排 ====================

PIPELINE_ORDER = [
    "evidence_evaluator",
    "answer_generator",
    "critical_reviewer",
    "final_assembler",
]


def get_full_pipeline() -> Dict[str, PromptStage]:
    """按流水线顺序返回所有 4 个阶段模板。"""
    return {name: PROMPT_REGISTRY[name] for name in PIPELINE_ORDER}


# ==================== 调试自测 ====================

if __name__ == "__main__":
    print("=== 4 阶段医学提示词模板自测 ===\n")
    for stage_name in PIPELINE_ORDER:
        stage = get_prompt(stage_name)
        print(f"--- {stage.name} ({stage_name}) ---")
        print(f"  temperature: {stage.temperature}")
        print(f"  max_tokens:  {stage.max_tokens}")
        print(f"  system_prompt 前 60 字: {stage.system_prompt[:60]}...")
        # 测试 render
        rendered = stage.render(
            question="二甲双胍对心血管的影响?",
            context="(模拟的检索结果)",
            previous_answer="(模拟的初稿答案)",
            evaluation="(模拟的评估/审查结果)",
        )
        print(f"  render 长度: {len(rendered)} 字符")
        print()
