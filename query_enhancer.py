# query_enhancer.py - 阶段五:查询理解与增强
#
# 功能:
#   1. 基础清洗
#   2. 医学实体识别(drug / disease / gene / organism / anatomy)
#   3. 同义词扩展(基于静态医学词典)
#   4. 生成多个查询版本(向量版 + 关键词版)
#   5. 提取元数据过滤条件(年份、期刊)
#   6. BGE query instruction
#
# 跑法:
#   python query_enhancer.py        # 跑 demo
#   # 或在别的脚本中:
#   from query_enhancer import QueryEnhancer
#   eq = QueryEnhancer().enhance("What is MI?")

import os
import re
import json
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import List, Dict, Optional
from datetime import datetime


# ==================== 医学同义词词典 ====================
# 实际应用中应从 UMLS、MeSH 等医学标准术语库构建
# 这里先手写一个常用子集
MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    # === 心血管(英文) ===
    "mi": ["myocardial infarction", "heart attack"],
    "ami": ["acute myocardial infarction"],
    "chf": ["congestive heart failure", "heart failure"],
    "hf": ["heart failure"],
    "af": ["atrial fibrillation"],
    "afib": ["atrial fibrillation"],
    "vte": ["venous thromboembolism", "deep vein thrombosis", "pulmonary embolism"],
    "dvt": ["deep vein thrombosis"],
    "pe": ["pulmonary embolism"],
    "htn": ["hypertension", "high blood pressure"],
    "cad": ["coronary artery disease", "coronary heart disease"],
    "cvd": ["cardiovascular disease"],
    "stroke": ["cerebrovascular accident", "cva"],

    # === 心血管(中文) ===
    "心梗": ["心肌梗死", "心肌梗塞", "myocardial infarction"],
    "心衰": ["心力衰竭", "心功能衰竭", "heart failure"],
    "房颤": ["心房颤动", "atrial fibrillation"],
    "中风": ["脑卒中", "脑梗", "脑出血", "stroke"],
    "冠心病": ["冠状动脉粥样硬化性心脏病", "冠状动脉性心脏病", "coronary heart disease"],
    "高血压": ["高血压病", "血压高", "hypertension"],

    # === 糖尿病 / 内分泌 ===
    "dm": ["diabetes mellitus", "diabetes"],
    "t2dm": ["type 2 diabetes mellitus", "type 2 diabetes", "non-insulin dependent diabetes"],
    "t1dm": ["type 1 diabetes mellitus", "type 1 diabetes", "insulin dependent diabetes"],
    "iddm": ["type 1 diabetes", "insulin dependent diabetes"],
    "niddm": ["type 2 diabetes", "non-insulin dependent diabetes"],
    # 中文
    "糖尿病": ["糖尿病 mellitus", "消渴症", "diabetes"],
    "二型糖尿病": ["2型糖尿病", "2型糖尿病 mellitus", "type 2 diabetes"],
    "一型糖尿病": ["1型糖尿病", "1型糖尿病 mellitus", "type 1 diabetes"],
    "甲亢": ["甲状腺功能亢进症", "hyperthyroidism"],
    "甲减": ["甲状腺功能减退症", "hypothyroidism"],

    # === 癌症 / 肿瘤 ===
    "ca": ["cancer", "carcinoma"],
    "nsclc": ["non-small cell lung cancer", "non-small cell lung carcinoma"],
    "sclc": ["small cell lung cancer"],
    "hcc": ["hepatocellular carcinoma"],
    "crc": ["colorectal cancer"],
    "aml": ["acute myeloid leukemia"],
    "cml": ["chronic myeloid leukemia"],
    "all": ["acute lymphoblastic leukemia"],
    "cll": ["chronic lymphocytic leukemia"],
    "nhl": ["non-hodgkin lymphoma"],
    "hl": ["hodgkin lymphoma"],
    # 中文
    "肺癌": ["肺肿瘤", "lung cancer", "nsclc"],
    "乳腺癌": ["乳腺肿瘤", "breast cancer"],
    "肝癌": ["肝肿瘤", "liver cancer", "hcc"],
    "胃癌": ["胃肿瘤", "gastric cancer"],
    "肠癌": ["结直肠癌", "colorectal cancer", "crc"],
    "白血病": ["血癌", "leukemia"],
    "淋巴瘤": ["淋巴癌", "lymphoma"],

    # === 神经 / 精神 ===
    "ad": ["alzheimer disease", "alzheimer's disease"],
    "pd": ["parkinson disease", "parkinson's disease"],
    "als": ["amyotrophic lateral sclerosis"],
    "ms": ["multiple sclerosis"],
    "ckd": ["chronic kidney disease"],
    "esrd": ["end stage renal disease"],
    "ckd-epi": [],
    "tbi": ["traumatic brain injury"],
    "mdd": ["major depressive disorder", "depression"],
    # 中文
    "阿尔茨海默": ["阿尔茨海默病", "老年痴呆", "alzheimer disease"],
    "帕金森": ["帕金森病", "震颤麻痹", "parkinson disease"],
    "癫痫": ["羊癫疯", "epilepsy"],
    "抑郁症": ["抑郁障碍", "depression"],
    "脑梗": ["脑梗塞", "脑梗死", "cerebral infarction"],
    "脑出血": ["脑溢血", "cerebral hemorrhage"],

    # === 感染 ===
    "hiv": ["human immunodeficiency virus"],
    "aids": ["acquired immune deficiency syndrome", "acquired immunodeficiency syndrome"],
    "tb": ["tuberculosis"],
    "mtb": ["mycobacterium tuberculosis"],
    "uti": ["urinary tract infection"],
    "rsv": ["respiratory syncytial virus"],
    "hpv": ["human papillomavirus"],
    "hbv": ["hepatitis b virus"],
    "hcv": ["hepatitis c virus"],
    "covid": ["covid-19", "sars-cov-2", "coronavirus disease 2019"],
    "sars": ["severe acute respiratory syndrome"],
    "mrsa": ["methicillin-resistant staphylococcus aureus"],
    # 中文
    "艾滋": ["艾滋病", "获得性免疫缺陷综合征", "aids"],
    "乙肝": ["乙型肝炎", "hbv"],
    "丙肝": ["丙型肝炎", "hcv"],
    "结核": ["肺结核", "结核病", "tuberculosis"],
    "新冠": ["新冠肺炎", "covid-19", "新型冠状病毒肺炎"],

    # === 呼吸 ===
    "copd": ["chronic obstructive pulmonary disease"],
    "ards": ["acute respiratory distress syndrome"],
    "osa": ["obstructive sleep apnea"],
    # 中文
    "慢阻肺": ["慢性阻塞性肺疾病", "copd"],

    # === 消化 ===
    "gerd": ["gastroesophageal reflux disease"],
    "ibd": ["inflammatory bowel disease"],
    "uc": ["ulcerative colitis"],
    "ibs": ["irritable bowel syndrome"],
    "nafld": ["non-alcoholic fatty liver disease"],
    "nash": ["non-alcoholic steatohepatitis"],
    # 中文
    "胃食管反流": ["反流性食管炎", "gerd"],
    "脂肪肝": ["非酒精性脂肪肝", "nafld"],
    "肝硬化": ["肝纤维化", "cirrhosis"],

    # === 风湿 / 免疫 ===
    "ra": ["rheumatoid arthritis"],
    "sle": ["systemic lupus erythematosus", "lupus"],
    "oa": ["osteoarthritis"],
    # 中文
    "类风湿": ["类风湿关节炎", "ra"],
    "红斑狼疮": ["系统性红斑狼疮", "sle"],
    "骨关节炎": ["退行性关节炎", "osteoarthritis"],

    # === 药物缩写(英文) ===
    "asa": ["aspirin", "acetylsalicylic acid"],
    "apap": ["acetaminophen", "paracetamol"],
    "nsaid": ["non-steroidal anti-inflammatory drug", "nonsteroidal anti-inflammatory drug"],
    "nsaids": ["non-steroidal anti-inflammatory drugs"],
    "ssri": ["selective serotonin reuptake inhibitor"],
    "ssris": ["selective serotonin reuptake inhibitors"],
    "snri": ["serotonin norepinephrine reuptake inhibitor"],
    "ppi": ["proton pump inhibitor"],
    "ppis": ["proton pump inhibitors"],
    "acei": ["angiotensin-converting enzyme inhibitor", "ace inhibitor"],
    "arb": ["angiotensin receptor blocker"],
    "statin": ["hmg-coa reductase inhibitor", "statin drug"],
    # 中文药物
    "阿司匹林": ["乙酰水杨酸", "aspirin"],
    "二甲双胍": ["格华止", "metformin"],
    "华法林": ["warfarin"],
    "胰岛素": ["insulin"],
    "他汀": ["他汀类药物", "statin"],
    "布洛芬": ["ibuprofen"],
    "对乙酰氨基酚": ["扑热息痛", "paracetamol", "acetaminophen"],
    "奥美拉唑": ["omeprazole"],
    # 新增
    "氯吡格雷": ["clopidogrel"],
    "恩格列净": ["empagliflozin"],
    "达格列净": ["dapagliflozin"],
    "利拉鲁肽": ["liraglutide"],
    "司美格鲁肽": ["semaglutide"],
    "阿莫西林": ["amoxicillin"],
    "头孢": ["头孢菌素", "cephalosporin"],
    "青霉素": ["penicillin"],
    "红霉素": ["erythromycin"],
    "强的松": ["prednisone"],
    "泼尼松": ["prednisone"],
    "地塞米松": ["dexamethasone"],
    "吗啡": ["morphine"],
    "芬太尼": ["fentanyl"],
    "利福平": ["rifampicin"],
    "异烟肼": ["isoniazid"],
    "奥希替尼": ["osimertinib"],
    "阿扎胞苷": ["azacitidine"],
    "来那度胺": ["lenalidomide"],
    "硼替佐米": ["bortezomib"],
    "伊布替尼": ["ibrutinib"],
    "纳武单抗": ["nivolumab"],
    "帕博利珠单抗": ["pembrolizumab"],
    "卡铂": ["carboplatin"],
    "顺铂": ["cisplatin"],
    "吉西他滨": ["gemcitabine"],
    "培美曲塞": ["pemetrexed"],
    "舒尼替尼": ["sunitinib"],
    "索拉菲尼": ["sorafenib"],
    "仑伐替尼": ["lenvatinib"],
    "艾乐明": ["baricitinib"],
    "托珠单抗": ["tocilizumab"],
    "阿达木单抗": ["adalimumab"],
    "英夫利西单抗": ["infliximab"],
    "戈利木单抗": ["golimumab"],
    "培塞利珠": ["certolizumab"],

    # === 新型降糖药 ===
    "sglt2": ["sodium-glucose cotransporter 2 inhibitor", "sglt2 inhibitor"],
    "glp-1": ["glucagon-like peptide 1 receptor agonist", "glp-1 receptor agonist"],
    "dpp-4": ["dipeptidyl peptidase 4 inhibitor", "dpp-4 inhibitor"],
    "西格列汀": ["sitagliptin"],
    "利格列汀": ["linagliptin"],
    "沙格列汀": ["saxagliptin"],
    "维格列汀": ["vildagliptin"],
    "卡格列净": ["canagliflozin"],
    "艾托格列净": ["ertugliflozin"],

    # === 蛋白 / 信号通路 ===
    "tnf": ["tumor necrosis factor", "tnf-alpha"],
    "vegf": ["vascular endothelial growth factor"],
    "egfr": ["epidermal growth factor receptor"],
    "her2": ["human epidermal growth factor receptor 2", "erbb2"],
    "pd-1": ["programmed cell death protein 1", "pdcd1"],
    "pd-l1": ["programmed death-ligand 1", "cd274"],
    "ctla-4": ["cytotoxic t-lymphocyte-associated protein 4"],
    "p53": ["tumor protein p53", "tp53"],
    "il-6": ["interleukin 6", "interleukin-6"],
    # 中文蛋白
    "egfr基因": ["表皮生长因子受体", "egfr"],
    "her2阳性": ["her2阳性乳腺癌", "erbb2 positive"],
}


# ==================== 医学实体模式 ====================
MEDICAL_PATTERNS: Dict[str, str] = {
    # 药物 - 单词边界 \b 防止子串误匹配
    "drug": r"\b(?:aspirin|metformin|atorvastatin|warfarin|insulin|"
            r"glipizide|lisinopril|amlodipine|metoprolol|losartan|"
            r"hydrochlorothiazide|omeprazole|pantoprazole|simvastatin|"
            r"rosuvastatin|clopidogrel|paracetamol|acetaminophen|"
            r"ibuprofen|naproxen|celecoxib|amoxicillin|azithromycin|"
            r"ciprofloxacin|doxycycline|prednisone|methylprednisolone|"
            r"dexamethasone|doxorubicin|cisplatin|paclitaxel|"
            r"tamoxifen|imatinib|trastuzumab|bevacizumab|rituximab|"
            r"remdesivir|molnupiravir|paxlovid|nirmatrelvir|"
            r"semaglutide|liraglutide|empagliflozin|dapagliflozin|"
            r"atenolol|carvedilol|digoxin|amiodarone|"
            r"furosemide|spironolactone|albuterol|salbutamol|"
            r"montelukast|fluticasone|budesonide|sertraline|"
            r"fluoxetine|citalopram|escitalopram|venlafaxine|"
            r"diazepam|lorazepam|midazolam|morphine|fentanyl)\b",

    # 疾病
    "disease": r"\b(?:diabetes(?:\s+mellitus)?|hypertension|asthma|copd|"
              r"arthritis|osteoporosis|alzheimer(?:'s)?(?:\s+disease)?|"
              r"parkinson(?:'s)?(?:\s+disease)?|epilepsy|depression|"
              r"schizophrenia|cancer|leukemia|lymphoma|carcinoma|"
              r"cardiovascular disease|coronary artery disease|"
              r"heart failure|stroke|myocardial infarction|"
              r"tuberculosis|malaria|pneumonia|hepatitis|"
              r"cirrhosis|nephritis|pancreatitis|"
              r"inflammatory bowel disease|crohn(?:'s)?(?:\s+disease)?|"
              r"ulcerative colitis|lupus|psoriasis|eczema)\b",

    # 基因 / 蛋白
    "gene_protein": r"\b(?:BRCA1|BRCA2|TP53|EGFR|HER2|ERBB2|KRAS|"
                  r"PIK3CA|AKT1|MAPK|ERK|JAK|STAT|VEGF|TNF|IL6|CD4|CD8|"
                  r"CD20|CD274|CTLA4|PDCD1|ARNO|ARF|PLD|GEF|"
                  r"p53|ras|myc|braf|mek|alk|ros1|met)\b",

    # 病原体
    "organism": r"\b(?:E\.?\s?coli|HIV|HSV|EBV|HPV|HBV|HCV|"
              r"SARS-CoV-2|SARS-CoV|SARS|MERS-CoV|"
              r"Influenza|Staphylococcus|Streptococcus|"
              r"Plasmodium|Mycobacterium|Pseudomonas|"
              r"Candida|Aspergillus|Toxoplasma|"
              r"Treponema|Helicobacter|Salmonella|"
              r"HTLV-1|HTLV)\b",

    # 解剖
    "anatomy": r"\b(?:brain|heart|liver|kidney|lung|pancreas|"
             r"intestine|colon|breast|prostate|skin|bone|"
             r"cardiovascular|cerebrovascular|hepatic|renal|"
             r"pulmonary|gastric|intestinal|thymus|spleen|"
             r"hippocampus|cerebellum|cortex)\b",

    # === 中文实体(无 \b,用汉字边界) ===
    "drug_cn": r"(?<![\u4e00-\u9fff])(?:阿司匹林|二甲双胍|华法林|胰岛素|"
              r"他汀(?:类药物)?|布洛芬|对乙酰氨基酚|扑热息痛|"
              r"奥美拉唑|阿莫西林|头孢(?:菌素)?|青霉素|红霉素|"
              r"地塞米松|强的松|泼尼松|吗啡|芬太尼|地西泮|"
              r"利福平|异烟肼|乙胺丁醇|吡嗪酰胺)(?![\u4e00-\u9fff])",

    "disease_cn": r"(?<![\u4e00-\u9fff])(?:糖尿病|高血压(?:病)?|"
                r"冠心病|心肌梗死|心肌梗塞|心力衰竭|心房颤动|"
                r"脑卒中|脑梗(?:塞|死)?|脑出血|中风|"
                r"阿尔茨海默(?:病)?|帕金森(?:病)?|癫痫|"
                r"抑郁症|精神分裂症|癌症|肿瘤|"
                r"肺癌|乳腺癌|肝癌|胃癌|肠癌|结直肠癌|"
                r"白血病|淋巴瘤|"
                r"艾滋病|肺结核|乙型肝炎|丙型肝炎|"
                r"慢性阻塞性肺疾病|慢阻肺|哮喘|肺炎|"
                r"肝硬化|脂肪肝|胃食管反流|"
                r"类风湿(?:关节炎)?|红斑狼疮|骨关节炎|"
                r"骨质疏松|贫血|白血病|甲亢|甲减)(?![\u4e00-\u9fff])",

    "gene_protein_cn": r"(?<![\u4e00-\u9fff])(?:表皮生长因子受体|"
                    r"程序性死亡受体|血管内皮生长因子|肿瘤坏死因子|"
                    r"白细胞介素|人类表皮生长因子受体|"
                    r"p53|ras|myc|egfr|her2|pd-1|pd-l1)(?![\u4e00-\u9fff])",

    "anatomy_cn": r"(?<![\u4e00-\u9fff])(?:脑|心|肝|肾|肺|胰(?:腺)?|"
                r"肠|结肠|乳腺|前列腺|皮肤|骨|"
                r"心血管|脑血管|胃|"
                r"胸腺|脾|海马|小脑|皮层)(?![\u4e00-\u9fff])",
}


# ==================== 时间范围模式 ====================
YEAR_PATTERNS: Dict[str, str] = {
    "recent_5_years": r"\b(?:recent(?:ly)?|latest|new(?:est)?|current|"
                    r"modern|past\s+5\s+years?|last\s+5\s+years?|"
                    r"in\s+the\s+past\s+5\s+years?)\b",
    "recent_3_years": r"\b(?:past\s+3\s+years?|last\s+3\s+years?|latest\s+3\s+years?)\b",
    "recent_10_years": r"\b(?:past\s+10\s+years?|last\s+10\s+years?|"
                     r"past\s+decade|recent\s+decade|over\s+the\s+past\s+decade)\b",
    "since_2020": r"\b(?:since\s+2020|from\s+2020|after\s+2020|post-2020|"
                r"in\s+2020s?|2020|2021|2022|2023|2024|2025)\b",
    "since_2010": r"\b(?:since\s+2010|from\s+2010|after\s+2010|post-2010)\b",
    "since_2015": r"\b(?:since\s+2015|from\s+2015|after\s+2015|post-2015)\b",
    "since_2000": r"\b(?:since\s+2000|from\s+2000|after\s+2000|post-2000)\b",
    "year_specific": r"\b(?:in\s+(?:19|20)\d{2}|during\s+(?:19|20)\d{2})\b",
    # 中文时间(注意:避免"近五年"/"近三年"在多个 pattern 里重复,否则字典序导致错的优先匹配)
    "recent_5_years_cn": r"(?<![\u4e00-\u9fff])(?:近三年|近五年|过去三年|过去五年|"
                         r"近年来|最近|最新|过去一年|过去1年|近一年|近1年)(?![\u4e00-\u9fff])",
    "since_2020_cn": r"(?<![\u4e00-\u9fff])(?:2020年以来|2020年至今|"
                      r"2020年之后|2020年以后)(?![\u4e00-\u9fff])",
}

# 期刊关键词
JOURNAL_KEYWORDS: Dict[str, str] = {
    "nature": r"\b(?:nature(?:\s+(?:medicine|biotechnology|genetics|"
             r"immunology|cell\s+biology|methods|communications?))?)\b",
    "science": r"\b(?:science|sci(?:ence)?\s*magazine)\b",
    "nejm": r"\b(?:nejm|new\s+england\s+journal\s+of\s+medicine)\b",
    "lancet": r"\b(?:lancet|the\s+lancet)\b",
    "jama": r"\b(?:jama|journal\s+of\s+the\s+american\s+medical\s+association)\b",
    "cell": r"(?<![a-zA-Z-])cell(?=\s+(?:published|reports|journal|issued\s+\d|is\s+a|is\s+one|$|[\.,;:]|2020))",
    "plos": r"\b(?:plos(?:\s+(?:biology|medicine|one|genetics|"
           r"computational\s+biology|pathogens))?)\b",
    "bmc": r"\b(?:bmc(?:\s+\w+)?)\b",
    # 中文期刊
    "中华医学杂志_cn": r"(?<![\u4e00-\u9fff])(?:中华医学杂志|"
                      r"中华内科杂志|中华外科杂志|"
                      r"中国实用内科|中国实用外科)(?![\u4e00-\u9fff])",
}


# ==================== EnhancedQuery ====================
@dataclass
class EnhancedQuery:
    """查询增强的结构化结果"""
    original: str
    cleaned: str
    entities: Dict[str, List[str]] = field(default_factory=dict)
    synonyms: List[str] = field(default_factory=list)
    query_variants: List[str] = field(default_factory=list)
    vector_query: str = ""                # 加了 BGE instruction
    keyword_query: str = ""               # 用于 BM25 / 关键词检索
    filter_conditions: Dict = field(default_factory=dict)
    enhancement_log: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        """人类可读的摘要"""
        lines = [
            f"原始: {self.original}",
            f"清洗: {self.cleaned}",
        ]
        if self.entities:
            lines.append(f"实体: {self.entities}")
        if self.synonyms:
            lines.append(f"同义词: {self.synonyms}")
        if self.filter_conditions:
            lines.append(f"过滤: {self.filter_conditions}")
        lines.append(f"向量查询: {self.vector_query}")
        lines.append(f"关键词查询: {self.keyword_query}")
        lines.append(f"query 版本数: {len(self.query_variants)}")
        return "\n".join(lines)


# ==================== QueryEnhancer ====================
class QueryEnhancer:
    """查询理解与增强器"""

    def __init__(self,
                 synonyms: Optional[Dict[str, List[str]]] = None,
                 patterns: Optional[Dict[str, str]] = None,
                 year_patterns: Optional[Dict[str, str]] = None,
                 journal_keywords: Optional[Dict[str, str]] = None,
                 vector_instruction: str = "Represent this question for searching relevant passages: "):
        self.synonyms = synonyms or MEDICAL_SYNONYMS
        self.patterns = patterns or MEDICAL_PATTERNS
        self.year_patterns = year_patterns or YEAR_PATTERNS
        self.journal_keywords = journal_keywords or JOURNAL_KEYWORDS
        self.vector_instruction = vector_instruction
        # 注意:current_year 在每次 enhance() 时重新计算,避免跨年不变

        # 预编译:英文 regex 模式(避免每次 enhance 重新编译;带 IGNORECASE)
        self._compiled_patterns = {
            k: re.compile(v, re.IGNORECASE) for k, v in self.patterns.items()
        }
        # 预编译:英文同义词 regex(短 key 用 \b,带 IGNORECASE)
        self._compiled_synonyms_en = [
            (re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), k, v)
            for k, v in self.synonyms.items()
            if not self._is_chinese(k) and v
        ]
        # 按 key 长度降序排(长的优先匹配,防短 key 子串误匹配)
        self._compiled_synonyms_en.sort(key=lambda x: -len(x[1]))

        # 预计算:中文 pattern 的候选词(避免每次 enhance 重新抽)
        self._cn_candidates: Dict[str, List[str]] = {}
        for src_name, src in [
            ("patterns", self.patterns),
            ("year_patterns", self.year_patterns),
            ("journal_keywords", self.journal_keywords),
        ]:
            for k, v in src.items():
                if self._is_chinese_pattern(v):
                    full_key = f"{src_name}:{k}"
                    self._cn_candidates[full_key] = self._extract_cn_candidates(v)

    # === 1. 基础清洗 ===
    def clean(self, query: str) -> str:
        q = query.strip()
        # 统一引号
        q = re.sub(r'["""„]', '"', q)
        q = re.sub(r"[''‚‘]", "'", q)
        # 去除多余空白
        q = re.sub(r"\s+", " ", q)
        # 去除末尾标点(问号等保留 — 是问句)
        return q

    # === 2. 实体识别 ===
    def _is_chinese_pattern(self, pattern: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' for c in pattern)

    def _extract_cn_candidates(self, pattern: str) -> List[str]:
        """从中文 pattern 里抽出所有候选词。

        关键规则:
        - 纯汉字 token(近三年、过去五年)→ 整体加入
        - 含数字/英文的 token(2020年以来、近1年)→ 只保留汉字部分(年以来、近代)
        - 单字 token(心、肝)→ 加入(允许 anatomy 单字识别)
        """
        candidates = set()
        # 抽 (?:X|Y|Z) 里的所有选项
        for group in re.findall(r'\(\?:([^)]+)\)', pattern):
            for token in group.split('|'):
                token = token.strip()
                if not token:
                    continue
                # 保留 token 中所有汉字(剥离数字/英文)
                cn_part = ''.join(c for c in token if '\u4e00' <= c <= '\u9fff')
                if cn_part:
                    candidates.add(cn_part)
        return list(candidates)

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {}
        for etype, pattern in self.patterns.items():
            if self._is_chinese_pattern(pattern):
                # 中文模式:从预计算表查 candidates
                candidates = self._cn_candidates.get(f"patterns:{etype}", [])
                found = [c for c in candidates if c in query]
                if found:
                    entities[etype] = list(set(found))
            else:
                # 英文模式:用预编译 regex
                compiled = self._compiled_patterns[etype]
                matches = compiled.findall(query)
                if matches:
                    seen = set()
                    uniq = []
                    for m in matches:
                        ml = m.lower().strip()
                        if ml and ml not in seen:
                            seen.add(ml)
                            uniq.append(ml)
                    entities[etype] = uniq
        return entities

    # === 3. 同义词扩展 ===
    def _is_chinese(self, text: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def _make_pattern(self, key: str) -> str:
        """根据 key 字符类型返回对应 pattern:
        - 英文 key:用 \b 单词边界(防子串误匹配)
        - 中文 key:子串包含即匹配(中文"词边界"不严格,直接包含更合理)
        """
        if self._is_chinese(key):
            return None  # 用 .find() 代替
        return r"\b" + re.escape(key) + r"\b"

    def expand_synonyms(self, query: str) -> List[str]:
        """返回所有命中的同义词。
        关键:长 key 优先匹配(如 "pd-1" 优先于 "pd"),
        防止短 key 的子串误匹配。
        支持中英文 key:
        - 英文用 \b(预编译)
        - 中文用 contains (子串包含即匹配,中文词边界不严格)
        """
        expanded = []
        seen_synonyms = set()  # 同义词去重(保持顺序)
        covered = set()  # 已覆盖的字符位置(英文用)

        # 英文:用预编译 regex + 位置去重
        for compiled, abbrev, fulls in self._compiled_synonyms_en:
            for match in compiled.finditer(query):
                positions = set(range(match.start(), match.end()))
                if positions & covered:
                    continue
                covered |= positions
                for f in fulls:
                    if f not in seen_synonyms:
                        seen_synonyms.add(f)
                        expanded.append(f)

        # 中文:子串包含即匹配
        for abbrev, fulls in self.synonyms.items():
            if not self._is_chinese(abbrev) or not fulls:
                continue
            if abbrev in query and abbrev not in seen_synonyms:
                seen_synonyms.add(abbrev)  # 防止同 key 重复触发
                for f in fulls:
                    if f not in seen_synonyms:
                        seen_synonyms.add(f)
                        expanded.append(f)
        return expanded

    # === 4. 生成多版本 query ===
    def generate_variants(self, cleaned: str, synonyms: List[str]) -> List[str]:
        variants = [cleaned]   # 原始版本
        for full in synonyms:
            # 完整短语版本:把同义词作为补充追加
            variants.append(f"{cleaned} {full}")
        return list(set(variants))

    # === 5. 提取过滤条件 ===
    # 数据驱动的过滤规则(避免 if-elif 链)
    YEAR_FILTER_RULES = {
        "recent_5_years":    lambda year, q: {"$gte": str(year - 5)},
        "recent_3_years":    lambda year, q: {"$gte": str(year - 3)},
        "recent_10_years":   lambda year, q: {"$gte": str(year - 10)},
        "since_2020":        lambda year, q: {"$gte": "2020"},
        "since_2015":        lambda year, q: {"$gte": "2015"},
        "since_2010":        lambda year, q: {"$gte": "2010"},
        "since_2000":        lambda year, q: {"$gte": "2000"},
        "year_specific":     lambda year, q: ({"$gte": m.group(1)} if (m := re.search(r'\b(20\d{2})\b', q)) else {"$gte": str(year)}),
    }

    def extract_filters(self, query: str, current_year: int = None) -> Dict:
        if current_year is None:
            current_year = datetime.now().year
        filters = {}

        # 时间范围
        for label, pattern in self.year_patterns.items():
            matched = False
            if self._is_chinese_pattern(pattern):
                # 从预计算表查 candidates
                candidates = self._cn_candidates.get(f"year_patterns:{label}", [])
                if any(c in query for c in candidates):
                    matched = True
            else:
                # 英文模式:用预编译 regex(优化:避免每次 re.search 重编译)
                compiled = re.compile(pattern, re.IGNORECASE)
                if compiled.search(query):
                    matched = True

            if matched:
                # 中文 _cn 后缀也复用同一规则
                base_label = label.removesuffix("_cn")
                rule_label = base_label if base_label in self.YEAR_FILTER_RULES else label
                if rule_label in self.YEAR_FILTER_RULES:
                    filters["year"] = self.YEAR_FILTER_RULES[rule_label](current_year, query)
                break

        # 期刊
        for journal, pattern in self.journal_keywords.items():
            matched = False
            if self._is_chinese_pattern(pattern):
                candidates = self._cn_candidates.get(f"journal_keywords:{journal}", [])
                if any(c in query for c in candidates):
                    matched = True
            else:
                compiled = re.compile(pattern, re.IGNORECASE)
                if compiled.search(query):
                    matched = True
            if matched:
                filters["journal_keyword"] = journal
                break

        return filters

    # === 6. 完整 enhance 流程 ===
    def enhance(self, query: str) -> EnhancedQuery:
        log = []

        # 1) 清洗
        cleaned = self.clean(query)
        log.append(f"[1] 清洗: {query!r} → {cleaned!r}")

        # 2) 实体识别
        entities = self.extract_entities(cleaned)
        if entities:
            log.append(f"[2] 实体识别: {entities}")
        else:
            log.append(f"[2] 实体识别: 未识别到医学实体")

        # 3) 同义词扩展
        synonyms = self.expand_synonyms(cleaned)
        if synonyms:
            log.append(f"[3] 同义词扩展: {synonyms}")
        else:
            log.append(f"[3] 同义词扩展: 无")

        # 4) 多版本 query
        variants = self.generate_variants(cleaned, synonyms)
        log.append(f"[4] 生成 {len(variants)} 个查询版本")

        # 5) 向量 query(加 BGE instruction)
        vector_query = self.vector_instruction + cleaned
        log.append(f"[5] 向量 query: {vector_query!r}")

        # 6) 关键词 query(原始 + 同义词)
        keyword_parts = [cleaned] + synonyms
        keyword_query = " ".join(keyword_parts)
        log.append(f"[6] 关键词 query: {keyword_query[:100]}{'...' if len(keyword_query)>100 else ''}")

        # 7) 过滤条件(每次调用时算 current_year,避免跨年 bug)
        filters = self.extract_filters(cleaned, current_year=datetime.now().year)
        if filters:
            log.append(f"[7] 提取过滤: {filters}")
        else:
            log.append(f"[7] 提取过滤: 无")

        return EnhancedQuery(
            original=query,
            cleaned=cleaned,
            entities=entities,
            synonyms=synonyms,
            query_variants=variants,
            vector_query=vector_query,
            keyword_query=keyword_query,
            filter_conditions=filters,
            enhancement_log=log,
        )


# ==================== Demo ====================
def run_demo():
    print("=" * 70)
    print("阶段五:查询理解与增强 — Demo")
    print("=" * 70)

    enhancer = QueryEnhancer()

    test_queries = [
        # === 基础能力 ===
        # 1. 简单英文
        "What is ARNO?",
        # 2. 缩写展开
        "What is the effect of MI on heart?",
        # 3. 药物 + 疾病
        "Does metformin reduce the risk of CVD?",
        # 4. 全中文医学
        "二甲双胍对心血管疾病有何影响?",
        # 5. 中文缩写
        "心梗病人能不能用阿司匹林?",

        # === 实体丰富度 ===
        # 6. 基因/蛋白
        "What is the role of EGFR mutations in lung cancer?",
        # 7. 肿瘤标记物
        "How does PSA relate to prostate cancer prognosis?",
        # 8. 传染病
        "What is the mechanism of action of remdesivir against COVID-19?",
        # 9. 中药/西药混合
        "黄芪对免疫功能有什么作用?",
        # 10. 联合用药
        "阿司匹林和氯吡格雷联合用药的效果和风险?",
        # 11. 糖尿病相关
        "SGLT2 inhibitors for heart failure in diabetes patients",
        # 12. 免疫检查点
        "PD-L1 expression and response to immunotherapy in NSCLC",

        # === 时间过滤（多种说法）===
        # 13. 近5年
        "Recent 5 years research on PD-1 immunotherapy",
        # 14. 近3年
        "Latest 3 years studies on CAR-T cell therapy",
        # 15. 近10年
        "Past 10 years clinical trials for Alzheimer's disease",
        # 16. 特定年份起
        "Clinical trials published since 2015 for rheumatoid arthritis",
        # 17. 特定年份
        "Landmark studies in 2019 about GLP-1 agonists",
        # 18. 中文近5年
        "近五年关于 PD-1 免疫疗法的研究进展",
        # 19. 中文近3年
        "近三年 mRNA 疫苗的研究进展",
        # 20. 中文特定年份
        "2020年以来关于新冠后遗症的研究",

        # === 期刊过滤 + 组合 ===
        # 21. Nature 发表
        "What did Nature publish about CRISPR?",
        # 22. NEJM + 时间
        "Recent NEJM papers on COVID-19 vaccine efficacy",
        # 23. Lancet
        "Lancet reviews on gut microbiome and metabolic disease",
        # 24. Cell + 基因
        "Cell published studies on p53 mutation in colorectal cancer",
        # 25. Science + 神经
        "Science articles on alpha-synuclein and Parkinson's disease",

        # === 多实体 + 复杂组合 ===
        # 26. 三实体组合
        "Impact of HbA1c control on cardiovascular outcomes in T2DM patients with hypertension",
        # 27. 药物相互作用
        "Metformin and vitamin B12 deficiency in diabetic patients",
        # 28. 诊断 + 治疗
        "ctDNA monitoring for EGFR mutant NSCLC targeted therapy response",
        # 29. 流行病学
        "Incidence and mortality trends of pancreatic cancer globally",
        # 30. 亚组分析
        "Subgroup analysis of sex differences in statin efficacy for CVD prevention",

        # === 医学统计术语 ===
        # 31. 统计学意义
        "Hazard ratio and overall survival in immunotherapy trials",
        # 32. 生物标志物
        "TMB and PD-1/PD-L1 inhibitor response correlation",
        # 33.  meta分析
        "Meta-analysis of ACE inhibitors in heart failure with reduced ejection fraction",

        # === 边界情况 ===
        # 34. 无实体纯时间
        "Recent advances in gene therapy for rare diseases",
        # 35. 无实体纯期刊
        "Cell reports on novel drug delivery systems",
        # 36. 缩写歧义(Asa=阿司匹林? Asa=美国过敏?)
        "ASA and bleeding risk in cardiovascular patients",
        # 37. 中文缩写歧义
        "高血压患者用 ARB 治疗的效果如何?",
        # 38. 全疾病描述（非缩写）
        "Myocardial infarction with ST-segment elevation treatment guidelines",
        # 39. 反向问法
        "Which statins are NOT associated with increased diabetes risk?",
        # 40. 长句（模拟真实用户输入）
        "我在查找关于二型糖尿病患者使用恩格列净（empagliflozin）对心血管结局影响的随机对照试验，尤其是近三年内发表在 NEJM 或 Lancet 上的研究",
    ]

    results = []
    for i, q in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"[Q{i}] {q}")
        print("─" * 70)

        eq = enhancer.enhance(q)
        for line in eq.enhancement_log:
            print(f"  {line}")
        print(f"\n  → 摘要:")
        print(f"     向量 query: {eq.vector_query[:80]}...")
        print(f"     关键词 query 前 80: {eq.keyword_query[:80]}...")
        if eq.filter_conditions:
            print(f"     过滤: {eq.filter_conditions}")
        results.append(eq.to_dict())

    # 保存 demo 结果
    os.makedirs("./output", exist_ok=True)
    out_path = "./output/query_enhancement_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 70}")
    print(f"[OK] Demo completed, results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
