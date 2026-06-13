# Query Enhancer 测试结果报告 (40 Queries)

## 汇总统计

| 维度 | 数值 |
|------|------|
| 总 query 数 | 40 |
| 有同义词扩展 | 22 |
| 有时间过滤 | 11 |
| 有期刊过滤 | 7 |
| 有实体识别 | 24 |
| 总同义词条数 | 63 |
| 总 query 变体数 | 103 |

## 详细结果

| # | Query | Entities | Synonyms | Time Filter | Journal Filter | Variants |
|---|-------|----------|----------|-------------|----------------|----------|
| 1 | What is ARNO? | ['gene_protein'] | 0 | - | - | 1 |
| 2 | What is the effect of MI on heart? | ['anatomy'] | 2 | - | - | 3 |
| 3 | Does metformin reduce the risk of CVD? | ['drug'] | 1 | - | - | 2 |
| 4 | 二甲双胍对心血管疾病有何影响? | ['drug_cn'] | 2 | - | - | 3 |
| 5 | 心梗病人能不能用阿司匹林? | ['drug_cn'] | 5 | - | - | 6 |
| 6 | What is the role of EGFR mutations in lung... | ['disease', 'gene_protein', 'anatomy'] | 1 | - | - | 2 |
| 7 | How does PSA relate to prostate cancer pro... | ['disease', 'anatomy'] | 0 | - | - | 1 |
| 8 | What is the mechanism of action of remdesi... | ['drug'] | 3 | - | - | 4 |
| 9 | 黄芪对免疫功能有什么作用? | - | 0 | - | - | 1 |
| 10 | 阿司匹林和氯吡格雷联合用药的效果和风险? | ['drug_cn'] | 3 | - | - | 4 |
| 11 | SGLT2 inhibitors for heart failure in diab... | ['disease', 'anatomy'] | 2 | - | - | 3 |
| 12 | PD-L1 expression and response to immunothe... | - | 4 | - | - | 5 |
| 13 | Recent 5 years research on PD-1 immunother... | - | 2 | year >=  | - | 3 |
| 14 | Latest 3 years studies on CAR-T cell thera... | - | 0 | year >=  | - | 1 |
| 15 | Past 10 years clinical trials for Alzheime... | ['disease'] | 0 | year >=  | - | 1 |
| 16 | Clinical trials published since 2015 for r... | ['disease'] | 0 | year >=  | - | 1 |
| 17 | Landmark studies in 2019 about GLP-1 agoni... | - | 2 | year >=  | - | 3 |
| 18 | 近五年关于 PD-1 免疫疗法的研究进展 | - | 2 | year >=  | - | 3 |
| 19 | 近三年 mRNA 疫苗的研究进展 | - | 0 | year >=  | - | 1 |
| 20 | 2020年以来关于新冠后遗症的研究 | - | 3 | year >=  | - | 4 |
| 21 | What did Nature publish about CRISPR? | - | 0 | - | nature | 1 |
| 22 | Recent NEJM papers on COVID-19 vaccine eff... | - | 3 | year >=  | nejm | 4 |
| 23 | Lancet reviews on gut microbiome and metab... | - | 0 | - | lancet | 1 |
| 24 | Cell published studies on p53 mutation in ... | ['disease', 'gene_protein'] | 2 | - | cell | 3 |
| 25 | Science articles on alpha-synuclein and Pa... | ['disease'] | 0 | - | science | 1 |
| 26 | Impact of HbA1c control on cardiovascular ... | ['disease', 'anatomy'] | 3 | - | - | 4 |
| 27 | Metformin and vitamin B12 deficiency in di... | ['drug'] | 0 | - | - | 1 |
| 28 | ctDNA monitoring for EGFR mutant NSCLC tar... | ['gene_protein'] | 3 | - | - | 4 |
| 29 | Incidence and mortality trends of pancreat... | ['disease'] | 0 | - | - | 1 |
| 30 | Subgroup analysis of sex differences in st... | - | 3 | - | - | 4 |
| 31 | Hazard ratio and overall survival in immun... | - | 0 | - | - | 1 |
| 32 | TMB and PD-1/PD-L1 inhibitor response corr... | - | 4 | - | - | 5 |
| 33 | Meta-analysis of ACE inhibitors in heart f... | ['disease', 'anatomy'] | 0 | - | - | 1 |
| 34 | Recent advances in gene therapy for rare d... | - | 0 | year >=  | - | 1 |
| 35 | Cell reports on novel drug delivery system... | - | 0 | - | cell | 1 |
| 36 | ASA and bleeding risk in cardiovascular pa... | ['anatomy'] | 2 | - | - | 3 |
| 37 | 高血压患者用 ARB 治疗的效果如何? | ['disease_cn'] | 4 | - | - | 5 |
| 38 | Myocardial infarction with ST-segment elev... | ['disease'] | 0 | - | - | 1 |
| 39 | Which statins are NOT associated with incr... | ['disease'] | 0 | - | - | 1 |
| 40 | 我在查找关于二型糖尿病患者使用恩格列净（empagliflozin）对心血管结局影响... | ['drug', 'disease_cn'] | 7 | year >=  | nejm | 8 |