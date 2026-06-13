"""Quick stage6 end-to-end test"""
import sys, os, time
sys.path.insert(0, r'C:\Users\Administrator\Desktop\medical_rag_project')
os.chdir(r'C:\Users\Administrator\Desktop\medical_rag_project')

from stage6_retrieval_pipeline import RetrievalPipeline

print("Initializing pipeline...")
pipeline = RetrievalPipeline(max_files=50)

print("\nRunning test queries...")
test_queries = [
    "二甲双胍对心血管疾病的影响",
    "What is the effect of metformin on cardiovascular disease?",
    "PD-1 免疫疗法近五年研究进展",
]

for q in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: {q}")
    result = pipeline.query(q, verbose=True)
    print(f"Answer preview: {result.answer[:150]}...")
    print(f"Retrieved: {len(result.retrieved_docs)} docs in {result.retrieval_time_ms:.0f}ms")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    if result.error:
        print(f"ERROR: {result.error}")