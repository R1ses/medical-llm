import csv, re, sys
from collections import defaultdict
from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
TEST_DIR = "data/cmmlu/test"
MED_SUBJECTS = ["anatomy", "clinical_knowledge", "college_medicine", "professional_medicine",
                "traditional_chinese_medicine", "nutrition", "virology", "college_medical_statistics"]
CHOICES = ["A", "B", "C", "D"]
SYS = "你是一个医学推理助手。请先在<think></think>中逐步分析，然后给出答案字母。"

def parse(text):
    after = text.split("</think>")[-1] if "</think>" in text else text
    m = re.search(r'(?<![a-zA-Z])([ABCD])(?![a-zA-Z])', after)
    return m.group(1) if m else None

llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85,
          max_model_len=2048, trust_remote_code=True)
tokenizer = llm.get_tokenizer()

prompts, meta = [], []
for subj in MED_SUBJECTS:
    with open(f"{TEST_DIR}/{subj}.csv", encoding="utf-8") as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            q, opts, ans = row[1], row[2:6], row[6].strip().upper()
            if ans not in CHOICES:
                continue
            opt_str = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
            messages = [{"role": "system", "content": SYS},
                        {"role": "user", "content": f"{q}\n{opt_str}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            meta.append((subj, ans))

print(f"总题数: {len(prompts)}")
params = SamplingParams(temperature=0, max_tokens=512)
outputs = llm.generate(prompts, params)

subj_c = defaultdict(int); subj_n = defaultdict(int)
tot_c = tot_n = no_parse = 0
for (subj, ans), out in zip(meta, outputs):
    pred = parse(out.outputs[0].text)
    subj_n[subj] += 1; tot_n += 1
    if pred is None:
        no_parse += 1
    elif pred == ans:
        subj_c[subj] += 1; tot_c += 1

print("\n==== 各科目 ====")
for subj in MED_SUBJECTS:
    if subj_n[subj]:
        print(f"{subj}: {subj_c[subj]}/{subj_n[subj]} = {subj_c[subj]/subj_n[subj]*100:.1f}%")
print(f"\n=== 医学总计: {tot_c}/{tot_n} = {tot_c/tot_n*100:.2f}% (解析失败 {no_parse}) ===")
print(f"模型: {MODEL}")
