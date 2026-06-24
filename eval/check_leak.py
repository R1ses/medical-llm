import json, csv, re, os
import pandas as pd

def norm(t):
    return re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9]', '', t).strip()

# 1. 训练题干（归一化）
train = set()
with open("data/train/mixed_sft/mixed_sft.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        for t in d["conversations"]:
            if t["from"] == "human":
                train.add(norm(t["value"].split("\n")[0]))
for pf in ["data/grpo/grpo_train.parquet", "data/grpo/grpo_val.parquet"]:
    if os.path.exists(pf):
        for prompt in pd.read_parquet(pf)["prompt"]:
            for m in prompt:
                if m["role"] == "user":
                    train.add(norm(m["content"].split("\n")[0]))
print(f"训练题干(去重): {len(train)}")

# 2. 评测题
MED = ["anatomy","clinical_knowledge","college_medicine","professional_medicine",
       "traditional_chinese_medicine","nutrition","virology","college_medical_statistics"]
ev = []
for s in MED:
    with open(f"data/cmmlu/test/{s}.csv", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) >= 7: ev.append((s, row[1]))
print(f"评测题: {len(ev)}")

# 3. 完全重合
exact = [(s,q) for s,q in ev if norm(q) in train]
print(f"\n=== 完全重合: {len(exact)}/{len(ev)} = {len(exact)/len(ev)*100:.2f}% ===")
for s,q in exact[:8]: print(f"  [{s}] {q[:45]}")

# 4. 近似重合（前15字）
tp = set(t[:15] for t in train if len(t)>=15)
near = [(s,q) for s,q in ev if len(norm(q))>=15 and norm(q)[:15] in tp and norm(q) not in train]
print(f"\n=== 近似重合(前15字相同): {len(near)} ===")
for s,q in near[:8]: print(f"  [{s}] {q[:45]}")
