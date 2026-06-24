
import json, random, os
import pandas as pd
from datasets import load_from_disk

random.seed(42)

CMEXAM_PATH = "data/CMExam"
SFT_JSONL   = "data/train/mixed_sft/mixed_sft.jsonl"
OUT_DIR     = "data/grpo"
N_TRAIN, N_VAL = 5000, 500

SYSTEM = "你是一个医学推理助手。请先在<think></think>中逐步分析，然后给出答案字母。"

# 1. 读 SFT 用过的 question，构建排除集（防数据重叠/作弊）
used = set()
with open(SFT_JSONL, encoding="utf-8") as f:
    for line in f:
        conv = json.loads(line)["conversations"]
        for turn in conv:
            if turn["from"] == "human":
                used.add(turn["value"].split("\n")[0].strip())  # 用题干首行作 key
print(f"SFT 已用问题数（去重）: {len(used)}")

# 2. 从 CMExam 取未用过的样本，构造 verl 5 列格式
ds = load_from_disk(CMEXAM_PATH)
samples = []
for item in ds:
    try:
        question = item["extra_info"]["question"]
        options  = item["extra_info"]["options"]
        answer   = str(item["reward_model"]["ground_truth"]).strip().upper()
        if not question or answer not in ["A", "B", "C", "D"]:
            continue
        if question.split("\n")[0].strip() in used:   # 排除 SFT 用过的
            continue
        opt_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        samples.append({
            "data_source": "cmexam",
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": f"{question}\n{opt_str}"},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "train", "index": len(samples)},
        })
    except Exception:
        continue

print(f"可用样本（已排除 SFT 重叠）: {len(samples)}")
random.shuffle(samples)

train = samples[:N_TRAIN]
val   = samples[N_TRAIN:N_TRAIN + N_VAL]
for i, s in enumerate(val):
    s["extra_info"]["split"] = "val"
    s["extra_info"]["index"] = i

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(train).to_parquet(f"{OUT_DIR}/grpo_train.parquet")
pd.DataFrame(val).to_parquet(f"{OUT_DIR}/grpo_val.parquet")
print(f"train: {len(train)} → {OUT_DIR}/grpo_train.parquet")
print(f"val:   {len(val)} → {OUT_DIR}/grpo_val.parquet")
