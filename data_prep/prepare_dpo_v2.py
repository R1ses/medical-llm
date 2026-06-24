import json

SRC = "data/train/dpo/preference_pairs_v2.jsonl"
OUT = "data/train/dpo/dpo_train_v2.jsonl"
SYS = "你是一个专业的医学助手，请根据患者的问题给出专业、准确的医学建议。"

ends = ("。", "！", "？", ".", "!", "?", "；", "）", "”", '"')
def truncated(t):
    return not t.strip().endswith(ends)

items = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
seen = set()
kept = 0
filt = 0
with open(OUT, "w", encoding="utf-8") as fout:
    for d in items:
        if d["question"] in seen:
            continue
        seen.add(d["question"])
        if truncated(d["chosen"]):
            filt += 1
            continue
        item = {
            "conversations": [
                {"from": "system", "value": SYS},
                {"from": "human", "value": d["question"]}
            ],
            "chosen": {"from": "gpt", "value": d["chosen"]},
            "rejected": {"from": "gpt", "value": d["rejected"]}
        }
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        kept += 1
print(f"保留 {kept} 条，过滤截断 {filt} -> {OUT}")
