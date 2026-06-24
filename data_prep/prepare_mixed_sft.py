import json, random
from datasets import load_from_disk

random.seed(42)

# 1. CMExam 15k，带 <think> COT
ds = load_from_disk('data/CMExam')
cmexam_samples = []
for item in ds:
    try:
        cot = item['extra_info']['cot'][0]['value']
        answer = item['reward_model']['ground_truth']
        question = item['extra_info']['question']
        options = item['extra_info']['options']
        if not cot or not answer or not question:
            continue
        opt_str = '\n'.join([f"{k}. {v}" for k, v in options.items()])
        cmexam_samples.append({
            "conversations": [
                {"from": "system", "value": "你是一个医学推理助手。请先在<think></think>中逐步分析，然后给出答案字母。"},
                {"from": "human", "value": f"{question}\n{opt_str}"},
                {"from": "gpt", "value": f"<think>\n{cot}\n</think>\n答案：{answer}"}
            ]
        })
    except:
        continue
random.shuffle(cmexam_samples)
cmexam_samples = cmexam_samples[:15000]
print(f"CMExam: {len(cmexam_samples)} 条")

# 2. Huatuo 10k，转成 sharegpt 格式
huatuo_samples = []
with open('data/train/med_sft/huatuo_10k.jsonl') as f:
    for line in f:
        item = json.loads(line)
        instruction = item.get('instruction', '') + item.get('input', '')
        output = item.get('output', '')
        if not instruction or not output:
            continue
        huatuo_samples.append({
            "conversations": [
                {"from": "system", "value": "你是一个专业的医学助手，请根据患者的问题给出专业、准确的医学建议。"},
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        })
print(f"Huatuo: {len(huatuo_samples)} 条")

# 3. 混合保存
import os
all_samples = cmexam_samples + huatuo_samples
random.shuffle(all_samples)
os.makedirs('data/train/mixed_sft', exist_ok=True)
out_path = 'data/train/mixed_sft/mixed_sft.jsonl'
with open(out_path, 'w') as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f"总计: {len(all_samples)} 条 → {out_path}")
