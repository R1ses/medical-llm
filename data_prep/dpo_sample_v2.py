
import json
from vllm import LLM, SamplingParams

MODEL = "output/grpo_mixed_merged"
HUATUO = "data/train/med_sft/huatuo_10k.jsonl"
OUT = "data/train/dpo/candidates_v2.jsonl"
N_QUESTIONS = 1500   # 取1500题，裁判后过滤还能剩800+
N_SAMPLES = 8
SYS = "你是一个专业的医学助手，请根据患者的问题给出专业、准确的医学建议。"

questions = []
with open(HUATUO, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
            q = d.get("instruction", "").strip()
            if q and len(q) > 5:
                questions.append(q)
        except: continue
        if len(questions) >= N_QUESTIONS: break
print(f"读取问题: {len(questions)}")

llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85, max_model_len=2048, trust_remote_code=True)
tokenizer = llm.get_tokenizer()

prompts = []
for q in questions:
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

params = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=0.95, max_tokens=768)
outputs = llm.generate(prompts, params)

with open(OUT, "w", encoding="utf-8") as f:
    for q, out in zip(questions, outputs):
        responses = [o.text.strip() for o in out.outputs]
        f.write(json.dumps({"question": q, "responses": responses}, ensure_ascii=False) + "\n")
print(f"已保存 {len(questions)} 题 × {N_SAMPLES} 采样 → {OUT}")
