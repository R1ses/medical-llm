import csv, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

MODEL = sys.argv[1]
TEST_DIR = "data/cmmlu/test"
MED_SUBJECTS = ["anatomy", "clinical_knowledge", "college_medicine", "professional_medicine",
                "traditional_chinese_medicine", "nutrition", "virology", "college_medical_statistics"]
CHOICES = ["A", "B", "C", "D"]
SYS = "你是一个专业的医学助手。"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda", trust_remote_code=True).eval()

choice_ids = [tok.encode(c, add_special_tokens=False)[0] for c in CHOICES]

subj_c = defaultdict(int)
subj_n = defaultdict(int)
tot_c = 0
tot_n = 0
for subj in MED_SUBJECTS:
    with open(f"{TEST_DIR}/{subj}.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            q, opts, ans = row[1], row[2:6], row[6].strip().upper()
            if ans not in CHOICES:
                continue
            opt_str = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
            messages = [{"role": "system", "content": SYS},
                        {"role": "user", "content": f"{q}\n{opt_str}"}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1]
            scores = torch.tensor([logits[i].item() for i in choice_ids])
            pred = CHOICES[scores.argmax().item()]
            subj_n[subj] += 1
            tot_n += 1
            if pred == ans:
                subj_c[subj] += 1
                tot_c += 1
    print(f"{subj}: {subj_c[subj]}/{subj_n[subj]} = {subj_c[subj]/subj_n[subj]*100:.1f}%")

print(f"\n=== 医学 logit 总计: {tot_c}/{tot_n} = {tot_c/tot_n*100:.2f}% ===")
print(f"模型: {MODEL}")
