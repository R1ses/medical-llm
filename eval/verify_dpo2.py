# verify_dpo2.py
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = "data/train/dpo/dpo_train_v2.jsonl"
items = [json.loads(l) for l in open(DATA, encoding="utf-8")][:30]

def logp(tok, m, sys, q, ans):
    msgs = [{"role":"system","content":sys},{"role":"user","content":q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pi = tok(prompt, return_tensors="pt").input_ids
    fi = tok(prompt+ans, return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        logits = m(fi).logits[0,:-1]
    tgt = fi[0,1:]
    lp = torch.log_softmax(logits,-1).gather(1,tgt.unsqueeze(1)).squeeze(1)
    return lp[pi.shape[1]-1:].sum().item()

def test(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()
    win, diffs = 0, []
    for it in items:
        sys = it["conversations"][0]["value"]; q = it["conversations"][1]["value"]
        lc = logp(tok, m, sys, q, it["chosen"]["value"])
        lr = logp(tok, m, sys, q, it["rejected"]["value"])
        if lc > lr: win += 1
        diffs.append(lc - lr)
    del m; torch.cuda.empty_cache()
    return win, sum(diffs)/len(diffs)

import sys
path = sys.argv[1]
win, avg = test(path)
print(f"{path}")
print(f"chosen 胜出: {win}/{len(items)} = {win/len(items)*100:.0f}%")
print(f"平均 logp(chosen)-logp(rejected): {avg:+.3f}")
