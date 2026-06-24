
import json, os, time, re, threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY  = os.environ.get("JUDGE_API_KEY", "")
BASE_URL = "https://api.deepseek.com"   
MODEL    = "deepseek-v4-flash" 
TARGET   = 800
CONCURRENCY = 3          # 温和并发，触发限流就降到 2 或 1

CAND = "data/train/dpo/candidates_v2.jsonl"
OUT  = "data/train/dpo/preference_pairs_v2.jsonl"

write_lock = threading.Lock()
_local = threading.local()
def get_client():
    if not hasattr(_local, "c"):
        _local.c = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
    return _local.c

def has_repeat(t):
    for i in range(0, len(t) - 24, 4):
        seg = t[i:i+12]
        if t.count(seg) >= 3:
            return True
    return False

def clean(responses):
    out = []
    for r in responses:
        r = r.strip()
        if len(r) < 20 or len(r) > 600:
            continue
        if has_repeat(r):
            continue
        out.append(r)
    return out

JUDGE = """你是一位资深临床医学专家。请从以下针对同一患者问题的多个回答中，选出【最好】和【最差】的一个。

【评判优先级】冲突时严格按此顺序裁决：
1. 安全性：含危险或错误医学建议的（如建议重症不就医、错误用药、有害饮食建议），直接判为差，无论多详尽
2. 准确性：诊断方向、病因、用药、检查建议是否正确，有无事实错误
3. 针对性：是否针对该患者的具体情况给建议，而非放之四海皆准的空泛内容
4. 完整性：是否覆盖原因+建议+注意事项（但啰嗦堆砌不算完整）
5. 表达：清晰、有条理、无复读、无截断

患者问题：
{question}

候选回答：
{answers}

请输出JSON（gap为best与worst的质量差距：5=差距巨大如一个安全准确另一个有害错误，3=差距明显，1=两者质量几乎一样）：
{{"best": 编号, "worst": 编号, "gap": 1到5, "reason": "关键差异一句话"}}
编号从1开始。"""

def judge(question, responses, max_retry=4):
    answers = "\n\n".join(f"【回答{i}】\n{r}" for i, r in enumerate(responses, 1))
    prompt = JUDGE.format(question=question, answers=answers)
    client = get_client()
    for a in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=2000)
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError("no json")
            o = json.loads(m.group(0))
            import re as _re
            def _num(x):
                m = _re.search(r"\d+", str(x))
                return int(m.group()) if m else -1
            b, w, gap = _num(o["best"]), _num(o["worst"]), _num(o.get("gap", 3))
            if b == w or not (1 <= b <= len(responses) and 1 <= w <= len(responses)):
                raise ValueError("bad idx")
            return b, w, gap, o.get("reason", "")
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "limit" in msg:
                time.sleep(3 * (a + 1))
            elif a == max_retry - 1:
                return None, None, str(e)
            else:
                time.sleep(2 ** a)
    return None, None, None, "fail"

def process(c):
    q = c["question"]
    resp = clean(c["responses"])
    if len(resp) < 2:
        return ("skip", q, None)
    b, w, gap, reason = judge(q, resp)
    if b is None:
        return ("fail", q, None)
    if gap <= 2:
        return ("weak", q, None)
    return ("ok", q, {"question": q, "chosen": resp[b-1], "rejected": resp[w-1], "gap": gap, "reason": reason})

def main():
    cands = [json.loads(l) for l in open(CAND, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["question"] for l in open(OUT, encoding="utf-8") if l.strip()}
    todo = [c for c in cands if c["question"] not in done]
    print(f"候选 {len(cands)}，已完成 {len(done)}，待处理 {len(todo)}")

    ok = [len(done)]; weak = [0]; fail = [0]; stop = threading.Event()
    fout = open(OUT, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(process, c): c for c in todo}
        for fut in as_completed(futures):
            if stop.is_set():
                continue
            status, q, pair = fut.result()
            if status == "ok":
                with write_lock:
                    if ok[0] >= TARGET:
                        stop.set(); continue
                    fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    fout.flush()
                ok[0] += 1
            elif status == "weak":
                weak[0] += 1
            elif status == "fail":
                fail[0] += 1
            if (ok[0] + weak[0] + fail[0]) % 25 == 0:
                print(f"进度: {ok[0]} 强信号 / 弱信号丢弃 {weak[0]} / 失败 {fail[0]}")
            if ok[0] >= TARGET:
                stop.set()
    fout.close()
    print(f"\n完成: {ok[0]} 强信号偏好对，丢弃弱信号 {weak[0]}，失败 {fail[0]}")

if __name__ == "__main__":
    main()
