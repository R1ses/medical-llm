import re

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "gsm8k":
        return _score_math(solution_str, ground_truth)
    else:
        return _score_medical(solution_str, ground_truth)

def _parse_response(text):
    has_think = "<think>" in text and "</think>" in text
    think_content = ""
    if has_think:
        think_content = text.split("<think>")[1].split("</think>")[0].strip()
    # 只从 </think> 之后提取答案
    after_think = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
    # 严格匹配：答案必须是独立的单个字母
    match = re.search(r'(?<![a-zA-Z])([ABCD])(?![a-zA-Z])', after_think)
    pred = match.group(1) if match else None
    return has_think, think_content, pred

def _score_medical(solution_str, ground_truth):
    gt = str(ground_truth).strip().upper()
    has_think, think_content, pred = _parse_response(solution_str)
    correct = (pred == gt)
    think_substantial = has_think and len(think_content) > 30

    # 格式奖励：有完整 <think> 且推理充实
    format_reward = 0.2 if think_substantial else (0.1 if has_think else 0.0)
    # 正确性奖励：只有答对才给，且 think 格式下加权
    if correct:
        acc_reward = 0.8 if think_substantial else 0.3
    else:
        acc_reward = 0.0

    return round(min(acc_reward + format_reward, 1.0), 4)

def _score_math(solution_str, ground_truth):
    gt = str(ground_truth).strip()
    has_think = "<think>" in solution_str and "</think>" in solution_str
    after_think = solution_str.split("</think>")[-1] if "</think>" in solution_str else solution_str
    numbers = re.findall(r'-?\d+\.?\d*', after_think)
    pred = numbers[-1] if numbers else None
    correct = (pred == gt)

    if correct and has_think:
        return 1.0
    elif correct:
        return 0.3
    elif has_think:
        return 0.1
    else:
        return 0.0

