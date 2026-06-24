# 基于 Qwen2.5-3B 的中文医疗推理大模型

复现 DeepSeek-R1 后训练范式（SFT → GRPO → DPO），训练具备 `<think>` 链式推理能力的中文医疗问答模型。

## 核心结果

CMMLU 医学子集（8 科 1639 题）生成式评测，准确率从 baseline **39.54%** 提升至 **65.04%**（+25.5pp）。

| 模型 | logit 法 | 生成式(CoT) | 生成式−logit | 解析失败 |
|------|---------|------------|-------------|---------|
| Qwen2.5-3B baseline | 45.88% | 39.54% | −6.34 | 22 |
| + Mixed SFT (LoRA) | 52.78% | 58.08% | +5.30 | 101 |
| + GRPO (全参数) | 60.16% | **65.04%** | +4.88 | 18 |
| + DPO (LoRA) | 59.85% | 64.67% | +4.82 | 14 |

### 关键发现：推理能力反转
`生成式−logit` 从 baseline 的 **−6.34** 反转为训练后的 **+5 左右**，证明模型从"有知识但 CoT 推理脱轨"转变为"推理助力答题"，是链式推理能力真实习得的证据。

### 增益分解
+25.5pp = 知识增益 +14.3pp（logit）+ 推理能力增益 +11.2pp（推理反转），二者各半，非格式刷分。

## 训练流程

- **Stage 1 — Mixed SFT (LoRA)**：Huatuo 医疗对话 10k + CMExam 医考题(含 COT) 15k，一阶段混合训练缓解灾难性遗忘
- **Stage 2 — GRPO (VeRL, 全参数)**：规则奖励（答案正确性 + `<think>` 格式完整性），规避 reward hacking
- **Stage 3 — DPO (LoRA)**：on-policy 自采样 + DeepSeek-V4 作 LLM-as-Judge 构造 800 条偏好数据，优化开放问答专业性与安全性

## 严谨性验证
严格泄漏检测（完整题干+选项精确匹配 + 字符 n-gram TF-IDF 模糊匹配）：训练-评测集精确重合 0%，近似重合(cosine≥0.8)仅 0.12%，排除数据泄漏。

## 目录结构
data_prep/   数据构造（SFT混合、GRPO、DPO采样与裁判）
configs/     训练配置（SFT/GRPO/DPO 的 yaml 与脚本）
reward/      GRPO 规则奖励函数
eval/        评测脚本（生成式/logit/偏好验证/泄漏检测

## 数据与模型
脚本默认从项目根目录的 `data/`、`output/` 读取，因体积/版权未上传，需自行准备：
- base model: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu)、CMExam、Huatuo26M-Lite

## API Key
DPO 裁判脚本通过环境变量读取：`export JUDGE_API_KEY=your_key`

## 技术栈
LLaMA-Factory · VeRL · vLLM · PyTorch · LoRA · GRPO · DPO · LLM-as-Judge



