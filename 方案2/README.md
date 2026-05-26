# 方案2：Base Model 负样本 + DPO 训练

## 文件说明

| 文件 | 用途 |
|------|------|
| `train_cot.json` | DeepSeek API 生成的高质量 CoT 训练数据（11,955条），含正确答案，用于 LoRA SFT 训练 |
| `reject.json` | Base model（Qwen2.5-0.5B-Instruct）零-shot 推理的错误 CoT（11,158条），含错误答案，用作 DPO 负样本 |
| `cot_train_basemodel.jsonl` | Base model 零-shot CoT 推理的原始输出（11,955条） |
| `train_cot_kaggle.py` | LoRA CoT 训练脚本，基于 train_cot.json 对 Qwen2.5-0.5B-Instruct 进行 SFT（Kaggle 双卡 DDP） |
| `infer_cot_kaggle.py` | LoRA 模型 CoT 推理脚本，加载训练好的 LoRA 权重进行推理 |
| `submit1.csv` | LoRA 模型在测试集上的提交结果 |
| `final/` | 训练好的 LoRA 模型权重（PEFT adapter），含 adapter_model.safetensors、tokenizer 等 |

## 数据格式

### train_cot.json / reject.json
```json
{
  "id": "0",
  "question": "食堂运来105千克的萝卜...",
  "cot": "推理过程...",
  "answer": "315"
}
```

### cot_train_basemodel.jsonl
每行一个 JSON 对象，包含 `id`、`question`、`cot` 字段。

## 流程

1. `gen_cot.py` → 调用 DeepSeek API 生成 CoT → `train_cot.json`
2. `train_cot_kaggle.py` → 用 train_cot.json LoRA SFT 训练 → `final/`
3. `infer_fewshot.py` → Base model 零-shot 推理 → `cot_train_basemodel.jsonl`
4. 后续：用 reject.json 做 DPO 训练，提升模型正确率
