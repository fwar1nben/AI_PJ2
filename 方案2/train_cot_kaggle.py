"""
Kaggle 训练脚本：用 CoT 数据 LoRA 微调 Qwen2.5-0.5B（双卡 DDP）
从 HuggingFace 自动下载预训练权重。

用法:
  torchrun --nproc_per_node=2 train_cot_kaggle.py

数据路径: /kaggle/input/datasets/jackhhh123/math-word-problems/train_cot.json
输出:     ./output/Qwen_COT/  (LoRA adapter 权重)
"""

import json
import os
import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ==================== 配置 ====================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "/kaggle/input/datasets/jackhhh123/math-word-problems/train_cot.json"
OUTPUT_DIR = "./output/Qwen_COT"
MAX_LENGTH = 768        # P99=596, 768 只截断 5/11955 条 (0.04%)
BATCH_SIZE = 4         # 每张卡 batch size
GRAD_ACCUM = 4         # 梯度累积 → 有效 batch = 2卡×4×4 = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
SAVE_STEPS = 1000


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main_process():
        print(msg)


# ==================== 加载模型和分词器 ====================
log(f"下载 {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
# DDP 模式下不要设 device_map，Trainer 会自动分配
# 梯度检查点由 TrainingArguments 的 gradient_checkpointing=True 自动启用


# ==================== 数据处理 ====================
def join_question(q):
    if isinstance(q, list):
        return "".join(q)
    return q


def process_func(example):
    q = join_question(example["question"])
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一个小学数学解题助手。请一步一步推理分析题目，最后用【答案】=XXX的格式给出最终答案。"},
            {"role": "user", "content": q},
            {"role": "assistant", "content": example["cot"]},
        ],
        tokenize=False,
    )
    # 找到 assistant 标记，prompt 部分 label 为 -100，只对 CoT 计算 loss
    marker = "<|im_start|>assistant\n"
    split_pos = full_text.rfind(marker) + len(marker)
    prompt_text = full_text[:split_pos]
    answer_text = full_text[split_pos:]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids + [tokenizer.pad_token_id]
    labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.pad_token_id]
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


log(f"加载数据: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_data = [d for d in train_data if d.get("cot_ok")]
log(f"CoT 正确数据: {len(train_data)} 条")

log("预处理数据...")
train_dataset = [process_func(d) for d in train_data]

# ==================== LoRA 配置 ====================
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, config)
if is_main_process():
    model.print_trainable_parameters()

# ==================== 训练 ====================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=10,
    num_train_epochs=EPOCHS,
    save_steps=SAVE_STEPS,
    learning_rate=LEARNING_RATE,
    save_on_each_node=True,
    gradient_checkpointing=True,
    bf16=True,
    report_to="none",
    save_total_limit=3,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

log("开始训练...")
trainer.train()

# 只在主进程保存最终模型
if is_main_process():
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log(f"模型已保存到: {final_dir}")
