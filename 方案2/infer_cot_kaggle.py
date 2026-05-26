"""
本地推理脚本：加载 LoRA 权重，输出 CoT 推理过程
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import time
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 配置 ====================
MODEL_NAME = "/mnt/workspace/models/Qwen2.5-0.5B-Instruct"
TEST_PATH = "/mnt/workspace/test.json"
LORA_PATH = "/mnt/workspace/final"
OUTPUT_COT = "cot_output.jsonl"
BATCH_SIZE = 128
MAX_NEW_TOKENS = 1024
SYSTEM_PROMPT = "你是一个小学数学解题助手。请一步一步推理分析题目，最后用【答案】=XXX的格式给出最终答案。"


def get_dtype():
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    n_gpus = torch.cuda.device_count()
    print(f"使用 GPU 数: {n_gpus}")
    t_start = time.time()

    print(f"加载测试数据: {TEST_PATH}")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:4000]
    print(f"测试数据: {len(test_data)} 条（只推理前4000）")

    device = "cuda:0"
    dtype = get_dtype()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[GPU 0] {gpu_name}, dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 左填充，避免 PAD token 污染生成起点
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).cuda(0)
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()

    with open(OUTPUT_COT, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="推理"):
            batch = test_data[i:i + BATCH_SIZE]
            texts = [tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                tokenize=False, add_generation_prompt=True,
            ) for row in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.6, top_p=0.9,
                    no_repeat_ngram_size=3, repetition_penalty=1.1,
                )
            input_len = inputs.input_ids.shape[1]
            for j, row in enumerate(batch):
                response = tokenizer.decode(generated_ids[j][input_len:], skip_special_tokens=True)
                fout.write(json.dumps({
                    "id": row["id"],
                    "question": row["question"] if isinstance(row["question"], str) else "".join(row["question"]),
                    "cot": response,
                }, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"完成，已保存到: {OUTPUT_COT}")
    print(f"总耗时: {time.time() - t_start:.1f}s")


main()
