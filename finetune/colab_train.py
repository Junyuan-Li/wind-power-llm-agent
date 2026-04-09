"""
Google Colab LoRA 微调脚本
================================================
使用方法（在 Colab 里逐 cell 执行）：

  1. 上传本文件 + finetune/wind_power_instruction_dataset.json 到 Colab
  2. 按顺序执行每个 CELL 注释块
  3. 训练完成后 adapter 会自动保存到 Google Drive

Cell 划分:
  CELL 1 - 安装依赖
  CELL 2 - 挂载 Google Drive
  CELL 3 - 上传数据集 (或从 Drive 读取)
  CELL 4 - 训练
  CELL 5 - 保存 & 测试
================================================
"""

# ============================================================
# CELL 1: 安装依赖
# ============================================================
# %%
# !pip install -q peft trl transformers accelerate bitsandbytes datasets

# ============================================================
# CELL 2: 挂载 Google Drive（用于保存模型）
# ============================================================
# %%
# from google.colab import drive
# drive.mount('/content/drive')
# SAVE_DIR = "/content/drive/MyDrive/wind_power_lora"

# ============================================================
# CELL 3: 上传数据集
# ============================================================
# %%
# 方式 A：直接上传 JSON 文件
# from google.colab import files
# uploaded = files.upload()   # 上传 wind_power_instruction_dataset.json
# DATASET_PATH = "wind_power_instruction_dataset.json"

# 方式 B：从 Google Drive 读取
# DATASET_PATH = "/content/drive/MyDrive/wind_power_lora/wind_power_instruction_dataset.json"

# ============================================================
# CELL 4: 训练主逻辑（直接粘贴运行）
# ============================================================
# %%

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---------- 配置 ----------
BASE_MODEL    = "Qwen/Qwen2.5-7B"   # 小模型快速验证；改成 "Qwen/Qwen2.5-7B" 做正式训练
DATASET_PATH  = "wind_power_instruction_dataset.json"
SAVE_DIR      = "/content/drive/MyDrive/wind_power_lora/adapter"
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 3
BATCH_SIZE    = 4
MAX_SEQ_LEN   = 512

# ---------- 检查 GPU ----------
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB" if torch.cuda.is_available() else "")
use_4bit = torch.cuda.is_available()

# ---------- 构建数据集 ----------
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw = json.load(f)

texts = [ALPACA_TEMPLATE.format(**item) for item in raw]
import random; random.shuffle(texts)
split = int(len(texts) * 0.9)
train_ds = Dataset.from_dict({"text": texts[:split]})
val_ds   = Dataset.from_dict({"text": texts[split:]})
print(f"数据集: 训练={len(train_ds)}, 验证={len(val_ds)}")

# ---------- 加载模型 ----------
bnb_config = None
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("⚡ 启用 QLoRA 4-bit 量化")

print(f"📦 加载模型: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
if use_4bit:
    model = prepare_model_for_kbit_training(model)

# ---------- 注入 LoRA ----------
lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    task_type=TaskType.CAUSAL_LM, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

tokenizer.model_max_length = MAX_SEQ_LEN

# ---------- 训练 ----------
sft_config = SFTConfig(
    output_dir              = SAVE_DIR,
    num_train_epochs        = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = 4,
    learning_rate           = LEARNING_RATE,
    lr_scheduler_type       = "cosine",
    warmup_steps            = 10,
    fp16                    = use_4bit is False and torch.cuda.is_available(),
    bf16                    = use_4bit and torch.cuda.is_available(),
    logging_steps           = 5,     # 改为 5，小数据集也能看到日志
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    report_to               = "none",
    dataset_text_field      = "text",
)

trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=train_ds, eval_dataset=val_ds,
    args=sft_config,
)

print("🚀 开始训练...")
result = trainer.train()
print(f"✅ 训练完成! 损失={result.training_loss:.4f}")

# ============================================================
# CELL 5: 保存 adapter + 简单推理测试
# ============================================================
# %%

import os
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# 保存元信息
with open(f"{SAVE_DIR}/training_meta.json", 'w', encoding='utf-8') as f:
    json.dump({
        "base_model": BASE_MODEL, "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA, "num_epochs": NUM_EPOCHS,
        "domain": "wind_power_prediction"
    }, f, ensure_ascii=False, indent=2)

print(f"✅ LoRA adapter 已保存到 Google Drive: {SAVE_DIR}")

# ---------- 推理测试 ----------
model.eval()
test_prompts = [
    "风速10m/s，温度20℃，气压1013hPa，请分析对风电功率的影响。",
    "风速2m/s，风机是否会启动？",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\n输入: {prompt}")
    print(f"输出: {resp.strip()[:200]}")
    print("-"*50)

# ---------- 下载 adapter（从 Colab 本地下载）----------
# from google.colab import files
# import shutil
# shutil.make_archive("/content/wind_power_adapter", "zip", SAVE_DIR)
# files.download("/content/wind_power_adapter.zip")
