"""
LoRA / QLoRA 微调训练器
基座模型: Qwen2.5-7B (可换成任意 causal LM)
方法: LoRA (低显存时自动启用 QLoRA 4-bit 量化)

运行方式:
    python finetune/lora_trainer.py              # 完整训练
    python finetune/lora_trainer.py --dry_run    # 验证流程（不保存模型）
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# ---- 懒加载：防止没安装时 import 报错 ----
def _check_deps():
    missing = []
    for pkg in ["transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ 缺少依赖包: {', '.join(missing)}")
        print(f"   安装方法: pip install {' '.join(missing)}")
        return False
    return True


class LoRATrainer:
    """
    LoRA / QLoRA 风电领域微调训练器

    使用方法:
        trainer = LoRATrainer()
        trainer.setup()
        trainer.train()
        trainer.save()
    """

    def __init__(
        self,
        base_model: str            = None,
        output_dir: str            = None,
        use_4bit: bool             = True,   # QLoRA
        lora_r: int                = None,
        lora_alpha: int            = None,
        lora_dropout: float        = None,
        learning_rate: float       = None,
        num_epochs: int            = None,
        batch_size: int            = None,
        max_seq_length: int        = 512,
        gradient_accumulation: int = 4,
    ):
        from config import FinetuneConfig, PROJECT_ROOT

        self.base_model       = base_model     or FinetuneConfig.BASE_MODEL
        self.output_dir       = Path(output_dir or (PROJECT_ROOT / "finetune" / "lora_output"))
        self.use_4bit         = use_4bit
        self.lora_r           = lora_r         or FinetuneConfig.LORA_R
        self.lora_alpha       = lora_alpha      or FinetuneConfig.LORA_ALPHA
        self.lora_dropout     = lora_dropout    or FinetuneConfig.LORA_DROPOUT
        self.learning_rate    = learning_rate   or FinetuneConfig.LEARNING_RATE
        self.num_epochs       = num_epochs      or FinetuneConfig.NUM_EPOCHS
        self.batch_size       = batch_size      or FinetuneConfig.BATCH_SIZE
        self.max_seq_length   = max_seq_length
        self.grad_accum       = gradient_accumulation

        self.model     = None
        self.tokenizer = None
        self.trainer   = None

    # ------------------------------------------------------------------
    # 1. 加载分词器 + 模型
    # ------------------------------------------------------------------
    def setup(self):
        """加载基座模型并注入 LoRA 适配器"""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

        # CPU 环境自动禁用 4-bit 量化（bitsandbytes 不支持 CPU）
        if not torch.cuda.is_available() and self.use_4bit:
            print("⚠️  未检测到 CUDA GPU，自动禁用 QLoRA 4-bit 量化（改用 fp32 CPU 模式）")
            self.use_4bit = False

        print(f"\n{'='*60}")
        print(f"🔧 LoRA 微调配置")
        print(f"{'='*60}")
        print(f"   基座模型   : {self.base_model}")
        print(f"   量化方式   : {'QLoRA 4-bit' if self.use_4bit else 'LoRA fp16'}")
        print(f"   LoRA rank  : {self.lora_r}")
        print(f"   LoRA alpha : {self.lora_alpha}")
        print(f"   学习率     : {self.learning_rate}")
        print(f"   训练轮数   : {self.num_epochs}")
        print(f"   Batch size : {self.batch_size} × 梯度累积 {self.grad_accum}")
        print(f"   输出目录   : {self.output_dir}")

        # ---- 量化配置 (QLoRA) ----
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit               = True,
                bnb_4bit_quant_type        = "nf4",
                bnb_4bit_use_double_quant  = True,
                bnb_4bit_compute_dtype     = torch.bfloat16,
            )
            print("\n⚡ 启用 QLoRA 4-bit 量化（显存需求约 ~6GB）")

        # ---- 加载分词器 ----
        print(f"\n📦 加载分词器: {self.base_model} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- 加载基座模型 ----
        print(f"📦 加载模型: {self.base_model} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config   = bnb_config,
            device_map            = "auto",
            trust_remote_code     = True,
        )

        # ---- 为量化训练做准备 ----
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # ---- 注入 LoRA ----
        lora_config = LoraConfig(
            r              = self.lora_r,
            lora_alpha     = self.lora_alpha,
            lora_dropout   = self.lora_dropout,
            task_type      = TaskType.CAUSAL_LM,
            bias           = "none",
            # 针对主流模型的 target_modules（自动兼容 Qwen / Llama）
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)

        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"\n✅ LoRA 注入完成")
        print(f"   可训练参数: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    # ------------------------------------------------------------------
    # 2. 构建数据集
    # ------------------------------------------------------------------
    def _build_dataset(self):
        from finetune.dataset_builder import WindPowerDatasetBuilder
        builder = WindPowerDatasetBuilder()
        train_ds, val_ds = builder.build_train_val()
        return train_ds, val_ds

    # ------------------------------------------------------------------
    # 3. 训练
    # ------------------------------------------------------------------
    def train(self, dry_run: bool = False):
        """
        启动 LoRA 训练

        参数:
            dry_run: True 时只跑1步验证流程，不保存模型
        """
        from trl import SFTTrainer, SFTConfig

        if self.model is None:
            self.setup()

        print(f"\n{'='*60}")
        print(f"🚀 开始训练{'（dry_run 模式）' if dry_run else ''}")
        print(f"{'='*60}")

        train_ds, val_ds = self._build_dataset()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        import torch
        use_gpu = torch.cuda.is_available()

        self.tokenizer.model_max_length = self.max_seq_length

        sft_config = SFTConfig(
            output_dir              = str(self.output_dir),
            num_train_epochs        = 1 if dry_run else self.num_epochs,
            max_steps               = 1 if dry_run else -1,
            per_device_train_batch_size = self.batch_size,
            gradient_accumulation_steps = self.grad_accum,
            learning_rate           = self.learning_rate,
            lr_scheduler_type       = "cosine",
            warmup_steps            = 10,
            fp16                    = use_gpu and not self.use_4bit,
            bf16                    = use_gpu and self.use_4bit,
            logging_steps           = 10,
            eval_strategy           = "epoch" if not dry_run else "no",
            save_strategy           = "epoch" if not dry_run else "no",
            load_best_model_at_end  = not dry_run,
            report_to               = "none",
            dataset_text_field      = "text",
        )

        self.trainer = SFTTrainer(
            model              = self.model,
            processing_class   = self.tokenizer,
            train_dataset      = train_ds,
            eval_dataset       = val_ds,
            args               = sft_config,
        )

        result = self.trainer.train()
        print(f"\n✅ 训练完成！")
        print(f"   总步数   : {result.global_step}")
        print(f"   训练损失 : {result.training_loss:.4f}")
        return result

    # ------------------------------------------------------------------
    # 4. 保存 LoRA adapter
    # ------------------------------------------------------------------
    def save(self, path: str = None):
        """
        保存 LoRA adapter 权重

        保存后目录结构:
            lora_output/
                adapter_model.bin      # LoRA 权重
                adapter_config.json    # LoRA 配置
                tokenizer.json         # 分词器

        合并使用方法:
            from peft import PeftModel
            model = AutoModelForCausalLM.from_pretrained("base_model")
            model = PeftModel.from_pretrained(model, "lora_output/")
        """
        save_dir = Path(path) if path else self.output_dir / "final_adapter"
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        # 保存训练元信息
        meta = {
            "base_model":    self.base_model,
            "lora_r":        self.lora_r,
            "lora_alpha":    self.lora_alpha,
            "lora_dropout":  self.lora_dropout,
            "num_epochs":    self.num_epochs,
            "learning_rate": self.learning_rate,
            "domain":        "wind_power_prediction",
        }
        with open(save_dir / "training_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n✅ LoRA adapter 已保存: {save_dir}")
        print(f"   adapter_model.bin      : LoRA 权重")
        print(f"   adapter_config.json    : LoRA 配置")
        print(f"   training_meta.json     : 训练元信息")
        print(f"\n💡 加载方法:")
        print(f"   from peft import PeftModel")
        print(f"   model = PeftModel.from_pretrained(base_model, '{save_dir}')")

    # ------------------------------------------------------------------
    # 5. 推理测试（加载 adapter 后验证效果）
    # ------------------------------------------------------------------
    def inference_test(self, test_inputs: list = None):
        """
        用训练后的模型做推理测试

        参数:
            test_inputs: 测试输入列表，None 时使用默认样本
        """
        import torch

        if test_inputs is None:
            test_inputs = [
                "风速10m/s，温度20℃，气压1013hPa，请分析对风电功率的影响。",
                "风速2m/s，风机是否会启动？",
                "为什么冬季风电效率比夏季高？",
            ]

        self.model.eval()
        print(f"\n{'='*60}")
        print("🧪 推理测试")
        print('='*60)

        for prompt in test_inputs:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 200,
                    temperature    = 0.7,
                    do_sample      = True,
                    pad_token_id   = self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            print(f"\n📝 输入: {prompt}")
            print(f"🤖 输出: {response.strip()[:300]}")
            print("-" * 40)


# ==============================================================
# 命令行入口
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="风电领域 LoRA 微调")
    parser.add_argument("--base_model",   type=str,   default=None,  help="基座模型 HuggingFace ID")
    parser.add_argument("--output_dir",   type=str,   default=None,  help="输出目录")
    parser.add_argument("--lora_r",       type=int,   default=None,  help="LoRA rank")
    parser.add_argument("--epochs",       type=int,   default=None,  help="训练轮数")
    parser.add_argument("--batch_size",   type=int,   default=None,  help="Batch size")
    parser.add_argument("--no_4bit",      action="store_true",        help="禁用 QLoRA 4-bit 量化")
    parser.add_argument("--dry_run",      action="store_true",        help="只验证流程，不完整训练")
    parser.add_argument("--inference_only", action="store_true",      help="只做推理测试")
    args = parser.parse_args()

    if not _check_deps():
        return

    trainer = LoRATrainer(
        base_model  = args.base_model,
        output_dir  = args.output_dir,
        use_4bit    = not args.no_4bit,
        lora_r      = args.lora_r,
        num_epochs  = args.epochs,
        batch_size  = args.batch_size,
    )

    if args.inference_only:
        trainer.setup()
        trainer.inference_test()
        return

    trainer.setup()
    trainer.train(dry_run=args.dry_run)

    if not args.dry_run:
        trainer.save()
        trainer.inference_test()

    print("\n" + "🎉"*20)
    print("  LoRA 微调完成！")
    print("🎉"*20)


if __name__ == "__main__":
    main()
