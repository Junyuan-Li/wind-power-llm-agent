"""
LoRA 本地推理客户端
训练完成后，用 LoRA adapter 替代 Ollama 进行推理

使用方法:
    from llm.lora_client import LoRAClient
    client = LoRAClient(adapter_path="finetune/lora_output/final_adapter")
    response = client.generate("风速10m/s，请分析发电功率")
"""
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class LoRAClient:
    """
    LoRA adapter 推理客户端
    接口与 OllamaClient 完全一致，可直接替换
    """

    def __init__(self, adapter_path: str = None, base_model: str = None):
        """
        参数:
            adapter_path: LoRA adapter 目录（含 adapter_config.json）
            base_model:   基座模型名，None 时从 adapter_path/training_meta.json 读取
        """
        from config import PROJECT_ROOT, FinetuneConfig

        # ✅ 优先级：显式传入 > finetune/lora_output/final_adapter > 0.25wind_power_adapter（备选）
        adapter_candidates = [
            adapter_path,
            PROJECT_ROOT / "finetune" / "lora_output" / "final_adapter",
            PROJECT_ROOT / "0.25wind_power_adapter",
            PROJECT_ROOT / "wind_power_adapter",
        ]
        
        self.adapter_path = None
        for candidate in adapter_candidates:
            if candidate:
                candidate_path = Path(candidate)
                if (candidate_path / "adapter_config.json").exists():
                    self.adapter_path = candidate_path
                    print(f"   ✅ 找到 LoRA Adapter: {candidate_path}")
                    break
        
        if self.adapter_path is None:
            print(f"❌ 未找到任何有效的 LoRA Adapter！")
            print(f"   已尝试路径：")
            for candidate in adapter_candidates[1:]:
                if candidate:
                    print(f"      - {candidate}")
            raise FileNotFoundError(f"No valid LoRA adapter found in: {adapter_candidates[1:]}")
        
        self.adapter_path = self.adapter_path.resolve()  # 转为绝对路径

        # 从元信息读取基座模型
        meta_file = self.adapter_path / "training_meta.json"
        if base_model:
            self.base_model = base_model
        elif meta_file.exists():
            with open(meta_file, encoding='utf-8') as f:
                meta_info = json.load(f)
                self.base_model = meta_info.get("base_model", FinetuneConfig.BASE_MODEL)
                print(f"   📋 从元信息读取基座模型: {self.base_model}")
        else:
            self.base_model = FinetuneConfig.BASE_MODEL
            print(f"   ⚠️  使用默认基座模型: {self.base_model}")

        self.model     = None
        self.tokenizer = None
        self._loaded   = False

    def _load(self):
        """加载模型（第一次调用 generate 时自动加载）"""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"📦 加载 LoRA 推理模型...")
        print(f"   基座: {self.base_model}")
        print(f"   适配器: {self.adapter_path}")

        # ✅ tokenizer 必须来自 base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else "cpu"

        # ✅ base model 路径处理
        base_model_path = self.base_model
        if not Path(base_model_path).exists():
            local_path = Path(f"D:/models/{Path(base_model_path).name}")
            if local_path.exists():
                print(f"   ✅ 使用本地模型: {local_path}")
                base_model_path = str(local_path)

        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # ✅ adapter 才用本地路径 - 必须用本地路径，不能被 transformers 当作 repo_id
        adapter_str = str(self.adapter_path)  # 已经是 resolve() 后的绝对路径
        print(f"   加载 adapter 权重: {adapter_str}")
        
        # ✅ 验证 adapter 文件完整性
        adapter_config = self.adapter_path / "adapter_config.json"
        adapter_model = self.adapter_path / "adapter_model.safetensors"
        if not adapter_config.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_str}")
        if not adapter_model.exists():
            raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_str}")
        
        # ✅ local_files_only=True 强制从本地路径加载，不从 HuggingFace Hub 尝试
        self.model = PeftModel.from_pretrained(
            base,
            adapter_str,
            is_trainable=False,
            # 关键：这两个参数确保 from_pretrained 识别为本地路径而不是 repo_id
        )

        self.model.eval()
        self._loaded = True
        print("✅ LoRA 推理模型已加载")

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, max_tokens: int = 512,
                 stream: bool = False) -> str:
        """
        生成文本（与 OllamaClient.generate 接口一致）

        参数:
            prompt:        用户输入
            system_prompt: 系统提示（会拼在 prompt 前）
            temperature:   采样温度
            max_tokens:    最大生成 token
            stream:        忽略，保持接口兼容

        返回:
            生成的文本字符串
        """
        import torch

        self._load()

        # 【重点修复】：必须使用和训练时完全一致的 Alpaca 模板
        # 否则模型不知道你在问他问题，会以为是文本续写（从而变成做选择题）
        instruction = system_prompt if system_prompt else "You are a professional wind power forecasting assistant."
        full_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{prompt}\n\n### Response:\n"
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens     = max_tokens,
            do_sample          = do_sample,
            pad_token_id       = self.tokenizer.eos_token_id,
            repetition_penalty = 1.1,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature   # 只在采样时传入，避免 warning

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def check_connection(self) -> bool:
        """检查 adapter 是否存在（对应 OllamaClient.check_connection）"""
        return (self.adapter_path / "adapter_config.json").exists()

    def check_availability(self) -> bool:
        """检查 LoRA 本地服务是否就绪（与 OllamaClient.check_availability 对应）"""
        if self.check_connection():
            print(f"✅ LoRA 本地环境就绪 (基座: {self.base_model}, Adapter: {self.adapter_path})")
            return True
        else:
            print(f"❌ 未找到 LoRA Adapter 配置 (预期路径: {self.adapter_path})")
            return False
