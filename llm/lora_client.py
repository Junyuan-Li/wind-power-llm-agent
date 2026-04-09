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

        self.adapter_path = Path(adapter_path or (
            PROJECT_ROOT / "finetune" / "lora_output" / "final_adapter"
        ))

        # 从元信息读取基座模型
        meta_file = self.adapter_path / "training_meta.json"
        if base_model:
            self.base_model = base_model
        elif meta_file.exists():
            with open(meta_file, encoding='utf-8') as f:
                self.base_model = json.load(f).get("base_model", FinetuneConfig.BASE_MODEL)
        else:
            self.base_model = FinetuneConfig.BASE_MODEL

        self.model     = None
        self.tokenizer = None
        self._loaded   = False

    def _load(self):
        """懒加载模型（首次调用时加载）"""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"📦 加载 LoRA 推理模型...")
        print(f"   基座: {self.base_model}")
        print(f"   适配器: {self.adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_path),   # 优先从 adapter 目录加载（训练时已保存）
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
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
