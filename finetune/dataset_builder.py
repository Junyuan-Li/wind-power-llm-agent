"""
微调数据集构建器
将 wind_power_instruction_dataset.json 转换为 HuggingFace Dataset 格式
支持三种数据类型: explanation / causality / anomaly
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROJECT_ROOT, FinetuneConfig


class WindPowerDatasetBuilder:
    """
    风电微调数据集构建器

    支持输入格式（JSON条目）:
        {
          "instruction": "请分析...",
          "input":       "风速10m/s, 温度20℃...",
          "output":      "风速较高，功率显著增加..."
        }

    输出格式（HuggingFace Dataset）:
        {"text": "<完整对话文本>"}   # 用于 Causal LM 训练
    """

    # Alpaca风格Prompt模板
    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    # 无input时的模板
    PROMPT_TEMPLATE_NO_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else FinetuneConfig.INSTRUCTION_DATASET
        self.raw_data: List[Dict] = []

    # ------------------------------------------------------------------
    # 加载原始 JSON
    # ------------------------------------------------------------------
    def load(self) -> "WindPowerDatasetBuilder":
        """加载指令数据集 JSON"""
        print(f"\n📂 加载数据集: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        # 统计各类型
        by_type: Dict[str, int] = {}
        for item in self.raw_data:
            t = item.get('type', 'unknown')
            by_type[t] = by_type.get(t, 0) + 1

        print(f"✅ 共 {len(self.raw_data)} 条样本")
        for t, cnt in by_type.items():
            print(f"   {t}: {cnt} 条")
        return self

    # ------------------------------------------------------------------
    # 格式化单条样本为训练文本
    # ------------------------------------------------------------------
    def _format(self, item: Dict) -> str:
        """将一条指令样本格式化为完整的训练文本"""
        instruction = item.get('instruction', '').strip()
        inp         = item.get('input', '').strip()
        output      = item.get('output', '').strip()

        if inp:
            return self.PROMPT_TEMPLATE.format(
                instruction=instruction, input=inp, output=output
            )
        else:
            return self.PROMPT_TEMPLATE_NO_INPUT.format(
                instruction=instruction, output=output
            )

    # ------------------------------------------------------------------
    # 构建 HuggingFace Dataset
    # ------------------------------------------------------------------
    def build(
        self,
        types: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Dataset:
        """
        构建训练数据集

        参数:
            types:       只保留指定类型, None 表示全部
            max_samples: 最多使用的样本数
            shuffle:     是否打乱
            seed:        随机种子
        """
        if not self.raw_data:
            self.load()

        data = self.raw_data
        if types:
            data = [d for d in data if d.get('type') in types]

        if shuffle:
            random.seed(seed)
            random.shuffle(data)

        if max_samples:
            data = data[:max_samples]

        texts = [self._format(item) for item in data]
        dataset = Dataset.from_dict({"text": texts})

        print(f"\n✅ 构建数据集: {len(dataset)} 条训练样本")
        return dataset

    # ------------------------------------------------------------------
    # 拆分训练/验证集
    # ------------------------------------------------------------------
    def build_train_val(
        self,
        val_ratio: float = 0.1,
        **kwargs,
    ):
        """
        构建训练集+验证集

        参数:
            val_ratio: 验证集比例 (默认 10%)
            **kwargs:  传递给 build()

        返回:
            (train_dataset, val_dataset)
        """
        full = self.build(**kwargs)
        split = full.train_test_split(test_size=val_ratio, seed=42)
        print(f"   训练集: {len(split['train'])} 条")
        print(f"   验证集: {len(split['test'])} 条")
        return split['train'], split['test']

    # ------------------------------------------------------------------
    # 预览
    # ------------------------------------------------------------------
    def preview(self, n: int = 2):
        """打印前 n 条格式化样本"""
        if not self.raw_data:
            self.load()
        print(f"\n{'='*60}")
        print("📋 样本预览")
        print('='*60)
        for i, item in enumerate(self.raw_data[:n]):
            print(f"\n--- 样本 {i+1} (type={item.get('type')}) ---")
            print(self._format(item)[:400])
            print("...")
        print('='*60)


if __name__ == "__main__":
    builder = WindPowerDatasetBuilder()
    builder.load()
    builder.preview(n=2)
    train_ds, val_ds = builder.build_train_val()
    print(f"\n首条训练样本:\n{train_ds[0]['text'][:300]}...")
