"""
🌪️ 风电智能预测解释系统 - 主入口
"""
import argparse
import pandas as pd
import torch
from pathlib import Path

from config import ModelConfig, DataConfig, LLMConfig, PROJECT_ROOT
from models import LSTMWindPowerPredictor, LSTMTrainer, LSTMEvaluator
from llm import WindPowerAgent
from rag import EnhancedRAGSystem


class WindPowerSystem:
    """风电智能预测解释系统"""
    
    def __init__(self, mode='inference'):
        """
        参数:
            mode: 'train' 或 'inference'
        """
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        self.model = None
        self.trainer = None
        self.agent = None
        
        print(f"\n{'='*60}")
        print(f"🌪️  风电智能预测解释系统")
        print(f"{'='*60}")
        print(f"模式: {mode}")
        print(f"设备: {self.device}")
        print(f"{'='*60}\n")
    
    def train_model(self, data_path=None):
        """训练LSTM模型"""
        print("\n" + "⚡"*30)
        print("     训练LSTM时序预测模型")
        print("⚡"*30 + "\n")
        
        # 1. 加载数据
        data_path = data_path or (PROJECT_ROOT / DataConfig.FEATURE_DATA_FILE)
        print(f"📂 加载数据: {data_path}")
        data = pd.read_csv(data_path)
        print(f"✅ 数据形状: {data.shape}")
        
        # 2. 准备序列数据
        trainer_instance = LSTMTrainer(None, device=self.device)
        X_seq, y_seq = trainer_instance.prepare_sequences(data)
        
        # 3. 标准化
        X_seq, y_seq = trainer_instance.normalize_data(X_seq, y_seq)
        
        # 4. 划分数据集
        train_data, val_data, test_data = trainer_instance.split_data(X_seq, y_seq)
        
        # 5. 创建DataLoader
        train_loader, val_loader, test_loader = trainer_instance.create_dataloaders(
            train_data, val_data, test_data
        )
        
        # 6. 创建模型
        input_dim = X_seq.shape[2]
        model = LSTMWindPowerPredictor(
            input_dim=input_dim,
            hidden_dim=ModelConfig.HIDDEN_DIM,
            num_layers=ModelConfig.NUM_LAYERS,
            dropout=ModelConfig.DROPOUT
        )
        
        model_info = model.get_model_info()
        print(f"\n📊 模型信息:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # 7. 训练
        trainer_instance.model = model.to(self.device)
        best_loss = trainer_instance.train(train_loader, val_loader)
        
        # 8. 评估
        trainer_instance.load_best_model()
        evaluator = LSTMEvaluator(
            trainer_instance.model, 
            device=self.device,
            y_mean=trainer_instance.y_mean,
            y_std=trainer_instance.y_std
        )
        results = evaluator.evaluate(test_loader)
        evaluator.plot_results(results)
        
        self.model = trainer_instance.model
        self.trainer = trainer_instance
        
        print(f"\n✅ 训练完成！")
        return results
    
    def _compute_norm_params(self):
        """从特征文件重新计算归一化参数（兼容旧 checkpoint）"""
        import numpy as np
        import pandas as pd
        try:
            df = pd.read_csv(PROJECT_ROOT / DataConfig.FEATURE_DATA_FILE)
            exclude = ['datetime', 'wind_power']
            feat_cols = [c for c in df.columns if c not in exclude]
            y = df['wind_power'].values.astype(np.float32)
            X = df[feat_cols].values.astype(np.float32)
            self.y_mean = float(y.mean())
            self.y_std  = float(y.std())
            self.X_mean = X.mean(axis=0)
            self.X_std  = X.std(axis=0) + 1e-8
            print(f"   ✅ 从数据文件计算归一化参数完成")
        except Exception as e:
            print(f"   ⚠️  计算归一化参数失败: {e}，使用 y_mean=0 y_std=1")
            self.y_mean = 0.0
            self.y_std  = 1.0
            self.X_mean = None
            self.X_std  = None

    def load_model(self, model_path=None):
        """加载训练好的模型"""
        model_path = model_path or ModelConfig.MODEL_SAVE_PATH
        
        print(f"\n📦 加载模型: {model_path}")
        
        if not Path(model_path).exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        # 创建模型实例（需要知道input_dim）
        # 这里假设使用86个特征
        self.model = LSTMWindPowerPredictor(
            input_dim=ModelConfig.INPUT_DIM,
            hidden_dim=ModelConfig.HIDDEN_DIM,
            num_layers=ModelConfig.NUM_LAYERS,
            dropout=ModelConfig.DROPOUT
        )
        
        # 兼容新版的 state_dict 存储格式
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # 如果 checkpoint 里有归一化参数就用，否则从数据文件重新计算（兼容旧版保存格式）
            y_mean = checkpoint.get("y_mean")
            y_std  = checkpoint.get("y_std")
            if y_mean is None or y_std is None:
                print("   ⚠️  checkpoint 未含归一化参数，将从数据文件重新计算...")
                self._compute_norm_params()
            else:
                self.y_mean = float(y_mean)
                self.y_std  = float(y_std)
                self.X_mean = checkpoint.get("X_mean", None)
                self.X_std  = checkpoint.get("X_std",  None)
            print(f"   归一化参数: y_mean={self.y_mean:.2f}, y_std={self.y_std:.2f}")
        else:
            self.model.load_state_dict(checkpoint)
            self._compute_norm_params()
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载成功")
        return True
    
    def initialize_agent(self, rag_system=None):
        """初始化LLM Agent"""
        print(f"\n🤖 初始化LLM Agent...")

        # 没有外部传入时，自动尝试创建 RAG 系统
        if rag_system is None:
            try:
                rag_system = EnhancedRAGSystem()
            except Exception as e:
                print(f"⚠️ RAG系统初始化失败，将跳过: {e}")

        # 将 system.predict 直接传入 agent.lstm_predictor 中
        self.agent = WindPowerAgent(
            rag_system=rag_system,
            use_cot=True,
            use_reflection=False,
            lstm_predictor=self.predict,
            llm_backend="lora",
            adapter_path="wind_power_adapter"
        )
        
        return self.agent.check_ready()
    
    def predict(self, input_data):
        """
        进行预测（输入→标准化→LSTM→反归一化→真实kW值）
        
        参数:
            input_data: 支持两种格式：
                - numpy array (seq_len, features): 真实序列数据
                - dict: 简单天气字典（演示模式，自动从数据文件取最新12步真实序列）
        """
        if self.model is None:
            print("❌ 模型未加载")
            return None

        import numpy as np
        import pandas as pd

        # --- 1. 构造输入序列 ---
        if isinstance(input_data, dict):
            # 演示模式：从真实特征文件取最后 SEQ_LEN 行，比补零准确得多
            try:
                feat_path = PROJECT_ROOT / DataConfig.FEATURE_DATA_FILE
                df = pd.read_csv(feat_path)
                exclude = ['datetime', 'wind_power']
                feat_cols = [c for c in df.columns if c not in exclude]
                seq = df[feat_cols].iloc[-ModelConfig.SEQ_LEN:].values.astype(np.float32)
                demo_mode = True
                print(f"   [Demo] 使用数据最后 {ModelConfig.SEQ_LEN} 步真实特征作为输入")
            except Exception as e:
                print(f"   [Demo] 无法加载特征文件({e})，使用零矩阵补全")
                seq = np.zeros((ModelConfig.SEQ_LEN, ModelConfig.INPUT_DIM), dtype=np.float32)
                seq[:, 0] = input_data.get('wind_speed', 10.0)
                seq[:, 1] = input_data.get('temperature', 20.0)
                seq[:, 2] = input_data.get('pressure', 1013.25)
                demo_mode = True
        else:
            seq = np.array(input_data, dtype=np.float32)
            demo_mode = False

        # --- 2. 特征标准化（与训练时一致）---
        if self.X_mean is not None and self.X_std is not None:
            X_mean = np.array(self.X_mean, dtype=np.float32)
            X_std  = np.array(self.X_std,  dtype=np.float32)
            X_std  = np.where(X_std == 0, 1.0, X_std)
            seq = (seq - X_mean) / X_std

        with torch.no_grad():
            input_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            raw_output = self.model(input_tensor).cpu().numpy().flatten()[0]

        # --- 3. 反归一化，还原为真实 kW 值 ---
        real_power = float(raw_output * self.y_std + self.y_mean)
        real_power = max(0.0, real_power)

        return {
            "predicted_power": round(real_power, 2),
            "unit": "kW",
            "mode": "demo (latest real data)" if demo_mode else "real"
        }
    
    def explain_prediction(self, prediction_data):
        """
        解释预测结果
        
        参数:
            prediction_data: 包含气象数据和预测结果的字典
        """
        if self.agent is None:
            print("❌ Agent未初始化")
            return None
        
        return self.agent.explain_prediction(prediction_data)
    
    def run_interactive(self):
        """交互式运行 - Agent 模式循环"""
        print("\n" + "="*60)
        print("🎮 欢迎使用 风电智能大模型 Agent (交互/决策模式)")
        print("="*60)
        print("你可以自由提问，例如:")
        print("  - 请帮我预测一下现在的风电功率。")
        print("  - 风速很高但功率异常，请解释一下可能的物理原因。")
        print("  - 请帮我生成一份针对目前气象的诊断报告。")
        print("或者输入 'exit' 退出")
        print("="*60 + "\n")
        
        # mock_data 用于演示需要传入预测模型的情况
        mock_data = {
            'wind_speed': 10.5,
            'temperature': 20.0,
            'pressure': 1013.25
        }

        while True:
            query = input("\n👤 你的请求 > ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("👋 再见！")
                break
                
            if not query:
                continue
                
            if self.agent:
                try:
                    response = self.agent.run(query=query, data=mock_data)
                    print(f"\n🤖 [Agent 最终回复]:\n{'-'*60}\n{response}\n{'-'*60}")
                except Exception as e:
                    import traceback
                    print(f"❌ Agent 运行出错: {e}")
                    traceback.print_exc()
            else:
                print("❌ Agent未初始化")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='风电智能预测解释系统')
    parser.add_argument('--mode', type=str, default='inference',
                       choices=['train', 'inference', 'interactive'],
                       help='运行模式')
    parser.add_argument('--data', type=str, default=None,
                       help='数据文件路径（训练模式）')
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径（推理模式）')
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = WindPowerSystem(mode=args.mode)
    
    if args.mode == 'train':
        # 训练模式
        system.train_model(data_path=args.data)
    
    elif args.mode == 'inference':
        # 推理模式
        if system.load_model(model_path=args.model):
            system.initialize_agent()
            print("\n✅ 系统就绪")
            print("   使用API或UI进行预测和解释")
    
    elif args.mode == 'interactive':
        # 交互模式
        if system.load_model(model_path=args.model):
            system.initialize_agent()
            system.run_interactive()


if __name__ == "__main__":
    main()
