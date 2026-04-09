"""
完整的LSTM训练脚本 - 使用新架构的模块化代码
"""
import pandas as pd
import torch
from pathlib import Path

from config import PROJECT_ROOT, DataConfig, ModelConfig
from models import LSTMWindPowerPredictor, LSTMTrainer


def main():
    """主函数 - 完整训练流程（使用 lstm_predictor.py 的逻辑）"""
    
    print("\n" + "⚡"*30)
    print("     LSTM时序预测模型训练")
    print("⚡"*30 + "\n")
    
    # 1. 加载数据
    print("="*80)
    print("步骤1: 加载增强特征数据")
    print("="*80)
    
    data_path = PROJECT_ROOT / DataConfig.FEATURE_DATA_FILE
    data = pd.read_csv(data_path)
    print(f"✅ 加载数据: {data.shape}")
    print(f"   特征数: {data.shape[1] - 2} (排除datetime和wind_power)")
    print(f"   样本数: {data.shape[0]}")
    
    # 2. 准备序列数据
    print("\n" + "="*80)
    print("步骤2: 准备时序序列")
    print("="*80)
    
    temp_trainer = LSTMTrainer(None)  # 临时trainer用于数据准备
    X_seq, y_seq = temp_trainer.prepare_sequences(data, seq_len=DataConfig.SEQUENCE_LENGTH)  
    
    # 3. 数据标准化
    print("\n" + "="*80)
    print("步骤3: 数据标准化")
    print("="*80)
    
    X_seq, y_seq = temp_trainer.normalize_data(X_seq, y_seq)
    
    # 保存归一化参数
    norm_params = {
        'y_mean': temp_trainer.y_mean,
        'y_std': temp_trainer.y_std,
        'X_mean': temp_trainer.X_mean,
        'X_std': temp_trainer.X_std
    }
    
    # 4. 划分数据集
    print("\n" + "="*80)
    print("步骤4: 划分训练/验证/测试集")
    print("="*80)
    
    train_data, val_data, test_data = temp_trainer.split_data(X_seq, y_seq)
    
    # 5. 创建DataLoader
    batch_size = ModelConfig.BATCH_SIZE
    train_loader, val_loader, test_loader = temp_trainer.create_dataloaders(
        train_data, val_data, test_data, batch_size=batch_size
    )
    
    print(f"\n✅ DataLoader创建完成 (batch_size={batch_size})")
    
    # 6. 创建模型
    print("\n" + "="*80)
    print("步骤5: 创建LSTM模型")
    print("="*80)
    
    input_dim = X_seq.shape[2]
    device = ModelConfig.DEVICE if torch.cuda.is_available() else 'cpu'
    
    model = LSTMWindPowerPredictor(
        input_dim=input_dim,
        hidden_dim=ModelConfig.HIDDEN_DIM,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT
    )
    
    print(f"✅ 模型创建完成")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    print(f"   设备: {device}")
    
    # 7. 训练模型
    print("\n" + "="*80)
    print("步骤6: 训练模型")
    print("="*80)
    
    trainer = LSTMTrainer(model, device=device)
    # 传递归一化参数（从 temp_trainer 复制）
    trainer.y_mean = temp_trainer.y_mean
    trainer.y_std = temp_trainer.y_std
    trainer.X_mean = temp_trainer.X_mean
    trainer.X_std = temp_trainer.X_std
    
    best_val_loss = trainer.train(
        train_loader, val_loader, 
        epochs=ModelConfig.MAX_EPOCHS, 
        lr=ModelConfig.LEARNING_RATE
    )
    
    # 8. 加载最佳模型
    print("\n" + "="*80)
    print("步骤7: 加载最佳模型")
    print("="*80)
    
    trainer.load_best_model()
    
    # 9. 评估模型
    print("\n" + "="*80)
    print("步骤8: 评估模型")
    print("="*80)
    
    results = trainer.evaluate(test_loader)
    
    # 10. 绘制结果
    print("\n" + "="*80)
    print("步骤9: 可视化结果")
    print("="*80)
    
    trainer.plot_results(results)
    
    print("\n" + "="*80)
    print("✅ LSTM模型训练完成！")
    print("="*80)
    
    print("\n📦 输出文件:")
    print(f"   - {ModelConfig.MODEL_SAVE_PATH} (最佳模型)")
    print("   - lstm_prediction_results.png (结果可视化)")
    
    print("\n💡 下一步:")
    print("   - 运行整合测试: python test_integration.py")
    print("   - 启动Web界面: streamlit run ui/streamlit_app.py")
    
    return model, trainer, results


if __name__ == "__main__":
    model, trainer, results = main()
