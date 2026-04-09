"""
LSTM模型训练器 - 完整版（从 lstm_predictor.py 迁移）
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig, DataConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class WindPowerDataset(Dataset):
    """风电时序数据集"""
    
    def __init__(self, X, y):
        """
        参数:
            X: 输入序列 (n_samples, seq_len, n_features)
            y: 目标值 (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMTrainer:
    """LSTM模型训练器（完整版）"""
    
    def __init__(self, model=None, device='cpu'):
        """
        参数:
            model: LSTM模型实例
            device: 训练设备
        """
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 标准化参数（训练时计算）
        self.y_mean = None
        self.y_std = None
        self.X_mean = None
        self.X_std = None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
    def prepare_sequences(self, data, seq_len=None, target_col=None):
        """
        准备时序数据
        
        参数:
            data: DataFrame
            seq_len: 序列长度（时间步）
            target_col: 目标列名
            
        返回:
            X, y: 序列数据
        """
        seq_len = seq_len or DataConfig.SEQUENCE_LENGTH
        target_col = target_col or DataConfig.TARGET_COL
        
        print(f"\n📊 准备时序数据（序列长度={seq_len}）...")
        
        # 选择所有特征（排除datetime和目标变量）
        exclude_cols = [DataConfig.DATETIME_COL, target_col]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        print(f"   使用特征数: {len(feature_cols)}")
        print(f"   特征列表（前10个）: {feature_cols[:10]}")
        
        X_data = data[feature_cols].values.astype(np.float32)
        y_data = data[target_col].values.astype(np.float32)
        
        # 创建序列
        X_seq, y_seq = [], []
        
        for i in range(len(data) - seq_len):
            X_seq.append(X_data[i:i+seq_len])
            y_seq.append(y_data[i+seq_len])
        
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)
        
        print(f"✅ 生成 {len(X_seq)} 个序列")
        print(f"   X形状: {X_seq.shape}")
        print(f"   y形状: {y_seq.shape}")
        
        return X_seq, y_seq
    
    def normalize_data(self, X_seq, y_seq):
        """数据标准化"""
        print("\n📊 标准化数据...")
        
        # 标准化目标变量
        self.y_mean = float(y_seq.mean())
        self.y_std = float(y_seq.std())
        y_seq_scaled = (y_seq - self.y_mean) / self.y_std
        
        print(f"✅ 目标变量标准化")
        print(f"   原始范围: [{y_seq.min():.2f}, {y_seq.max():.2f}]")
        print(f"   标准化后: [{y_seq_scaled.min():.2f}, {y_seq_scaled.max():.2f}]")
        
        # 标准化特征
        n_samples, seq_len, n_features = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, n_features)
        self.X_mean = X_seq_reshaped.mean(axis=0)
        self.X_std = X_seq_reshaped.std(axis=0) + 1e-8
        X_seq_scaled = (X_seq_reshaped - self.X_mean) / self.X_std
        X_seq_scaled = X_seq_scaled.reshape(n_samples, seq_len, n_features)
        
        print(f"✅ 特征标准化完成")
        
        return X_seq_scaled, y_seq_scaled
    
    def split_data(self, X, y):
        """划分数据集"""
        # 先划分训练+验证 vs 测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=DataConfig.TEST_RATIO, random_state=42
        )
        
        # 再划分训练 vs 验证
        val_ratio_adjusted = DataConfig.VAL_RATIO / (1 - DataConfig.TEST_RATIO)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42
        )
        
        print(f"\n✅ 数据集划分:")
        print(f"   训练集: {X_train.shape}")
        print(f"   验证集: {X_val.shape}")
        print(f"   测试集: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_dataloaders(self, train_data, val_data, test_data, batch_size=128):
        """创建DataLoader"""
        batch_size = batch_size
        
        train_dataset = WindPowerDataset(*train_data)
        val_dataset = WindPowerDataset(*val_data)
        test_dataset = WindPowerDataset(*test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """训练模型"""
        print(f"\n🚀 开始训练LSTM模型...")
        print(f"   设备: {self.device}")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   验证批次数: {len(val_loader)}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # 保存完整的checkpoint（包含归一化参数）
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'input_dim': self.model.input_dim,
                    'y_mean': self.y_mean,
                    'y_std': self.y_std,
                    'X_mean': self.X_mean,
                    'X_std': self.X_std,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch + 1
                }
                torch.save(checkpoint, ModelConfig.MODEL_SAVE_PATH)
        
        print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")
        print(f"   模型已保存: {ModelConfig.MODEL_SAVE_PATH}")
        
        return best_val_loss
    
    def load_best_model(self):
        """加载最佳模型（包含归一化参数）"""
        if self.model is not None and ModelConfig.MODEL_SAVE_PATH.exists():
            checkpoint = torch.load(ModelConfig.MODEL_SAVE_PATH, map_location=self.device)
            
            # 检查checkpoint格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 新格式：包含归一化参数
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.y_mean = checkpoint.get('y_mean')
                self.y_std = checkpoint.get('y_std')
                self.X_mean = checkpoint.get('X_mean')
                self.X_std = checkpoint.get('X_std')
                print(f"✅ 已加载最佳模型（含归一化参数）：{ModelConfig.MODEL_SAVE_PATH}")
            else:
                # 旧格式：仅state_dict
                self.model.load_state_dict(checkpoint)
                print(f"✅ 已加载最佳模型（旧格式）：{ModelConfig.MODEL_SAVE_PATH}")
                print(f"⚠️ 归一化参数未找到，需要重新训练或手动设置")
            
            self.model.to(self.device)
        else:
            print("⚠️ 未找到保存的模型")
    
    def evaluate(self, test_loader):
        """评估模型"""
        print(f"\n📈 评估模型...")
        
        self.model.eval()
        predictions_list = []
        actuals_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)
                
                predictions_list.extend(predictions.cpu().numpy())
                actuals_list.extend(y_batch.numpy())
        
        predictions_np = np.array(predictions_list)
        actuals_np = np.array(actuals_list)
        
        # 计算指标
        mse = np.mean((predictions_np - actuals_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_np - actuals_np))
        r2 = 1 - (np.sum((actuals_np - predictions_np) ** 2) / 
                  np.sum((actuals_np - np.mean(actuals_np)) ** 2))
        
        print(f"\n📊 评估结果:")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE:  {mae:.2f}")
        print(f"   R²:   {r2:.4f}")
        
        return {
            'predictions': predictions_np,
            'actuals': actuals_np,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def plot_results(self, results):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练历史
        axes[0, 0].plot(self.history['train_loss'], label='训练损失')
        axes[0, 0].plot(self.history['val_loss'], label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练历史')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 预测 vs 真实
        axes[0, 1].scatter(results['actuals'], results['predictions'], 
                          alpha=0.5, s=10)
        min_val = min(results['actuals'].min(), results['predictions'].min())
        max_val = max(results['actuals'].max(), results['predictions'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 
                       'r--', lw=2, label='理想预测')
        axes[0, 1].set_xlabel('真实值')
        axes[0, 1].set_ylabel('预测值')
        axes[0, 1].set_title(f'预测对比 (R²={results["r2"]:.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 时序对比
        n_show = min(500, len(results['actuals']))
        axes[1, 0].plot(results['actuals'][:n_show], 
                       label='真实值', alpha=0.7)
        axes[1, 0].plot(results['predictions'][:n_show], 
                       label='预测值', alpha=0.7)
        axes[1, 0].set_xlabel('样本')
        axes[1, 0].set_ylabel('功率')
        axes[1, 0].set_title('时序对比（前500样本）')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差分布
        errors = results['actuals'] - results['predictions']
        axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('预测误差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title(f'误差分布 (MAE={results["mae"]:.2f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lstm_prediction_results.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ 结果图已保存: lstm_prediction_results.png")
        
        plt.close()
