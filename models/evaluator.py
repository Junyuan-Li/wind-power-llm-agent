"""
LSTM模型评估器
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LSTMEvaluator:
    """LSTM模型评估器"""
    
    def __init__(self, model, device='cpu', y_mean=None, y_std=None):
        """
        参数:
            model: 训练好的模型
            device: 设备
            y_mean: 目标变量均值（用于反标准化）
            y_std: 目标变量标准差（用于反标准化）
        """
        self.model = model.to(device)
        self.device = device
        self.y_mean = y_mean
        self.y_std = y_std
        
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
        
        # 计算标准化尺度的指标
        results = self._compute_metrics(predictions_np, actuals_np)
        results['predictions'] = predictions_np
        results['actuals'] = actuals_np
        
        # 反标准化到原始尺度
        if self.y_mean is not None and self.y_std is not None:
            predictions_original = predictions_np * self.y_std + self.y_mean
            actuals_original = actuals_np * self.y_std + self.y_mean
            
            results_original = self._compute_metrics(predictions_original, actuals_original)
            results['predictions_original'] = predictions_original
            results['actuals_original'] = actuals_original
            results['rmse_original'] = results_original['rmse']
            results['mae_original'] = results_original['mae']
            results['r2_original'] = results_original['r2']
            results['mape_original'] = results_original['mape']
            
            print(f"\n📊 原始尺度评估结果:")
            print(f"   RMSE: {results_original['rmse']:.2f}")
            print(f"   MAE:  {results_original['mae']:.2f}")
            print(f"   R²:   {results_original['r2']:.4f}")
            print(f"   MAPE: {results_original['mape']:.2f}%")
        else:
            print(f"\n📊 标准化尺度评估结果:")
            print(f"   RMSE: {results['rmse']:.4f}")
            print(f"   MAE:  {results['mae']:.4f}")
            print(f"   R²:   {results['r2']:.4f}")
        
        return results
    
    def _compute_metrics(self, predictions, actuals):
        """计算评估指标"""
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE
        non_zero_mask = actuals != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
        else:
            mape = 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def plot_results(self, results, save_path='lstm_evaluation_results.png'):
        """绘制评估结果"""
        # 使用原始尺度数据（如果有）
        if 'predictions_original' in results:
            predictions = results['predictions_original']
            actuals = results['actuals_original']
            r2 = results['r2_original']
            mae = results['mae_original']
            title_suffix = "(原始尺度)"
        else:
            predictions = results['predictions']
            actuals = results['actuals']
            r2 = results['r2']
            mae = results['mae']
            title_suffix = "(标准化尺度)"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 预测 vs 真实
        axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=10)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', lw=2, label='理想预测')
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title(f'预测对比 {title_suffix} (R²={r2:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 时序对比
        n_show = min(500, len(actuals))
        axes[0, 1].plot(actuals[:n_show], label='真实值', alpha=0.7)
        axes[0, 1].plot(predictions[:n_show], label='预测值', alpha=0.7)
        axes[0, 1].set_xlabel('样本')
        axes[0, 1].set_ylabel('功率')
        axes[0, 1].set_title(f'时序对比（前{n_show}样本）')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 误差分布
        errors = actuals - predictions
        axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('预测误差')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title(f'误差分布 (MAE={mae:.2f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 误差散点图
        axes[1, 1].scatter(predictions, errors, alpha=0.5, s=10)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('预测值')
        axes[1, 1].set_ylabel('误差')
        axes[1, 1].set_title('残差分析')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 评估图表已保存: {save_path}")
        plt.close()
    
    def get_prediction_summary(self, results, n_samples=10):
        """获取预测摘要"""
        if 'predictions_original' in results:
            predictions = results['predictions_original']
            actuals = results['actuals_original']
        else:
            predictions = results['predictions']
            actuals = results['actuals']
        
        errors = actuals - predictions
        abs_errors = np.abs(errors)
        
        # 找出最好和最差的预测
        best_indices = np.argsort(abs_errors)[:n_samples]
        worst_indices = np.argsort(abs_errors)[-n_samples:]
        
        summary = {
            'best_predictions': [
                {
                    'index': int(idx),
                    'actual': float(actuals[idx]),
                    'predicted': float(predictions[idx]),
                    'error': float(errors[idx])
                }
                for idx in best_indices
            ],
            'worst_predictions': [
                {
                    'index': int(idx),
                    'actual': float(actuals[idx]),
                    'predicted': float(predictions[idx]),
                    'error': float(errors[idx])
                }
                for idx in worst_indices
            ]
        }
        
        return summary
