"""
LSTM风电功率预测模型
"""
import torch
import torch.nn as nn


class LSTMWindPowerPredictor(nn.Module):
    """基于LSTM的风电功率预测模型"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, input_dim)
        返回:
            out: (batch, 1) 预测功率
        """
        # LSTM处理序列
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 通过输出层
        out = self.output_layer(last_output)
        
        return out.squeeze(-1)
    
    def get_model_info(self):
        """返回模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
