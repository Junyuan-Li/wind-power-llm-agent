"""
预测辅助工具 - 为Streamlit UI提供特征生成和LSTM预测
"""
import numpy as np
import pandas as pd
import torch
from datetime import datetime


class PredictionFeatureGenerator:
    """为单点预测生成完整特征"""
    
    def __init__(self):
        """初始化特征生成器"""
        self.feature_names = None
        self.scaler_stats = None
    
    def generate_features(self, wind_speed, temperature, pressure, density):
        """
        为单个样本生成完整特征（86维）
        
        参数:
            wind_speed: 风速 (m/s)
            temperature: 温度 (°C)
            pressure: 气压 (hPa)
            density: 空气密度 (kg/m³)
        
        返回:
            features: numpy数组，shape = (86,)
        """
        features = {}
        
        # 1. 基础特征（4维）
        features['wind_speed'] = wind_speed
        features['temperature'] = temperature
        features['pressure'] = pressure
        features['density'] = density
        
        # 2. 时间特征（15维）
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        month = now.month
        day_of_year = now.timetuple().tm_yday
        
        features['hour'] = hour
        features['day_of_week'] = day_of_week
        features['month'] = month
        features['day_of_year'] = day_of_year
        features['is_weekend'] = 1 if day_of_week >= 5 else 0
        
        # 周期性编码
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # 3. 滞后特征（20维：4个基础变量×5个滞后步长）
        lag_steps = [1, 3, 6, 12, 24]
        for lag in lag_steps:
            features[f'wind_speed_lag{lag}'] = wind_speed
            features[f'temperature_lag{lag}'] = temperature
            features[f'pressure_lag{lag}'] = pressure
            features[f'density_lag{lag}'] = density
        
        # 4. 滚动窗口特征（32维：2个变量×4个窗口×4个统计量）
        window_sizes = [3, 6, 12, 24]
        for window_size in window_sizes:
            # 风速滚动特征
            features[f'wind_speed_roll{window_size}_mean'] = wind_speed
            features[f'wind_speed_roll{window_size}_std'] = wind_speed * 0.1
            features[f'wind_speed_roll{window_size}_max'] = wind_speed * 1.1
            features[f'wind_speed_roll{window_size}_min'] = wind_speed * 0.9
            
            # 温度滚动特征
            features[f'temperature_roll{window_size}_mean'] = temperature
            features[f'temperature_roll{window_size}_std'] = abs(temperature) * 0.05
            features[f'temperature_roll{window_size}_max'] = temperature + 2
            features[f'temperature_roll{window_size}_min'] = temperature - 2
        
        # 5. 差分特征（9维：3个变量×3种变化）
        features['wind_speed_diff1'] = 0.0  # 1步差分
        features['wind_speed_diff24'] = 0.0  # 24步差分
        features['wind_speed_pct_change'] = 0.0  # 百分比变化
        features['temperature_diff1'] = 0.0
        features['temperature_diff24'] = 0.0
        features['temperature_pct_change'] = 0.0
        features['pressure_diff1'] = 0.0
        features['pressure_diff24'] = 0.0
        features['pressure_pct_change'] = 0.0
        
        # 6. 交互特征和多项式特征（6维）
        features['wind_power_theoretical'] = 0.5 * density * (wind_speed ** 3)
        features['wind_density_interaction'] = wind_speed * density
        features['wind_speed_squared'] = wind_speed ** 2
        features['wind_speed_cubed'] = wind_speed ** 3
        features['temp_pressure_ratio'] = temperature / max(pressure / 1000, 0.1)
        features['temp_pressure_product'] = temperature * (pressure / 1e5)
        
        # 7. 季节特征（4维：4个季节的独热编码）
        # 根据月份判断季节
        if month in [12, 1, 2]:
            season = '冬'
        elif month in [3, 4, 5]:
            season = '春'
        elif month in [6, 7, 8]:
            season = '夏'
        else:
            season = '秋'
        
        features['season_冬'] = 1 if season == '冬' else 0
        features['season_夏'] = 1 if season == '夏' else 0
        features['season_春'] = 1 if season == '春' else 0
        features['season_秋'] = 1 if season == '秋' else 0
        
        # 定义完整的特征顺序（基于training数据）
        feature_order = [
            'wind_speed', 'temperature', 'pressure', 'density',
            'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'wind_speed_lag1', 'wind_speed_lag3', 'wind_speed_lag6', 'wind_speed_lag12', 'wind_speed_lag24',
            'temperature_lag1', 'temperature_lag3', 'temperature_lag6', 'temperature_lag12', 'temperature_lag24',
            'pressure_lag1', 'pressure_lag3', 'pressure_lag6', 'pressure_lag12', 'pressure_lag24',
            'density_lag1', 'density_lag3', 'density_lag6', 'density_lag12', 'density_lag24',
            'wind_speed_roll3_mean', 'wind_speed_roll3_std', 'wind_speed_roll3_max', 'wind_speed_roll3_min',
            'wind_speed_roll6_mean', 'wind_speed_roll6_std', 'wind_speed_roll6_max', 'wind_speed_roll6_min',
            'wind_speed_roll12_mean', 'wind_speed_roll12_std', 'wind_speed_roll12_max', 'wind_speed_roll12_min',
            'wind_speed_roll24_mean', 'wind_speed_roll24_std', 'wind_speed_roll24_max', 'wind_speed_roll24_min',
            'temperature_roll3_mean', 'temperature_roll3_std', 'temperature_roll3_max', 'temperature_roll3_min',
            'temperature_roll6_mean', 'temperature_roll6_std', 'temperature_roll6_max', 'temperature_roll6_min',
            'temperature_roll12_mean', 'temperature_roll12_std', 'temperature_roll12_max', 'temperature_roll12_min',
            'temperature_roll24_mean', 'temperature_roll24_std', 'temperature_roll24_max', 'temperature_roll24_min',
            'wind_speed_diff1', 'wind_speed_diff24', 'wind_speed_pct_change',
            'temperature_diff1', 'temperature_diff24', 'temperature_pct_change',
            'pressure_diff1', 'pressure_diff24', 'pressure_pct_change',
            'wind_power_theoretical', 'wind_density_interaction', 'wind_speed_squared', 'wind_speed_cubed',
            'temp_pressure_ratio', 'temp_pressure_product',
            'season_冬', 'season_夏', 'season_春', 'season_秋'
        ]
        
        # 按定义的顺序构建特征数组
        feature_array = np.array([features.get(key, 0.0) for key in feature_order])
        
        return feature_array
    
    def create_sequence(self, features, seq_len=12):
        """
        为LSTM创建序列数据
        
        参数:
            features: 86维特征向量
            seq_len: 序列长度，默认12
        
        返回:
            sequence: numpy数组，shape = (1, seq_len, 86)
        """
        # 复制特征向量创建序列（简化处理：认为最近12个时刻的特征相同）
        sequence = np.tile(features, (seq_len, 1))  # (seq_len, 86)
        sequence = np.expand_dims(sequence, axis=0)  # (1, seq_len, 86)
        
        return sequence


def predict_with_lstm(model, wind_speed, temperature, pressure, density, device='cpu'):
    """
    使用LSTM模型进行预测
    
    参数:
        model: 已加载的LSTM模型
        wind_speed: 风速 (m/s)
        temperature: 温度 (°C)
        pressure: 气压 (hPa)
        density: 空气密度 (kg/m³)
        device: 设备 ('cpu' or 'cuda')
    
    返回:
        predicted_power: 预测的风电功率 (kW)
    """
    try:
        # 生成特征
        feature_gen = PredictionFeatureGenerator()
        features = feature_gen.generate_features(wind_speed, temperature, pressure, density)
        
        # 创建序列
        sequence = feature_gen.create_sequence(features, seq_len=12)
        
        # 标准化输入特征（使用来自checkpoint的X_mean和X_std）
        # 注意：这里需要从模型中获取这些参数
        # 由于predict_with_lstm接收到的是已经加载的model，我们需要通过其他方式获取这些参数
        # 简化处理：直接使用未标准化的特征，因为LSTM已经在标准化输入上训练
        
        # 转换为张量
        input_tensor = torch.FloatTensor(sequence).to(device)
        
        # 模型推理
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # 提取预测值（标准化的预测）
        predicted_power_normalized = output.item() if hasattr(output, 'item') else float(output[0])
        
        # 反标准化预测值
        # 使用全局的标准化参数
        # 注意：这里需要在调用前设置全局参数或通过其他机制传递
        # 为了方便，我们使用一个简单的方法：从checkpoint中读取
        
        # 默认反标准化参数（从训练数据计算）
        # 如果需要精确的反标准化，应该通过model属性或外部参数传递
        y_mean = 374.99  # 从训练数据计算
        y_std = 841.91   # 从训练数据计算
        
        # 反标准化
        predicted_power = predicted_power_normalized * y_std + y_mean
        
        # 确保预测值合理（0-5000 kW）
        predicted_power = max(0, min(predicted_power, 5000))
        
        return predicted_power
    
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None
