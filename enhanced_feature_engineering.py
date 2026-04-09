"""
增强版特征工程 - 添加时间、滞后、滚动窗口和交互特征
"""

import pandas as pd
import numpy as np


class EnhancedFeatureEngineer:
    """增强版特征工程器"""
    
    def __init__(self, data_path='data_feature_layer.csv'):
        """
        参数:
            data_path: 数据路径
        """
        self.data_path = data_path
        self.data = None
        
    def load_and_enhance(self):
        """加载并增强特征"""
        
        print("\n" + "🔥"*30)
        print("     增强版特征工程")
        print("🔥"*30 + "\n")
        
        # 加载基础数据
        print("📂 加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"✅ 加载完成: {self.data.shape}")
        
        # 确保datetime列存在
        if 'datetime' not in self.data.columns:
            # 生成时间序列
            self.data['datetime'] = pd.date_range(
                start='2023-01-01',
                periods=len(self.data),
                freq='H'
            )
        
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        features = {}
        
        # ========== 1. 基础特征 ==========
        print("\n📊 [1/8] 提取基础特征...")
        basic_cols = ['wind_speed', 'temperature', 'pressure', 'density']
        for col in basic_cols:
            if col in self.data.columns:
                features[col] = self.data[col]
        print(f"   ✓ {len([c for c in basic_cols if c in self.data.columns])}个基础特征")
        
        # ========== 2. 时间特征 ==========
        print("\n📅 [2/8] 构建时间特征...")
        features['hour'] = self.data['datetime'].dt.hour
        features['day_of_week'] = self.data['datetime'].dt.dayofweek
        features['month'] = self.data['datetime'].dt.month
        features['day_of_year'] = self.data['datetime'].dt.dayofyear
        features['is_weekend'] = (self.data['datetime'].dt.dayofweek >= 5).astype(int)
        
        # 周期性编码
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        print(f"   ✓ 11个时间特征")
        
        # ========== 3. 风向特征（如果存在） ==========
        if 'wind_direction' in self.data.columns:
            print("\n🧭 [3/8] 构建风向特征...")
            wind_dir = self.data['wind_direction'].fillna(0)
            wind_dir_rad = np.deg2rad(wind_dir)
            features['wind_dir_sin'] = np.sin(wind_dir_rad)
            features['wind_dir_cos'] = np.cos(wind_dir_rad)
            print(f"   ✓ 2个风向特征")
        else:
            print("\n🧭 [3/8] 跳过风向特征（数据中不存在）")
        
        # ========== 4. 滞后特征 ==========
        print("\n⏮️  [4/8] 构建滞后特征...")
        lag_features = ['wind_speed', 'temperature', 'pressure', 'density']
        lag_steps = [1, 3, 6, 12, 24]
        lag_count = 0
        
        for feature_name in lag_features:
            if feature_name in self.data.columns:
                for lag in lag_steps:
                    col_name = f'{feature_name}_lag{lag}'
                    features[col_name] = self.data[feature_name].shift(lag)
                    lag_count += 1
        
        print(f"   ✓ {lag_count}个滞后特征")
        
        # ========== 5. 滚动窗口统计特征 ==========
        print("\n📈 [5/8] 构建滚动窗口特征...")
        window_sizes = [3, 6, 12, 24]
        rolling_features = ['wind_speed', 'temperature']
        rolling_count = 0
        
        for feature_name in rolling_features:
            if feature_name in self.data.columns:
                for window in window_sizes:
                    # 均值
                    features[f'{feature_name}_roll{window}_mean'] = \
                        self.data[feature_name].rolling(window=window, min_periods=1).mean()
                    # 标准差
                    features[f'{feature_name}_roll{window}_std'] = \
                        self.data[feature_name].rolling(window=window, min_periods=1).std().fillna(0)
                    # 最大最小值
                    features[f'{feature_name}_roll{window}_max'] = \
                        self.data[feature_name].rolling(window=window, min_periods=1).max()
                    features[f'{feature_name}_roll{window}_min'] = \
                        self.data[feature_name].rolling(window=window, min_periods=1).min()
                    rolling_count += 4
        
        print(f"   ✓ {rolling_count}个滚动窗口特征")
        
        # ========== 6. 差分特征（变化率） ==========
        print("\n📉 [6/8] 构建差分特征...")
        diff_features = ['wind_speed', 'temperature', 'pressure']
        diff_count = 0
        
        for feature_name in diff_features:
            if feature_name in self.data.columns:
                features[f'{feature_name}_diff1'] = self.data[feature_name].diff(1).fillna(0)
                features[f'{feature_name}_diff24'] = self.data[feature_name].diff(24).fillna(0)
                features[f'{feature_name}_pct_change'] = self.data[feature_name].pct_change().fillna(0)
                diff_count += 3
        
        print(f"   ✓ {diff_count}个差分特征")
        
        # ========== 7. 交互特征 ==========
        print("\n🔗 [7/8] 构建交互特征...")
        if 'wind_speed' in self.data.columns and 'density' in self.data.columns:
            features['wind_power_theoretical'] = 0.5 * self.data['density'] * (self.data['wind_speed'] ** 3)
            features['wind_density_interaction'] = self.data['wind_speed'] * self.data['density']
            features['wind_speed_squared'] = self.data['wind_speed'] ** 2
            features['wind_speed_cubed'] = self.data['wind_speed'] ** 3
        
        if 'temperature' in self.data.columns and 'pressure' in self.data.columns:
            features['temp_pressure_ratio'] = self.data['temperature'] / (self.data['pressure'] / 1000)
            features['temp_pressure_product'] = self.data['temperature'] * (self.data['pressure'] / 1e5)
        
        print(f"   ✓ 6个交互特征")
        
        # ========== 8. 季节特征 ==========
        print("\n🍂 [8/8] 构建季节特征...")
        season_cols = [col for col in self.data.columns if col.startswith('season_')]
        season_count = 0
        for col in season_cols:
            features[col] = self.data[col]
            season_count += 1
        
        if season_count == 0:
            # 从月份生成季节
            month = features['month']
            for season, months in [('春', [3,4,5]), ('夏', [6,7,8]), ('秋', [9,10,11]), ('冬', [12,1,2])]:
                features[f'season_{season}'] = month.isin(months).astype(int)
                season_count += 1
        
        print(f"   ✓ {season_count}个季节特征")
        
        # ========== 转换为DataFrame并处理缺失值 ==========
        print("\n🔧 后处理...")
        feature_df = pd.DataFrame(features)
        
        # 填充缺失值
        feature_df = feature_df.bfill().ffill().fillna(0)
        
        # 添加目标变量
        if 'wind_power' in self.data.columns:
            feature_df['wind_power'] = self.data['wind_power']
        
        # 添加datetime
        feature_df['datetime'] = self.data['datetime']
        
        print("\n" + "="*80)
        print("✅ 增强特征工程完成!")
        print("="*80)
        print(f"\n📊 特征统计:")
        print(f"   总特征数: {len(feature_df.columns)-2} (不含datetime和wind_power)")
        print(f"   样本数: {len(feature_df)}")
        print(f"\n📋 特征类别:")
        print(f"   • 基础特征: {len(basic_cols)}")
        print(f"   • 时间特征: 11")
        print(f"   • 风向特征: {2 if 'wind_direction' in self.data.columns else 0}")
        print(f"   • 滞后特征: {lag_count}")
        print(f"   • 滚动窗口特征: {rolling_count}")
        print(f"   • 差分特征: {diff_count}")
        print(f"   • 交互特征: 6")
        print(f"   • 季节特征: {season_count}")
        
        return feature_df
    
    def save(self, output_path='data_feature_enhanced.csv'):
        """保存增强特征"""
        if self.data is None:
            print("⚠️  请先运行load_and_enhance()")
            return
        
        enhanced_df = self.load_and_enhance()
        enhanced_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 增强特征已保存至: {output_path}")
        
        return enhanced_df


def main():
    """主函数"""
    
    engineer = EnhancedFeatureEngineer('data_feature_layer.csv')
    enhanced_data = engineer.load_and_enhance()
    
    # 保存
    enhanced_data.to_csv('data_feature_enhanced.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 增强特征已保存至: data_feature_enhanced.csv")
    
    # 显示前几行
    print("\n📖 数据预览:")
    feature_cols = [col for col in enhanced_data.columns if col not in ['datetime', 'wind_power']]
    print(f"\n前5行（部分特征）:")
    print(enhanced_data[feature_cols[:10]].head())
    
    print("\n" + "="*80)
    print("✅ 完成！请使用 data_feature_enhanced.csv 训练LSTM模型")
    print("="*80)
    
    return enhanced_data


if __name__ == "__main__":
    enhanced_data = main()
