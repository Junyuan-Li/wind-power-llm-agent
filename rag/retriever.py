"""
RAG检索器 - 3层检索系统
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))
from config import RAGConfig


class Layer1_PhysicsKnowledgeRAG:
    """第一层：物理知识检索"""
    
    def __init__(self, vector_kb=None):
        """
        参数:
            vector_kb: VectorKnowledgeBase实例
        """
        self.vector_kb = vector_kb
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索物理知识
        
        参数:
            query: 查询文本
            top_k: 返回前k个结果
            
        返回:
            知识列表
        """
        if self.vector_kb is None:
            return self._demo_physics_knowledge()
        
        # 使用向量数据库检索
        results = self.vector_kb.semantic_search(query, top_k=top_k)
        return results
    
    def _demo_physics_knowledge(self) -> List[Dict]:
        """演示模式的物理知识"""
        return [
            {
                'content': '风能功率与风速的立方成正比：P = 0.5 × ρ × A × v³ × Cp，其中ρ是空气密度，A是扫掠面积，v是风速，Cp是功率系数。',
                'type': '物理',
                'score': 0.95
            },
            {
                'content': '空气密度受温度和气压影响：密度 = 气压 / (R × 温度)。温度降低或气压升高都会增加空气密度，从而提高风电功率。',
                'type': '物理',
                'score': 0.88
            },
            {
                'content': '风速在3-25 m/s范围内风机正常运行。低于切入风速（约3m/s）不发电，高于切出风速（约25m/s）停机保护。',
                'type': '物理',
                'score': 0.82
            }
        ]


class Layer2_HistoricalWeatherRAG:
    """第二层：历史相似天气检索"""
    
    def __init__(self, historical_data_path: str = None):
        """
        参数:
            historical_data_path: 历史数据路径
        """
        self.data = None
        self.data_path = historical_data_path or RAGConfig.HISTORICAL_DATA_FILE
        self._load_data()
        
    def _load_data(self):
        """加载历史数据"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Layer2 RAG加载历史数据: {self.data.shape}")
        except Exception as e:
            print(f"⚠️  Layer2 RAG无法加载数据: {e}")
            self.data = None
    
    def retrieve(self, current_weather: Dict, top_k: int = 5) -> List[Dict]:
        """
        检索历史相似天气
        
        参数:
            current_weather: 当前天气字典
            top_k: 返回前k个最相似样本
            
        返回:
            相似历史记录列表
        """
        if self.data is None:
            return self._demo_historical_cases(current_weather)
        
        feature_cols = ['wind_speed', 'temperature', 'pressure', 'density']
        available_cols = [col for col in feature_cols if col in self.data.columns]
        
        if not available_cols:
            return self._demo_historical_cases(current_weather)
        
        # 构建当前天气向量
        current_vector = np.array([
            current_weather.get(col, 0) for col in available_cols
        ]).reshape(1, -1)
        
        # 历史数据矩阵
        historical_matrix = self.data[available_cols].values
        
        # 计算相似度（欧式距离）
        distances = np.sqrt(np.sum((historical_matrix - current_vector) ** 2, axis=1))
        
        # 找到最相似的top_k个
        top_indices = np.argsort(distances)[:top_k]
        
        # 构建返回结果
        results = []
        for idx in top_indices:
            dist = float(distances[idx])
            # 改进相似度计算：使用更合理的公式
            # 当距离<5时，相似度>0.5；距离=10时，相似度≈0.09
            similarity = np.exp(-dist / 5.0)  # 指数衰减，更平滑
            
            similar_case = {
                'wind_speed': self.data.loc[idx, 'wind_speed'] if 'wind_speed' in self.data.columns else None,
                'temperature': self.data.loc[idx, 'temperature'] if 'temperature' in self.data.columns else None,
                'pressure': self.data.loc[idx, 'pressure'] if 'pressure' in self.data.columns else None,
                'density': self.data.loc[idx, 'density'] if 'density' in self.data.columns else None,
                'wind_power': self.data.loc[idx, 'wind_power'] if 'wind_power' in self.data.columns else None,
                'similarity': similarity,
                'distance': dist,
                'match_quality': '优秀' if similarity > 0.8 else '良好' if similarity > 0.5 else '一般' if similarity > 0.3 else '差'
            }
            results.append(similar_case)
        
        # 检查匹配质量
        avg_similarity = np.mean([r['similarity'] for r in results])
        if avg_similarity < 0.3:
            print(f"⚠️  Layer2匹配质量较低（平均相似度={avg_similarity:.3f}）")
            print(f"   可能原因：数据集中缺少风速{current_weather.get('wind_speed', 0):.1f}m/s附近的样本")
        
        return results
    
    def _demo_historical_cases(self, current_weather: Dict) -> List[Dict]:
        """演示模式的历史案例"""
        wind_speed = current_weather.get('wind_speed', 8.0)
        
        return [
            {
                'wind_speed': wind_speed + 0.2,
                'temperature': 276.0,
                'pressure': 101300,
                'density': 1.24,
                'wind_power': 0.5 * 1.24 * (wind_speed + 0.2) ** 3,
                'similarity': 0.98,
                'distance': 0.5
            },
            {
                'wind_speed': wind_speed - 0.3,
                'temperature': 274.5,
                'pressure': 101400,
                'density': 1.26,
                'wind_power': 0.5 * 1.26 * (wind_speed - 0.3) ** 3,
                'similarity': 0.95,
                'distance': 0.8
            }
        ]
    
    def get_statistics(self, similar_cases: List[Dict]) -> Dict:
        """计算历史相似案例的统计信息"""
        powers = [case['wind_power'] for case in similar_cases if case['wind_power'] is not None]
        
        if not powers:
            return {}
        
        return {
            'avg_power': np.mean(powers),
            'std_power': np.std(powers),
            'min_power': np.min(powers),
            'max_power': np.max(powers),
            'count': len(powers)
        }


class Layer3_PredictionExplanationRAG:
    """第三层：预测解释RAG"""
    
    def __init__(self):
        """初始化预测解释RAG"""
        self.feature_importance_rules = {
            'wind_speed': {
                'weight': 0.7,
                'explanation': '风速是影响风电功率的最关键因素，功率与风速的立方成正比'
            },
            'density': {
                'weight': 0.15,
                'explanation': '空气密度影响风能捕获效率，密度越大，单位体积空气动能越大'
            },
            'temperature': {
                'weight': 0.08,
                'explanation': '温度通过影响空气密度间接影响功率，低温增加密度'
            },
            'pressure': {
                'weight': 0.05,
                'explanation': '气压主要通过影响空气密度来作用于风电功率'
            }
        }
    
    def explain_prediction(
        self, 
        input_features: Dict,
        predicted_power: float,
        historical_avg: float = None
    ) -> Dict:
        """
        生成预测解释
        
        参数:
            input_features: 输入特征字典
            predicted_power: 预测功率
            historical_avg: 历史平均功率
            
        返回:
            解释字典
        """
        explanations = []
        
        # 风速分析
        wind_speed = input_features.get('wind_speed', 0)
        
        if wind_speed < 3:
            explanations.append({
                'factor': '风速过低',
                'impact': '负面',
                'detail': f'当前风速{wind_speed:.1f}m/s低于切入风速3m/s，风机无法有效发电'
            })
        elif wind_speed > 20:
            explanations.append({
                'factor': '风速极高',
                'impact': '正面但受限',
                'detail': f'当前风速{wind_speed:.1f}m/s接近额定风速，功率趋于饱和或进入保护模式'
            })
        elif 8 <= wind_speed <= 15:
            explanations.append({
                'factor': '风速理想',
                'impact': '正面',
                'detail': f'当前风速{wind_speed:.1f}m/s处于最佳发电区间[8-15m/s]，发电效率高'
            })
        
        # 密度分析
        density = input_features.get('density', 1.225)
        if density > 1.28:
            explanations.append({
                'factor': '空气密度高',
                'impact': '正面',
                'detail': f'空气密度{density:.3f}kg/m³高于标准值，有利于提升发电功率约{(density-1.225)/1.225*100:.1f}%'
            })
        elif density < 1.17:
            explanations.append({
                'factor': '空气密度低',
                'impact': '负面',
                'detail': f'空气密度{density:.3f}kg/m³低于标准值，降低发电效率约{(1.225-density)/1.225*100:.1f}%'
            })
        
        # 温度分析
        temperature = input_features.get('temperature', 288)
        if isinstance(temperature, (int, float)):
            temp_celsius = temperature - 273.15 if temperature > 200 else temperature
            if temp_celsius < 0:
                explanations.append({
                    'factor': '低温环境',
                    'impact': '正面',
                    'detail': f'温度{temp_celsius:.1f}℃较低，空气密度增大，利于发电'
                })
            elif temp_celsius > 30:
                explanations.append({
                    'factor': '高温环境',
                    'impact': '负面',
                    'detail': f'温度{temp_celsius:.1f}℃较高，空气密度减小，降低发电效率'
                })
        
        return {
            'predicted_power': predicted_power,
            'explanations': explanations,
            'confidence': self._calculate_confidence(input_features, explanations)
        }
    
    def _calculate_confidence(self, features: Dict, explanations: List[Dict]) -> float:
        """计算预测置信度"""
        base_confidence = 0.8
        
        warnings = [exp for exp in explanations if exp['impact'] == '警告']
        base_confidence -= len(warnings) * 0.15
        
        wind_speed = features.get('wind_speed', 0)
        if 8 <= wind_speed <= 15:
            base_confidence += 0.1
        
        return max(0.0, min(1.0, base_confidence))


class EnhancedRAGSystem:
    """增强版3层RAG系统整合"""
    
    def __init__(self, vector_kb=None):
        """
        参数:
            vector_kb: 向量知识库实例
        """
        self.layer1 = Layer1_PhysicsKnowledgeRAG(vector_kb)
        self.layer2 = Layer2_HistoricalWeatherRAG()
        self.layer3 = Layer3_PredictionExplanationRAG()
        
        print("✅ 增强版3层RAG系统初始化完成")
    
    def retrieve_all_layers(
        self, 
        current_weather: Dict,
        predicted_power: float = None,
        query_text: str = None
    ) -> Dict:
        """
        完整的3层RAG检索
        
        参数:
            current_weather: 当前天气
            predicted_power: 预测功率
            query_text: 查询文本（可选）
            
        返回:
            完整检索结果，格式化为Agent可用
        """
        # Layer 1: 物理知识检索
        if query_text is None:
            ws = current_weather.get('wind_speed', 0)
            temp = current_weather.get('temperature', 273)
            temp_c = temp - 273.15 if temp > 200 else temp
            query_text = f"风速{ws}m/s 温度{temp_c:.1f}℃"
        
        physics_knowledge = self.layer1.retrieve(query_text, top_k=RAGConfig.TOP_K)
        
        # Layer 2: 历史相似天气检索
        similar_cases = self.layer2.retrieve(current_weather, top_k=RAGConfig.TOP_K)
        
        # Layer 3: 预测解释
        if predicted_power is not None:
            stats = self.layer2.get_statistics(similar_cases)
            explanation = self.layer3.explain_prediction(
                current_weather,
                predicted_power,
                historical_avg=stats.get('avg_power')
            )
        else:
            explanation = None
        
        # 返回格式化结果（适配Agent接口）
        return {
            'physics_knowledge': [item['content'] for item in physics_knowledge],
            'historical_cases': similar_cases,
            'explanations': explanation['explanations'] if explanation else []
        }
    
    def retrieve_for_question(self, question: str) -> Dict:
        """
        针对问题的检索（用于因果分析）
        
        参数:
            question: 用户问题
            
        返回:
            检索到的知识
        """
        physics_knowledge = self.layer1.retrieve(question, top_k=RAGConfig.TOP_K)
        
        return {
            'physics_knowledge': [item['content'] for item in physics_knowledge],
            'historical_cases': [],
            'explanations': []
        }
