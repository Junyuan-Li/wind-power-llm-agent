"""
Prompt构建器
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import LLMConfig


class PromptBuilder:
    """Prompt模板构建器"""
    
    def __init__(self, use_cot=None):
        """
        参数:
            use_cot: 是否使用Chain-of-Thought推理
        """
        self.use_cot = use_cot if use_cot is not None else LLMConfig.USE_COT
    
    def build_prediction_explanation_prompt(self, prediction_data, rag_context):
        """
        构建预测解释Prompt
        
        参数:
            prediction_data: 预测数据字典
                {
                    'wind_speed': float,
                    'temperature': float,
                    'predicted_power': float,
                    ...
                }
            rag_context: RAG检索到的知识
                {
                    'physics_knowledge': [...],
                    'historical_cases': [...],
                    'explanations': [...]
                }
        
        返回:
            完整的prompt字符串
        """
        # 格式化输入数据
        input_section = self._format_input_data(prediction_data)
        
        # 格式化RAG知识
        knowledge_section = self._format_rag_context(rag_context)
        
        # 构建基础prompt
        if self.use_cot:
            task_instruction = """请按照以下步骤分析:

1️⃣ **气象条件分析**：分析当前气象参数特征
2️⃣ **物理机制推理**：基于风电物理原理，解释这些条件如何影响发电
3️⃣ **历史案例参考**：对比相似历史场景，验证预测合理性
4️⃣ **综合结论**：给出最终预测解释和置信度评估

请详细说明每个步骤的推理过程。"""
        else:
            task_instruction = """请基于以上信息，解释本次风电功率预测结果：
- 分析气象条件对发电的影响
- 结合物理原理说明预测依据
- 参考历史案例验证合理性
- 评估预测的置信度"""
        
        prompt = f"""# 风电功率预测解释任务

## 📊 当前预测数据
{input_section}

## 📚 相关知识与案例
{knowledge_section}

## 🎯 任务要求
{task_instruction}

请以专业、准确、易懂的方式给出解释："""
        
        return prompt
    
    def build_anomaly_diagnosis_prompt(self, prediction_data, rag_context):
        """
        构建异常诊断Prompt
        
        参数:
            prediction_data: 预测数据（包含异常标记）
            rag_context: RAG知识上下文
        """
        input_section = self._format_input_data(prediction_data)
        knowledge_section = self._format_rag_context(rag_context)
        
        prompt = f"""# 风电功率异常诊断

## 🚨 观测数据
{input_section}

**异常描述**: {prediction_data.get('anomaly_description', '功率输出异常')}

## 📚 相关知识
{knowledge_section}

## 🔍 诊断任务
请分析可能的异常原因：
1. 气象因素异常
2. 设备运行异常
3. 季节性/时段性特征
4. 其他外部因素

给出诊断结论和建议措施："""
        
        return prompt
    
    def build_causality_analysis_prompt(self, question, rag_context):
        """
        构建因果分析Prompt
        
        参数:
            question: 用户问题（如"为什么风速增加功率反而下降？"）
            rag_context: RAG知识
        """
        knowledge_section = self._format_rag_context(rag_context)
        
        prompt = f"""# 风电因果关系分析

## ❓ 用户问题
{question}

## 📚 相关物理知识
{knowledge_section}

## 💡 分析要求
基于风电物理原理，深入分析：
1. 涉及的物理机制
2. 参数之间的因果关系
3. 可能的特殊情况说明
4. 实际案例参考

请给出科学、准确的解释："""
        
        return prompt
    
    def _format_input_data(self, data):
        """格式化输入数据"""
        lines = []
        
        # 气象参数
        if 'wind_speed' in data:
            lines.append(f"🌬️ **风速**: {data['wind_speed']:.2f} m/s")
        if 'temperature' in data:
            lines.append(f"🌡️ **温度**: {data['temperature']:.2f} °C")
        if 'pressure' in data:
            lines.append(f"📊 **气压**: {data['pressure']:.2f} hPa")
        if 'density' in data:
            lines.append(f"💨 **空气密度**: {data['density']:.4f} kg/m³")
        
        # 时间信息
        if 'datetime' in data:
            lines.append(f"📅 **时间**: {data['datetime']}")
        
        # 预测结果
        if 'predicted_power' in data:
            lines.append(f"\n⚡ **预测功率**: {data['predicted_power']:.2f} kW")
        
        if 'confidence' in data:
            lines.append(f"📈 **置信度**: {data['confidence']:.1%}")
        
        return "\n".join(lines)
    
    def _format_rag_context(self, context):
        """格式化RAG上下文"""
        sections = []
        
        # Layer 1: 物理知识
        if 'physics_knowledge' in context and context['physics_knowledge']:
            physics_items = []
            for i, item in enumerate(context['physics_knowledge'][:3], 1):
                physics_items.append(f"{i}. {item}")
            sections.append("### 🔬 物理原理\n" + "\n".join(physics_items))
        
        # Layer 2: 历史案例
        if 'historical_cases' in context and context['historical_cases']:
            cases_items = []
            for i, case in enumerate(context['historical_cases'][:2], 1):
                case_str = f"{i}. 风速{case.get('wind_speed', 'N/A'):.1f}m/s, " \
                          f"温度{case.get('temperature', 'N/A'):.1f}°C → " \
                          f"功率{case.get('wind_power', 'N/A'):.1f}kW"
                cases_items.append(case_str)
            sections.append("### 📖 相似历史案例\n" + "\n".join(cases_items))
        
        # Layer 3: 解释模板
        if 'explanations' in context and context['explanations']:
            exp_items = []
            for i, exp in enumerate(context['explanations'][:2], 1):
                exp_items.append(f"{i}. {exp}")
            sections.append("### 💡 解释参考\n" + "\n".join(exp_items))
        
        return "\n\n".join(sections) if sections else "无相关知识"
    
    def build_system_prompt(self):
        """构建系统prompt"""
        return LLMConfig.SYSTEM_PROMPT


# 使用示例
if __name__ == "__main__":
    builder = PromptBuilder(use_cot=True)
    
    # 测试数据
    pred_data = {
        'wind_speed': 8.5,
        'temperature': 15.0,
        'pressure': 1013.25,
        'predicted_power': 1250.5,
        'confidence': 0.85
    }
    
    rag_ctx = {
        'physics_knowledge': [
            "风电功率与风速的三次方成正比",
            "空气密度随温度升高而降低"
        ],
        'historical_cases': [
            {'wind_speed': 8.3, 'temperature': 14.8, 'wind_power': 1230.0}
        ]
    }
    
    prompt = builder.build_prediction_explanation_prompt(pred_data, rag_ctx)
    print(prompt)
