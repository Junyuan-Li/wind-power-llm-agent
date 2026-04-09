"""
Chain-of-Thought 推理链
"""
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))


class ReasoningChain:
    """思维链推理器（Chain-of-Thought）"""
    
    def __init__(self, llm_client):
        """
        参数:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client
    
    def execute_cot_reasoning(self, initial_prompt, max_steps=4):
        """
        执行Chain-of-Thought推理
        
        参数:
            initial_prompt: 初始问题prompt
            max_steps: 最大推理步骤数
            
        返回:
            推理结果字典
        """
        print(f"\n🧠 开始Chain-of-Thought推理...")
        
        reasoning_steps = []
        current_context = initial_prompt
        
        for step in range(1, max_steps + 1):
            print(f"   步骤{step}/{max_steps}...")
            
            # 构建推理prompt
            step_prompt = self._build_step_prompt(current_context, step, max_steps)
            
            # LLM生成
            response = self.llm_client.generate(step_prompt, temperature=0.7)
            
            if not response:
                print(f"   ⚠️ 步骤{step}生成失败")
                break
            
            # 保存推理步骤
            reasoning_steps.append({
                'step': step,
                'content': response
            })
            
            # 更新上下文
            current_context += f"\n\n**步骤{step}结果**:\n{response}"
        
        # 生成最终结论
        final_prompt = self._build_final_prompt(initial_prompt, reasoning_steps)
        final_conclusion = self.llm_client.generate(final_prompt, temperature=0.6)
        
        # 如果最终结论生成失败，使用fallback
        if not final_conclusion:
            if reasoning_steps:
                final_conclusion = "基于以上分析步骤，预测结果合理。" + reasoning_steps[-1].get('content', '')
            else:
                final_conclusion = "推理过程未完成，无法生成结论。"
        
        print(f"✅ Chain-of-Thought推理完成")
        
        return {
            'reasoning_steps': reasoning_steps,
            'final_conclusion': final_conclusion
        }
    
    def _build_step_prompt(self, context, step, max_steps):
        """构建单步推理prompt"""
        if step == 1:
            instruction = "首先，分析输入的气象数据特征"
        elif step == 2:
            instruction = "其次，基于物理原理推理这些条件对风电发电的影响机制"
        elif step == 3:
            instruction = "然后，参考历史相似案例，验证推理的合理性"
        else:
            instruction = "最后，综合以上分析，评估预测的可信度"
        
        return f"""{context}

**当前推理步骤** ({step}/{max_steps}):
{instruction}

请简洁回答（2-3句话）："""
    
    def _build_final_prompt(self, initial_prompt, steps):
        """构建最终总结prompt"""
        steps_summary = "\n\n".join([
            f"**步骤{s['step']}**: {s['content']}"
            for s in steps
        ])
        
        return f"""{initial_prompt}

## 推理过程
{steps_summary}

## 最终任务
基于以上逐步推理，给出完整的预测解释（包含：气象分析、物理机制、历史对比、置信度评估）："""
    
    def extract_key_insights(self, reasoning_result):
        """从推理结果中提取关键见解"""
        # 处理空值情况
        if not reasoning_result or not isinstance(reasoning_result, dict):
            return {
                'meteorological_analysis': '推理结果为空',
                'physical_mechanism': '',
                'historical_comparison': '',
                'confidence_assessment': ''
            }
        
        final_text = reasoning_result.get('final_conclusion', '') or ''
        
        insights = {
            'meteorological_analysis': '',
            'physical_mechanism': '',
            'historical_comparison': '',
            'confidence_assessment': ''
        }
        
        # 简单的关键词匹配提取
        if final_text and ('气象' in final_text or '风速' in final_text):
            insights['meteorological_analysis'] = self._extract_section(final_text, ['气象', '风速', '温度'])
        
        if final_text and ('物理' in final_text or '原理' in final_text):
            insights['physical_mechanism'] = self._extract_section(final_text, ['物理', '原理', '机制'])
        
        if final_text and ('历史' in final_text or '案例' in final_text):
            insights['historical_comparison'] = self._extract_section(final_text, ['历史', '案例', '相似'])
        
        if final_text and ('置信' in final_text or '可信' in final_text):
            insights['confidence_assessment'] = self._extract_section(final_text, ['置信', '可信', '准确'])
        
        return insights
    
    def _extract_section(self, text, keywords):
        """提取包含关键词的文本片段"""
        sentences = re.split(r'[。！？\n]', text)
        relevant_sentences = []
        
        for sent in sentences:
            if any(kw in sent for kw in keywords):
                relevant_sentences.append(sent.strip())
        
        return '。'.join(relevant_sentences[:2])  # 最多返回2句


class SelfReflection:
    """自我反思机制"""
    
    def __init__(self, llm_client):
        """
        参数:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client
    
    def reflect_on_answer(self, original_question, generated_answer):
        """
        对生成的答案进行自我反思和验证
        
        参数:
            original_question: 原始问题
            generated_answer: 生成的答案
            
        返回:
            反思结果
        """
        reflection_prompt = f"""作为一个风电专家，请对以下回答进行评估：

**原始问题**:
{original_question}

**生成的回答**:
{generated_answer}

**评估维度**:
1. 准确性：回答是否符合风电物理原理？
2. 完整性：是否覆盖了关键要点？
3. 逻辑性：推理是否连贯合理？
4. 专业性：术语使用是否准确？

请给出评分（1-5分）和改进建议："""
        
        reflection = self.llm_client.generate(reflection_prompt, temperature=0.5)
        
        # 解析评分
        score = self._extract_score(reflection)
        
        return {
            'reflection': reflection,
            'score': score,
            'needs_revision': score < 4 if score else False
        }
    
    def _extract_score(self, text):
        """从反思文本中提取评分"""
        # 寻找数字模式
        patterns = [r'(\d)[分/点]', r'评分[:：]\s*(\d)', r'得分[:：]\s*(\d)']
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    return score
        
        return None


# 使用示例
if __name__ == "__main__":
    from ollama_client import OllamaClient
    
    client = OllamaClient()
    if client.check_availability():
        # 测试CoT推理
        chain = ReasoningChain(client)
        
        test_prompt = """当前气象条件：
        - 风速: 12.5 m/s
        - 温度: 18°C
        - 预测功率: 2500 kW
        
        请解释此预测结果。"""
        
        result = chain.execute_cot_reasoning(test_prompt, max_steps=3)
        print(f"\n最终结论:\n{result['final_conclusion']}")
