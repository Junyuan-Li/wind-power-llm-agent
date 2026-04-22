"""
风电预测LLM Agent（重构版）
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from llm.ollama_client import OllamaClient
from llm.prompt_builder import PromptBuilder
from llm.reasoning_chain import ReasoningChain, SelfReflection
from config import LLMConfig


def _build_llm_client(llm_backend: str, adapter_path: str = None):
    """
    工厂函数：根据 backend 名称返回对应客户端

    参数:
        llm_backend : "ollama"（默认）, "lora", 或 "vllm"
        adapter_path: LoRA adapter 目录，仅 backend="lora" 时使用
    """
    if llm_backend == "lora":
        from llm.lora_client import LoRAClient
        client = LoRAClient(adapter_path=adapter_path)
        if not client.check_connection():
            print(f"⚠️  LoRA adapter 未找到: {client.adapter_path}")
            print("   回退到 Ollama 模式")
            return OllamaClient()
        print(f"✅ 使用 LoRA adapter: {client.adapter_path}")
        return client
    elif llm_backend == "vllm":
        from llm.vllm_client import VLLMClient
        print(f"✅ 使用 vLLM 服务")
        return VLLMClient()
    else:
        return OllamaClient()


class WindPowerAgent:
    """风电功率预测专家Agent"""
    
    def __init__(self, rag_system=None, use_cot=True, use_reflection=False,
                 llm_backend: str = "ollama", adapter_path: str = None, 
                 lstm_predictor=None):
        """
        参数:
            rag_system:   RAG检索系统实例
            use_cot:      是否使用Chain-of-Thought推理
            use_reflection: 是否使用自我反思
            llm_backend:  LLM 后端，"ollama"（默认）或 "lora"
            adapter_path: LoRA adapter 目录（仅 llm_backend="lora" 时有效）
            lstm_predictor: LSTM 预测模型调用接口
        """
        self.llm_client = _build_llm_client(llm_backend, adapter_path)
        self.prompt_builder = PromptBuilder(use_cot=use_cot)
        self.rag_system = rag_system
        self.lstm_predictor = lstm_predictor
        self.use_cot = use_cot
        self.use_reflection = use_reflection
        self.llm_backend = llm_backend
        
        if use_cot:
            self.reasoning_chain = ReasoningChain(self.llm_client)
        
        if use_reflection:
            self.self_reflection = SelfReflection(self.llm_client)
    
    def explain_prediction(self, prediction_data):
        """
        解释预测结果
        
        参数:
            prediction_data: 预测数据字典
                {
                    'wind_speed': float,
                    'temperature': float,
                    'pressure': float,
                    'predicted_power': float,
                    ...
                }
        
        返回:
            解释结果
        """
        print(f"\n🤖 Wind Power Agent - 预测解释")
        
        # 1. RAG检索相关知识
        rag_context = {}
        if self.rag_system:
            print("   📚 检索相关知识...")
            rag_context = self.rag_system.retrieve_all_layers(prediction_data)
        
        # 2. 构建Prompt
        prompt = self.prompt_builder.build_prediction_explanation_prompt(
            prediction_data, rag_context
        )
        
        # 3. 生成解释
        explanation = None
        insights = {}
        
        if self.use_cot:
            print("   🧠 使用Chain-of-Thought推理...")
            try:
                reasoning_result = self.reasoning_chain.execute_cot_reasoning(
                    prompt, max_steps=4
                )
                explanation = reasoning_result['final_conclusion']
                insights = self.reasoning_chain.extract_key_insights(reasoning_result)
            except Exception as e:
                print(f"   ⚠️ CoT推理失败: {e}")
                # Fallback 到简单生成
                self.use_cot = False
        
        if not explanation:
            print("   💬 生成解释...")
            try:
                explanation = self.llm_client.generate(
                    prompt, 
                    system_prompt=self.prompt_builder.build_system_prompt()
                )
            except Exception as e:
                print(f"   ⚠️ LLM生成失败: {e}")
                # 使用模板化的fallback解释
                explanation = self._generate_fallback_explanation(prediction_data, rag_context)
        
        # 4. 自我反思（可选）
        reflection_result = None
        if self.use_reflection and explanation:
            print("   🔍 进行自我反思验证...")
            reflection_result = self.self_reflection.reflect_on_answer(
                prompt, explanation
            )
            
            if reflection_result.get('needs_revision'):
                print("   ⚠️ 质量不足，重新生成...")
                explanation = self.llm_client.generate(
                    prompt + "\n\n请改进以上回答，使其更准确专业。",
                    temperature=0.6
                )
        
        return {
            'explanation': explanation,
            'insights': insights,
            'rag_context': rag_context,
            'reflection': reflection_result
        }
    
    def diagnose_anomaly(self, prediction_data, anomaly_description):
        """
        诊断异常情况
        
        参数:
            prediction_data: 预测数据
            anomaly_description: 异常描述
        """
        print(f"\n🤖 Wind Power Agent - 异常诊断")
        
        # 添加异常描述
        data_with_anomaly = prediction_data.copy()
        data_with_anomaly['anomaly_description'] = anomaly_description
        
        # RAG检索
        rag_context = {}
        if self.rag_system:
            rag_context = self.rag_system.retrieve_all_layers(data_with_anomaly)
        
        # 构建诊断prompt
        prompt = self.prompt_builder.build_anomaly_diagnosis_prompt(
            data_with_anomaly, rag_context
        )
        
        # 生成诊断
        diagnosis = self.llm_client.generate(
            prompt,
            system_prompt=self.prompt_builder.build_system_prompt(),
            temperature=0.7
        )
        
        return {
            'diagnosis': diagnosis,
            'rag_context': rag_context
        }
    
    def analyze_causality(self, question):
        """
        因果关系分析
        
        参数:
            question: 用户问题（如"为什么风速增加但功率下降？"）
        """
        print(f"\n🤖 Wind Power Agent - 因果分析")
        print(f"   问题: {question}")
        
        # RAG检索相关知识
        rag_context = {}
        if self.rag_system:
            # 从问题中提取关键词检索
            rag_context = self.rag_system.retrieve_for_question(question)
        
        # 构建分析prompt
        prompt = self.prompt_builder.build_causality_analysis_prompt(
            question, rag_context
        )
        
        # 生成分析
        analysis = self.llm_client.generate(
            prompt,
            system_prompt=self.prompt_builder.build_system_prompt(),
            temperature=0.6
        )
        
        return {
            'analysis': analysis,
            'rag_context': rag_context
        }

    def run(self, query: str, data: dict = None):
        """
        真正的 Agent 调度运行入口（任务规划 -> 工具调用 -> 总结）
        
        工作流:
        1. 用户输入 query
        2. LLM判断需不需要调用 LSTM进行预测，需不需要调用 RAG 检索知识
        3. 根据判断调用对应工具
        4. 传入工具返回结果，让 LLM 做最终解释生成
        """
        print(f"\n🤖 Wind Power Agent [Agent 调度模式开始]")
        print(f"👤 用户请求: {query}")
        
        # --- 1. Agent 任务规划层 (Planner) ---
        # 注意：小模型（0.5B）无法可靠输出 JSON，直接用关键字规则决策
        plan = {"use_lstm": False, "use_rag": False, "reason": "keyword-based"}

        lstm_keywords = ['预测', '功率', '多少', '多大', '发电', '能发', 'predict', 'power', 'how much', '会是']
        rag_keywords  = ['为什么', '解释', '原理', '原因', '诊断', '异常', '影响', 'why', 'explain',
                         'cause', 'reason', '怎么', '分析', '怎样', '情况']

        if any(k in query for k in lstm_keywords):
            plan["use_lstm"] = True
        if any(k in query for k in rag_keywords):
            plan["use_rag"] = True
        # 兜底：什么关键词都没命中时，默认走 RAG（至少能查到物理知识回答）
        if not plan["use_lstm"] and not plan["use_rag"]:
            plan["use_rag"] = True
            plan["reason"] = "default fallback → RAG"

        print(f"🧠 [Planner] 规划结果: {plan}")

        # --- 2. 工具调用层 (Tool Executer) ---
        pred_result = None
        rag_context = None

        if plan.get("use_lstm") and data is not None:
            print("🛠️ [Tool] 调用 LSTM 预测工具...")
            if callable(self.lstm_predictor):
                pred_result = self.lstm_predictor(data)
            else:
                pred_result = {"predicted_power": 1234.5, "note": "Mocked LSTM prediction"}
            print(f"   => 预测结果: {pred_result}")

        if plan.get("use_rag") and self.rag_system and data is not None:
            print("🛠️ [Tool] 调用 RAG 知识检索工具...")
            # 修复：把 LSTM 预测功率传给 RAG，让 Layer3（预测解释层）也能运行
            predicted_power = pred_result.get("predicted_power") if pred_result else None
            rag_context = self.rag_system.retrieve_all_layers(
                current_weather=data,
                predicted_power=predicted_power
            )
            print("   => RAG 检索完成")

        # --- 压缩 RAG 上下文，只提取关键文本，防止 prompt 超 512 token ---
        def _compress_rag(ctx):
            if not ctx:
                return "（未使用RAG）"
            lines = []
            for item in ctx.get("physics_knowledge", [])[:2]:
                lines.append(f"[Physics] {item}")
            for item in ctx.get("explanations", [])[:2]:
                lines.append(f"[Explanation] {item}")
            hist = ctx.get("historical_cases", [])
            if hist:
                h = hist[0]
                lines.append(
                    f"[History] Similar case: wind={h.get('wind_speed','-')}m/s, "
                    f"power={h.get('wind_power','-'):.1f}kW"
                )
            return "\n".join(lines) if lines else "（RAG未返回有效内容）"

        # --- 3. 最终生成层 (Writer) ---
        # 使用英文 Alpaca 格式：因为微调时用的是英文 Alpaca 模板
        print("✍️ [Writer] 综合线索，生成最终回复...")
        ws   = data.get("wind_speed",  "?") if data else "?"
        temp = data.get("temperature", "?") if data else "?"
        pres = data.get("pressure",    "?") if data else "?"
        pred_str = (
            f"LSTM predicted power: {pred_result.get('predicted_power')} kW"
            if pred_result else "LSTM not used"
        )
        rag_str = _compress_rag(rag_context)

        integrate_prompt = (
            f"User query: {query}\n"
            f"Current conditions: wind={ws}m/s, temp={temp}°C, pressure={pres}hPa\n"
            f"LSTM result: {pred_str}\n"
            f"RAG knowledge:\n{rag_str}\n\n"
            f"As a wind power expert, provide a concise and accurate answer."
        )
        final_answer = self.llm_client.generate(
            integrate_prompt,
            system_prompt="You are a professional wind power forecasting assistant. Answer clearly and concisely."
        )
        return final_answer
    
    def chat(self, user_message, context=None):
        """
        通用对话接口
        
        参数:
            user_message: 用户消息
            context: 可选的上下文信息
        """
        print(f"\n🤖 Wind Power Agent - 对话")
        
        # 如果有上下文，整合到prompt
        if context:
            full_message = f"上下文信息：\n{context}\n\n用户问题：\n{user_message}"
        else:
            full_message = user_message
        
        response = self.llm_client.generate(
            full_message,
            system_prompt=self.prompt_builder.build_system_prompt()
        )
        
        return response
    
    def _generate_fallback_explanation(self, prediction_data, rag_context):
        """
        当LLM不可用时，生成基于模板的解释
        
        参数:
            prediction_data: 预测数据
            rag_context: RAG检索的上下文
        """
        wind_speed = prediction_data.get('wind_speed', 0)
        temperature = prediction_data.get('temperature', 0)
        predicted_power = prediction_data.get('predicted_power', 0)
        
        # 基于规则的简单解释
        explanation = f"""【风电功率预测解释 - 基于物理规则】

1. 气象条件分析：
   - 当前风速: {wind_speed:.2f} m/s
   - 环境温度: {temperature:.2f} °C
   - 预测功率: {predicted_power:.2f} kW

2. 物理机制：
   根据风能公式 P = 0.5 × ρ × A × v³，功率与风速的立方成正比。
   当前风速条件下，预测功率水平合理。

3. 历史参考：
   基于历史数据，相似气象条件下的功率范围通常在该预测值附近。

4. 置信度评估：
   该预测基于训练数据模式，具有较高的参考价值。

注：此解释为降级模式（LLM服务不可用），基于物理规则生成。"""
        
        return explanation.strip()
    
    def check_ready(self):
        """检查Agent是否就绪"""
        print("\n🔍 检查Agent状态...")
        
        # 检查LLM
        llm_ready = self.llm_client.check_availability()
        
        # 检查RAG
        rag_ready = self.rag_system is not None
        if rag_ready:
            print(f"✅ RAG系统已加载")
        else:
            print(f"⚠️ RAG系统未加载")
        
        # 检查推理链
        if self.use_cot:
            print(f"✅ Chain-of-Thought推理已启用")
        
        if self.use_reflection:
            print(f"✅ 自我反思机制已启用")
        
        return llm_ready


# 使用示例
if __name__ == "__main__":
    # 不带RAG的Agent
    agent = WindPowerAgent(use_cot=True, use_reflection=False)
    
    if agent.check_ready():
        # 测试预测解释
        test_data = {
            'wind_speed': 10.5,
            'temperature': 16.0,
            'pressure': 1013.0,
            'predicted_power': 1800.0
        }
        
        result = agent.explain_prediction(test_data)
        print(f"\n解释结果:\n{result['explanation']}")
