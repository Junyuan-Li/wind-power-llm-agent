import os
import sys

# 确保脚本能在根目录正常运行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.agent import WindPowerAgent

def main():
    print("🚀 正在初始化本地 LoRA Agent (这可能需要花费半分钟左右加载模型)...")
    
    # 将 backend 设为 lora，并指定刚才解压的 adapter 目录
    # 注意：这里会自动去 HuggingFace 下载 Qwen/Qwen2.5-0.5B 的基础模型到本地缓存，然后挂载 adapter
    agent = WindPowerAgent(
        llm_backend="lora", 
        adapter_path="wind_power_adapter"
    )
    
    print("\n✅ 初始化完成，开始测试对话...\n")
    print("=" * 50)
    
    test_queries = [
        "你好，请问你是谁？",
        "现在的风速是 2m/s，风机会启动吗？",
        "风速达到了25m/s，出于安全考虑系统会怎么做？"
    ]
    
    for query in test_queries:
        print(f"👤 [用户]: {query}")
        # WindPowerAgent 虽然封装了专业预测分析方法，但底层客户端仍在 llm_client 里
        response = agent.llm_client.generate(query)
        print(f"🤖 [Agent]: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
