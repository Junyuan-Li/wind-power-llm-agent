"""
Ollama LLM客户端
"""
import requests
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import LLMConfig


class OllamaClient:
    """Ollama本地LLM客户端"""
    
    def __init__(self, base_url=None, model_name=None):
        """
        参数:
            base_url: Ollama服务地址
            model_name: 模型名称
        """
        self.base_url = base_url or LLMConfig.OLLAMA_BASE_URL
        self.model_name = model_name or LLMConfig.MODEL_NAME
        self.api_url = f"{self.base_url}/api/generate"
        
    def generate(self, prompt, system_prompt=None, temperature=None, 
                 max_tokens=None, stream=False):
        """
        生成文本
        
        参数:
            prompt: 用户输入
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式输出
            
        返回:
            生成的文本
        """
        temperature = temperature or LLMConfig.TEMPERATURE
        max_tokens = max_tokens or LLMConfig.MAX_TOKENS
        system_prompt = system_prompt or LLMConfig.SYSTEM_PROMPT
        
        # 构建完整prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        # 使用流式模式：边生成边读取，不受单次响应超时限制
        payload["stream"] = True

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=(10, LLMConfig.TIMEOUT),  # (连接超时, 读取超时)
                stream=True
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_response += chunk['response']
                    if chunk.get('done', False):
                        break
            return full_response

        except requests.exceptions.ReadTimeout:
            print(f"⏰ Ollama响应超时（>{LLMConfig.TIMEOUT}s），返回fallback")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama API调用失败: {e}")
            return None
    
    def chat(self, messages, temperature=None, max_tokens=None):
        """
        对话接口（支持多轮对话）
        
        参数:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            
        返回:
            生成的响应
        """
        # 转换为单个prompt（Ollama不直接支持chat格式）
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.insert(0, f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        return self.generate(
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def check_availability(self):
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                print(f"✅ Ollama服务可用")
                print(f"   可用模型: {', '.join(model_names)}")
                
                if self.model_name in model_names:
                    print(f"   当前模型 '{self.model_name}' 已加载")
                    return True
                else:
                    print(f"   ⚠️ 模型 '{self.model_name}' 未找到")
                    return False
            else:
                print(f"❌ Ollama服务不可用")
                return False
                
        except Exception as e:
            print(f"❌ 无法连接到Ollama服务: {e}")
            print(f"   请确保Ollama正在运行: ollama serve")
            return False


# 使用示例
if __name__ == "__main__":
    client = OllamaClient()
    
    # 检查服务
    if client.check_availability():
        # 测试生成
        response = client.generate("什么是风电功率预测？")
        print(f"\n生成结果:\n{response}")
