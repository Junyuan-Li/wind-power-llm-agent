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
        """
        try:
            temperature = temperature if temperature is not None else LLMConfig.TEMPERATURE
            max_tokens = max_tokens if max_tokens is not None else LLMConfig.MAX_TOKENS
            system_prompt = system_prompt if system_prompt is not None else LLMConfig.SYSTEM_PROMPT
            timeout_val = LLMConfig.TIMEOUT
        except:
            temperature = 0.7
            max_tokens = 512
            system_prompt = "You are a helpful assistant."
            timeout_val = 60
        
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
            "stream": True
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=(10, timeout_val),
                stream=True
            )
            response.raise_for_status()

            full_response = ""
            chunk_count = 0
            
            for content_chunk in response.iter_content(chunk_size=1024):
                if not content_chunk:
                    continue
                
                # 将bytes转换为str
                if isinstance(content_chunk, bytes):
                    content_chunk = content_chunk.decode('utf-8', errors='ignore')
                
                # 可能有多行数据
                for line in content_chunk.split('\n'):
                    if not line.strip():
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                            chunk_count += 1
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                
                if chunk_count > 0 and chunk.get('done', False):
                    break
            
            if not full_response:
                return None
            
            return full_response

        except:
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
