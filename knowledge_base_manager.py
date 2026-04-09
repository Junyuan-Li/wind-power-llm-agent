"""
STEP 3-5: RAG系统实现
功能：加载知识库 → Chunk切分 → 向量化 → 检索
"""

import os
from pathlib import Path
from typing import List, Dict
import numpy as np


class KnowledgeBase:
    """知识库管理类"""
    
    def __init__(self, knowledge_dir="knowledge"):
        """
        初始化知识库
        
        参数:
            knowledge_dir: 知识库文件夹路径
        """
        self.knowledge_dir = knowledge_dir
        self.knowledge_files = {
            'wind_energy_physics': 'wind_energy_physics.txt',
            'meteorology_dynamics': 'meteorology_dynamics.txt', 
            'seasonal_patterns': 'seasonal_patterns.txt'
        }
        self.raw_knowledge = {}
        self.chunks = []
        
    def load_knowledge(self):
        """加载所有知识库文件"""
        print("📚 加载知识库...")
        
        for name, filename in self.knowledge_files.items():
            filepath = os.path.join(self.knowledge_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.raw_knowledge[name] = content
                    print(f"   ✅ {name}: {len(content)} 字符")
            else:
                print(f"   ⚠️  未找到: {filepath}")
        
        print(f"\n✅ 成功加载 {len(self.raw_knowledge)} 个知识库文件")
        return self.raw_knowledge
    
    def create_chunks(self):
        """
        将知识库切分成chunks
        
        根据STEP 4要求:
        - 单一知识点
        - 结构统一
        - 长度控制（200-400字）
        """
        print("\n✂️  开始Chunk切分...")
        
        self.chunks = []
        
        for kb_name, content in self.raw_knowledge.items():
            # 按段落分割（## 标题作为分隔符）
            sections = content.split('\n## ')
            
            # 确定知识类型
            if 'physics' in kb_name:
                kb_type = '物理'
            elif 'meteorology' in kb_name:
                kb_type = '气象'
            elif 'seasonal' in kb_name:
                kb_type = '季节'
            else:
                kb_type = '通用'
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                # 第一个section包含# 主标题，需要处理
                if i == 0 and section.startswith('#'):
                    section = section.split('\n', 1)[1] if '\n' in section else ''
                    if not section.strip():
                        continue
                
                # 提取标题和内容
                lines = section.strip().split('\n', 1)
                title = lines[0].strip()
                text = lines[1].strip() if len(lines) > 1 else ''
                
                if not text:
                    continue
                
                # 创建chunk
                chunk = {
                    'id': len(self.chunks),
                    'title': title,
                    'content': text,
                    'type': kb_type,
                    'source': kb_name,
                    'length': len(text)
                }
                
                self.chunks.append(chunk)
        
        print(f"✅ 生成 {len(self.chunks)} 个知识块")
        
        # 统计信息
        print(f"\n📊 Chunk统计：")
        print(f"   总数量: {len(self.chunks)}")
        print(f"   平均长度: {np.mean([c['length'] for c in self.chunks]):.0f} 字符")
        print(f"   最短: {min([c['length'] for c in self.chunks])} 字符")
        print(f"   最长: {max([c['length'] for c in self.chunks])} 字符")
        
        # 按类型统计
        type_counts = {}
        for chunk in self.chunks:
            t = chunk['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"\n📋 类型分布：")
        for t, count in type_counts.items():
            print(f"   {t}: {count} 个")
        
        return self.chunks
    
    def get_chunk_by_id(self, chunk_id: int):
        """根据ID获取chunk"""
        for chunk in self.chunks:
            if chunk['id'] == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_type(self, kb_type: str):
        """根据类型筛选chunks"""
        return [c for c in self.chunks if c['type'] == kb_type]
    
    def search_chunks_simple(self, query: str, top_k: int = 3):
        """
        简单的关键词搜索（用于演示，实际应使用向量检索）
        
        参数:
            query: 查询文本
            top_k: 返回top k个结果
        """
        print(f"\n🔍 搜索关键词: '{query}'")
        
        # 简单的关键词匹配打分
        results = []
        query_lower = query.lower()
        
        for chunk in self.chunks:
            score = 0
            content_lower = (chunk['title'] + ' ' + chunk['content']).lower()
            
            # 计算查询词出现次数
            for word in query_lower.split():
                score += content_lower.count(word)
            
            if score > 0:
                results.append({
                    'chunk': chunk,
                    'score': score
                })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top k
        top_results = results[:top_k]
        
        print(f"✅ 找到 {len(results)} 个相关结果，返回前 {len(top_results)} 个\n")
        
        for i, result in enumerate(top_results, 1):
            chunk = result['chunk']
            print(f"结果 {i} (分数: {result['score']}):")
            print(f"   标题: {chunk['title']}")
            print(f"   类型: {chunk['type']}")
            print(f"   内容: {chunk['content'][:100]}...")
            print()
        
        return top_results
    
    def save_chunks(self, output_path='knowledge_chunks.txt'):
        """保存chunks到文件（用于检查）"""
        print(f"\n💾 保存chunks到: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 风电知识库 Chunks\n\n")
            f.write(f"总数量: {len(self.chunks)}\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
            f.write("="*80 + "\n\n")
            
            for chunk in self.chunks:
                f.write(f"## Chunk {chunk['id']}\n\n")
                f.write(f"**标题**: {chunk['title']}\n")
                f.write(f"**类型**: {chunk['type']}\n")
                f.write(f"**来源**: {chunk['source']}\n")
                f.write(f"**长度**: {chunk['length']} 字符\n\n")
                f.write(f"**内容**:\n{chunk['content']}\n\n")
                f.write("-"*80 + "\n\n")
        
        print("✅ 保存完成！")


def main():
    """主函数 - 演示RAG系统基础功能"""
    
    print("\n" + "🧠 "*20)
    print("     风电预测系统 - RAG知识库系统")
    print("🧠 "*20 + "\n")
    
    # 1. 创建知识库
    kb = KnowledgeBase()
    
    # 2. 加载知识
    kb.load_knowledge()
    
    # 3. 创建chunks
    kb.create_chunks()
    
    # 4. 保存chunks
    kb.save_chunks()
    
    # 5. 演示搜索功能
    print("\n" + "="*80)
    print("🔍 演示搜索功能")
    print("="*80)
    
    # 示例查询1: 风速与功率
    kb.search_chunks_simple("风速 功率 立方", top_k=3)
    
    # 示例查询2: 温度影响
    kb.search_chunks_simple("温度 密度 影响", top_k=3)
    
    # 示例查询3: 季节规律
    kb.search_chunks_simple("冬季 风能 特征", top_k=3)
    
    print("\n" + "="*80)
    print("✅ RAG知识库系统演示完成！")
    print("="*80)
    
    print("\n📦 输出文件：")
    print("   - knowledge_chunks.txt (所有知识块)")
    
    print("\n💡 下一步：")
    print("   - STEP 5: 使用FAISS或Chroma进行向量化存储")
    print("   - STEP 6: 设计LLM Agent与RAG集成")
    print("   - STEP 7: 训练时序预测模型")
    
    return kb


if __name__ == "__main__":
    import pandas as pd
    kb = main()
