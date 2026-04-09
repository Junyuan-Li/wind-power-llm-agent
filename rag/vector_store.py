"""
向量数据库模块 - FAISS + HuggingFace Embeddings
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import RAGConfig

# 兼容不同版本的langchain
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.schema import Document
    except ImportError:
        print("⚠️ Langchain未安装，向量检索功能将不可用")
        print("   安装: pip install langchain langchain-community sentence-transformers faiss-cpu")


class VectorKnowledgeBase:
    """向量化知识库 - 实现语义检索"""
    
    def __init__(self, embedding_model_name=None):
        """
        初始化向量知识库
        
        参数:
            embedding_model_name: HuggingFace模型名称
        """
        self.embedding_model_name = embedding_model_name or RAGConfig.EMBEDDING_MODEL
        self.embeddings = None
        self.vector_store = None
        self.chunks = []
        
    def initialize_embeddings(self):
        """初始化embedding模型"""
        print("🔧 初始化Embedding模型...")
        print(f"   模型: {self.embedding_model_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Embedding模型加载完成")
        except Exception as e:
            print(f"❌ Embedding模型加载失败: {e}")
            self.embeddings = None
        
    def load_chunks_from_file(self, chunks_file: str):
        """
        从文件加载chunks
        
        参数:
            chunks_file: chunks文件路径
        """
        print(f"\n📥 从文件加载chunks: {chunks_file}")
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                # 简单按段落分割
                content = f.read()
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                self.chunks = []
                for i, para in enumerate(paragraphs):
                    self.chunks.append({
                        'id': i,
                        'title': f'Chunk {i+1}',
                        'content': para,
                        'type': 'knowledge',
                        'source': chunks_file,
                        'length': len(para)
                    })
            
            print(f"✅ 加载了 {len(self.chunks)} 个知识块")
        except Exception as e:
            print(f"❌ 加载chunks失败: {e}")
            self.chunks = []
        
    def load_chunks_from_kb_manager(self, kb_manager):
        """
        从KnowledgeBase加载chunks
        
        参数:
            kb_manager: KnowledgeBase实例
        """
        print("\n📥 从知识库管理器加载chunks...")
        
        if not kb_manager.chunks:
            print("⚠️  知识库为空，请先加载和创建chunks")
            return
        
        self.chunks = kb_manager.chunks
        print(f"✅ 加载了 {len(self.chunks)} 个知识块")
        
    def build_vector_store(self):
        """构建向量数据库"""
        print("\n🏗️  构建向量数据库...")
        
        if not self.chunks:
            print("⚠️  没有可用的chunks")
            return
        
        if self.embeddings is None:
            self.initialize_embeddings()
        
        if self.embeddings is None:
            print("❌ Embedding模型未就绪，无法构建向量库")
            return
        
        # 准备文档
        documents = []
        for chunk in self.chunks:
            # 组合标题和内容作为文档
            title = chunk.get('title', '')
            content = chunk.get('content', '')
            text = f"{title}\n{content}" if title else content
            
            # 创建Document对象
            doc = Document(
                page_content=text,
                metadata={
                    'chunk_id': chunk.get('id', 0),
                    'title': title,
                    'type': chunk.get('type', 'unknown'),
                    'source': chunk.get('source', ''),
                    'length': chunk.get('length', len(text))
                }
            )
            documents.append(doc)
        
        print(f"   准备向量化 {len(documents)} 个文档...")
        
        try:
            # 创建FAISS向量存储
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            print("✅ 向量数据库构建完成！")
        except Exception as e:
            print(f"❌ 向量数据库构建失败: {e}")
            self.vector_store = None
        
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        语义检索
        
        参数:
            query: 查询文本
            top_k: 返回top k个结果
            
        返回:
            检索结果列表
        """
        if self.vector_store is None:
            print("⚠️  向量数据库未构建")
            return []
        
        print(f"\n🔍 语义检索: '{query}'")
        
        try:
            # 执行相似度搜索
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=top_k
            )
            
            # 整理结果
            results = []
            for doc, score in docs_with_scores:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(1 / (1 + score)),  # 转换为相似度分数
                    'type': doc.metadata.get('type', 'unknown')
                }
                results.append(result)
            
            print(f"✅ 找到 {len(results)} 个相关结果")
            return results
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []
    
    def save_vector_store(self, save_path=None):
        """保存向量数据库"""
        save_path = save_path or RAGConfig.VECTOR_DB_PATH
        
        if self.vector_store is None:
            print("⚠️  没有可保存的向量数据库")
            return
        
        print(f"\n💾 保存向量数据库到: {save_path}")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            self.vector_store.save_local(str(save_path))
            
            # 保存chunks信息
            chunks_path = os.path.join(save_path, 'chunks.pkl')
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            print("✅ 保存完成！")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def load_vector_store(self, load_path=None):
        """加载已保存的向量数据库"""
        load_path = load_path or RAGConfig.VECTOR_DB_PATH
        
        print(f"\n📂 加载向量数据库从: {load_path}")
        
        try:
            if self.embeddings is None:
                self.initialize_embeddings()
            
            if self.embeddings is None:
                print("❌ Embedding模型未就绪")
                return False
            
            self.vector_store = FAISS.load_local(
                str(load_path),
                self.embeddings,
                allow_dangerous_deserialization=True  # 信任本地文件
            )
            
            # 加载chunks信息
            chunks_path = os.path.join(load_path, 'chunks.pkl')
            if os.path.exists(chunks_path):
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
            
            print("✅ 向量数据库加载完成！")
            return True
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
