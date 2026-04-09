"""
全局配置文件
"""
import os
from pathlib import Path

# ============= 路径配置 =============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
MODELS_DIR = PROJECT_ROOT / "models"

# ============= 数据配置 =============
class DataConfig:
    """数据处理配置"""
    # 原始数据文件
    RAW_DATA_FILE = "data.csv"
    
    # 特征工程数据（新位置）
    FEATURE_DATA_FILE = FEATURES_DATA_DIR / "data_feature_enhanced.csv"
    
    # 时序配置
    SEQUENCE_LENGTH = 12  # 使用过去12小时预测
    TRAIN_RATIO = 0.64    # 训练集比例
    VAL_RATIO = 0.16      # 验证集比例
    TEST_RATIO = 0.20     # 测试集比例
    
    # 特征列
    TARGET_COL = 'wind_power'
    DATETIME_COL = 'datetime'
    BASE_FEATURES = ['wind_speed', 'temperature', 'pressure', 'density']

# ============= 模型配置 =============
class ModelConfig:
    """LSTM模型配置"""
    # 模型结构
    INPUT_DIM = 86        # 特征维度（自动从数据推断）
    HIDDEN_DIM = 64       # 隐藏层维度
    NUM_LAYERS = 2        # LSTM层数
    DROPOUT = 0.2         # Dropout比率
    SEQ_LEN = 12          # 序列长度
    
    # 训练参数
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 10
    
    # 学习率调度
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 3
    
    # 设备
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # 模型保存路径
    MODEL_SAVE_PATH = PROJECT_ROOT / "best_lstm_model.pth"
    
# ============= RAG配置 =============
class RAGConfig:
    """RAG系统配置"""
    # 向量数据库
    VECTOR_DB_PATH = PROJECT_ROOT / "faiss_index"
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    # 检索参数
    TOP_K = 5             # 检索Top-K文档
    RERANK_TOP_K = 3      # Rerank后保留Top-K
    
    # Chunk配置
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 50
    
    # 知识库路径（新位置）
    PHYSICS_KNOWLEDGE = KNOWLEDGE_DIR / "ml_knowledge.txt"
    KNOWLEDGE_CHUNKS = KNOWLEDGE_DIR / "knowledge_chunks.txt"
    
    # 历史数据检索（新位置）
    HISTORICAL_DATA_FILE = FEATURES_DATA_DIR / "data_feature_layer.csv"
    SIMILARITY_METHOD = "euclidean"  # euclidean / cosine

# ============= LLM配置 =============
class LLMConfig:
    """LLM Agent配置"""
    # Ollama配置
    OLLAMA_BASE_URL = "http://localhost:11434"
    MODEL_NAME = "llama3:latest"  # 使用完整模型名
    
    # 生成参数
    TEMPERATURE = 0.7
    TOP_P = 0.9
    MAX_TOKENS = 512   # 减小输出长度加快响应
    TIMEOUT = 180      # Ollama请求超时时间（秒）
    
    # Prompt模板
    SYSTEM_PROMPT = """你是一个风电功率预测领域的专家AI助手。
你的任务是：
1. 分析气象数据与风电功率的关系
2. 基于物理知识解释预测结果
3. 提供因果推理和异常诊断

请使用专业、准确、简洁的语言。"""
    
    # Chain-of-Thought
    USE_COT = True  # 是否使用思维链推理
    
# ============= 微调配置 =============
class FinetuneConfig:
    """LoRA微调配置"""
    # LoRA参数
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    
    # 训练参数
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    
    # 数据集（新位置）
    INSTRUCTION_DATASET = PROJECT_ROOT / "finetune" / "wind_power_instruction_dataset.json"
    
    # 基座模型
    # CPU 可跑的小模型（适合验证流程）：
    #   "Qwen/Qwen2.5-0.5B"          ~1GB，下载快，CPU 可训练
    #   "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  ~2GB，CPU 可训练
    # GPU 推荐（需 ≥8GB 显存 + QLoRA）：
    #   "meta-llama/Meta-Llama-3-8B"  需 Meta 授权
    #   "Qwen/Qwen2.5-7B"
    BASE_MODEL = "Qwen/Qwen2.5-0.5B"  # 默认小模型，CPU 可跑
    
# ============= 评估配置 =============
class EvalConfig:
    """评估配置"""
    # 预测评估指标
    METRICS = ['RMSE', 'MAE', 'R2', 'MAPE']
    
    # RAG评估
    RAG_EVAL_METRICS = ['Recall@K', 'Precision@K', 'MRR']
    
    # LLM评估（LLM-as-a-judge）
    LLM_JUDGE_MODEL = "gpt-4"
    EVAL_ASPECTS = ['准确性', '相关性', '可解释性', '专业性']
    
# ============= UI配置 =============
class UIConfig:
    """Streamlit UI配置"""
    PAGE_TITLE = "🌪️ 风电智能预测解释系统"
    PAGE_ICON = "🌪️"
    LAYOUT = "wide"
    
    # 默认输入值
    DEFAULT_WIND_SPEED = 8.5
    DEFAULT_TEMPERATURE = 15.0
    DEFAULT_PRESSURE = 1013.0
    DEFAULT_DENSITY = 1.225

# ============= 日志配置 =============
class LogConfig:
    """日志配置"""
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "logs" / "system.log"
    
# ============= 导出配置 =============
__all__ = [
    'DataConfig',
    'ModelConfig',
    'RAGConfig',
    'LLMConfig',
    'FinetuneConfig',
    'EvalConfig',
    'UIConfig',
    'LogConfig',
    'PROJECT_ROOT'
]
