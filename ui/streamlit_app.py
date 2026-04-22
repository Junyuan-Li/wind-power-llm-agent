"""
🌪️ 风电智能预测解释系统 - Streamlit UI
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import ModelConfig, UIConfig, PROJECT_ROOT
from models import LSTMWindPowerPredictor, predict_with_lstm
from llm import WindPowerAgent


# 页面配置
st.set_page_config(
    page_title=UIConfig.PAGE_TITLE,
    page_icon=UIConfig.PAGE_ICON,
    layout=UIConfig.LAYOUT
)


@st.cache_resource
def load_model():
    """加载LSTM模型"""
    model_path = ModelConfig.MODEL_SAVE_PATH
    
    if not model_path.exists():
        st.error(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model = LSTMWindPowerPredictor(
        input_dim=ModelConfig.INPUT_DIM,
        hidden_dim=ModelConfig.HIDDEN_DIM,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    return model


@st.cache_resource
def load_agent():
    """加载LLM Agent"""
    agent = WindPowerAgent(use_cot=True, use_reflection=False)
    return agent


def main():
    """主界面"""
    
    # 标题
    st.title("🌪️ 风电智能预测解释系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")
        
        # 模式选择
        mode = st.radio(
            "选择模式",
            ["📊 预测 + 解释", "💬 专家问答", "📈 批量分析"],
            index=0
        )
        
        st.markdown("---")
        
        # 系统状态
        st.subheader("🔧 系统状态")
        
        model_loaded = False
        agent_loaded = False
        
        with st.spinner("加载模型..."):
            model = load_model()
            if model is not None:
                st.success("✅ LSTM模型已加载")
                model_loaded = True
            else:
                st.error("❌ LSTM模型加载失败")
        
        with st.spinner("加载Agent..."):
            agent = load_agent()
            if agent is not None:
                st.success("✅ LLM Agent已加载")
                agent_loaded = True
            else:
                st.warning("⚠️ LLM Agent未就绪")
    
    # 主内容区
    if mode == "📊 预测 + 解释":
        show_prediction_mode(model, agent, model_loaded, agent_loaded)
    
    elif mode == "💬 专家问答":
        show_qa_mode(agent, agent_loaded)
    
    elif mode == "📈 批量分析":
        show_batch_mode(model, agent, model_loaded)


def show_prediction_mode(model, agent, model_loaded, agent_loaded):
    """预测+解释模式"""
    
    st.header("📊 风电功率预测与智能解释")
    
    # 初始化预测结果
    predicted_power = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 输入气象数据")
        
        wind_speed = st.number_input(
            "🌬️ 风速 (m/s)",
            min_value=0.0,
            max_value=30.0,
            value=UIConfig.DEFAULT_WIND_SPEED,
            step=0.1
        )
        
        temperature = st.number_input(
            "🌡️ 温度 (°C)",
            min_value=-30.0,
            max_value=50.0,
            value=UIConfig.DEFAULT_TEMPERATURE,
            step=0.5
        )
        
        pressure = st.number_input(
            "📊 气压 (hPa)",
            min_value=900.0,
            max_value=1100.0,
            value=UIConfig.DEFAULT_PRESSURE,
            step=0.1
        )
        
        density = st.number_input(
            "💨 空气密度 (kg/m³)",
            min_value=0.8,
            max_value=1.5,
            value=UIConfig.DEFAULT_DENSITY,
            step=0.001,
            format="%.3f"
        )
        
        predict_button = st.button("🚀 开始预测", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📤 预测结果")
        
        if predict_button:
            if not model_loaded:
                st.error("❌ 模型未加载，无法预测")
            else:
                with st.spinner("⚡ 使用LSTM模型预测中..."):
                    # 使用已训练的LSTM模型进行预测
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    
                    predicted_power = predict_with_lstm(
                        model,
                        wind_speed=wind_speed,
                        temperature=temperature,
                        pressure=pressure,
                        density=density,
                        device=device
                    )
                    
                    if predicted_power is not None:
                        st.metric(
                            label="⚡ LSTM预测功率",
                            value=f"{predicted_power:.2f} kW",
                            delta=f"基于LSTM模型 - 风速 {wind_speed} m/s"
                        )
                    else:
                        st.error("❌ LSTM预测失败，请检查模型")
                    
                    # 显示输入摘要
                    with st.expander("📋 输入数据摘要"):
                        data_summary = pd.DataFrame({
                            '参数': ['风速', '温度', '气压', '空气密度'],
                            '数值': [
                                f"{wind_speed:.1f} m/s",
                                f"{temperature:.1f} °C",
                                f"{pressure:.1f} hPa",
                                f"{density:.3f} kg/m³"
                            ]
                        })
                        st.dataframe(data_summary, use_container_width=True)
    
    # 智能解释区域
    st.markdown("---")
    st.subheader("🧠 智能解释")
    
    if predict_button and model_loaded and predicted_power is not None:
        if not agent_loaded:
            st.warning("⚠️ LLM Agent未就绪，无法生成解释")
        else:
            with st.spinner("🤖 生成解释中..."):
                prediction_data = {
                    'wind_speed': wind_speed,
                    'temperature': temperature,
                    'pressure': pressure,
                    'density': density,
                    'predicted_power': predicted_power
                }
                
                try:
                    result = agent.explain_prediction(prediction_data)
                    
                    if result and result.get('explanation'):
                        st.markdown("### 📝 专家解释")
                        st.info(result['explanation'])
                        
                        # 显示RAG检索的知识
                        if result.get('rag_context'):
                            with st.expander("📚 参考知识"):
                                rag_ctx = result['rag_context']
                                
                                if rag_ctx.get('physics_knowledge'):
                                    st.markdown("**🔬 物理原理:**")
                                    for item in rag_ctx['physics_knowledge'][:3]:
                                        st.markdown(f"- {item}")
                                
                                if rag_ctx.get('historical_cases'):
                                    st.markdown("**📖 历史案例:**")
                                    for case in rag_ctx['historical_cases'][:2]:
                                        st.markdown(f"- 风速{case.get('wind_speed', 'N/A'):.1f}m/s → 功率{case.get('wind_power', 'N/A'):.1f}kW")
                    else:
                        st.error("❌ 解释生成失败，请检查Ollama服务")
                
                except Exception as e:
                    st.error(f"❌ 生成解释时出错: {e}")
                    st.info("💡 请确保 Ollama 服务正在运行：`ollama serve`")


def show_qa_mode(agent, agent_loaded):
    """专家问答模式"""
    
    st.header("💬 风电专家问答")
    
    if not agent_loaded:
        st.error("❌ LLM Agent未加载")
        return
    
    st.markdown("""
    向风电专家提问，获取专业解答。例如：
    - "为什么风速增加但功率反而下降？"
    - "空气密度如何影响风电功率？"
    - "什么是风电的切入风速和切出风速？"
    """)
    
    # 问题输入
    user_question = st.text_area(
        "🤔 您的问题:",
        placeholder="在这里输入您的问题...",
        height=100
    )
    
    ask_button = st.button("🚀 提问", type="primary")
    
    if ask_button and user_question:
        with st.spinner("🤖 思考中..."):
            try:
                response = agent.chat(user_question)
                
                if response:
                    st.markdown("### 📝 专家回答")
                    st.success(response)
                else:
                    st.error("❌ 回答生成失败")
            
            except Exception as e:
                st.error(f"❌ 出错: {e}")


def show_batch_mode(model, agent, model_loaded):
    """批量分析模式"""
    
    st.header("📈 批量数据分析")
    
    st.markdown("上传CSV文件进行批量预测分析")
    
    uploaded_file = st.file_uploader(
        "选择CSV文件",
        type=['csv'],
        help="文件需包含: wind_speed, temperature, pressure, density 列"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 数据预览")
            st.dataframe(df.head(10))
            
            if st.button("🚀 开始批量预测"):
                st.info("⚠️ 批量预测功能开发中...")
                st.markdown("""
                批量预测需要：
                1. 完整的时序特征工程
                2. 序列数据准备
                3. 模型推理优化
                
                请使用 `main.py` 的批量预测功能
                """)
        
        except Exception as e:
            st.error(f"❌ 文件读取失败: {e}")


# 页脚
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🌪️ 风电智能预测解释系统 v1.0 | 
        基于 LSTM + RAG + LLM Agent
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    show_footer()
