import streamlit as st
import logging
import os
import argparse
from query_expander import QueryExpander
from vector_search import VectorSearch
from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    DEFAULT_TEMPERATURE,
    LOG_FORMAT,
    LOG_FILE,
    LOG_LEVEL,
    APP_TITLE,
    APP_ICON,
    APP_HEADER
)

# 配置日志
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# 初始化OpenAI客户端
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='expand',
                    help='查询处理模式：expand（扩展）、enhance（增强）、decompose（分解）、vector（向量搜索）、normal（基础向量检索）')
parser.add_argument('--top_k', type=int, default=5,
                    help='返回结果数量（仅vector/normal模式有效）')
args = parser.parse_args()

# 获取模式（优先使用命令行参数，其次使用环境变量，最后使用默认值）
MODE = args.mode or os.getenv('MODE', 'expand')
TOP_K = args.top_k or int(os.getenv('TOP_K', 5))

# 设置页面配置
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """主函数"""
    st.title(APP_ICON+APP_TITLE)
    st.header(APP_HEADER)
    
    # 显示当前模式
    mode_display = "查询扩展" if MODE == "expand" else "基础向量检索"
    st.sidebar.info(f"当前运行模式：{mode_display}")
    processor = QueryExpander(temperature=DEFAULT_TEMPERATURE)

    # 创建输入框
    user_query = st.text_input("请输入您的规则查询：", placeholder="例如：载具在近战范围内可以使用警戒射击技能吗？")

    # 创建提交按钮
    if st.button("提交问题"):
        if user_query:
            with st.spinner("机魂正在思索..."):
                try:                        
                    # 扩展查询
                    expanded_query = processor.expand_query(user_query)
                    # 对扩展查询进行搜索
                    st.write("\n搜索结果：")
                    results = processor.vector_search.search(expanded_query)
                    for i, result in enumerate(results, 1):
                        st.write(f"{i}. {result['text']}")                            
                except Exception as e:
                    st.error(f"处理查询时出错：{str(e)}")
        else:
            st.warning("请输入问题！")

    
    # 添加模式说明
    st.markdown("### 模式说明")
    st.markdown("""
        **查询扩展模式**：
        - 生成多个相关问题
        - 扩大检索范围
        - 整合所有相关信息
    """)


if __name__ == "__main__":
    main() 