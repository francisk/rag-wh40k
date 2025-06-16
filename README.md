# 战锤40K规则助手

一个基于RAG（检索增强生成）技术的战锤40K规则查询助手，帮助玩家快速查找和理解游戏规则。

## 功能特点

- 🔍 智能规则检索：快速定位相关规则内容
- 📚 多派系支持：支持多个战锤40K派系的规则查询
- 💡 上下文理解：基于上下文的智能问答
- 🚀 高性能：使用Pinecone向量数据库实现快速检索
- 🤖 智能对话：基于GPT模型的自然语言交互

## 技术栈

- Python 3.8+
- OpenAI GPT API
- Pinecone 向量数据库
- LangChain
- Streamlit

## 安装说明

1. 克隆项目
```bash
git clone [项目地址]
cd rag-wh40k
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建 `.env` 文件并添加以下配置：
```
OPENAI_API_KEY="你的OpenAI API密钥"
PINECONE_API_KEY="你的Pinecone API密钥"
```

## 项目结构

```
rag-wh40k/
├── DATAUPLOD/          # 数据上传相关脚本
├── app.py             # 主应用入口
├── config.py          # 配置文件
├── requirements.txt   # 项目依赖
└── README.md         # 项目说明文档
```

## 使用说明

1. 启动应用
```bash
streamlit run app.py
```

2. 在浏览器中访问 `http://localhost:8501`

3. 在输入框中输入你的问题，系统会返回相关的规则解释

## 数据上传

使用 `DATAUPLOD/upsert.py` 脚本上传新的规则数据：

```bash
python3 DATAUPLOD/upsert.py --source content.md --faction [派系名称] --chunk-size 1500 --chunk-overlap 150
```

参数说明：
- `--source`: 源文件路径
- `--faction`: 派系名称
- `--chunk-size`: 文本块大小
- `--chunk-overlap`: 文本块重叠大小
- `--output`: 输出文件名（可选）

## 配置说明

在 `config.py` 中可以修改以下配置：

- 模型参数（温度、top_k等）
- 嵌入模型选择
- LLM模型选择
- 应用标题和图标

## 注意事项

1. 请确保已正确配置API密钥
2. 建议使用虚拟环境运行项目
3. 首次运行需要等待模型加载

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

[添加许可证信息] 
