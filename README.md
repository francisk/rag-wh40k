# 查询处理系统

## 项目概述
本项目实现了一个WH40K（桌游）规则和数据查询处理系统，目前只实现了query扩展模式

## 功能特点
- **查询扩展**：生成查询变体，选择最贴合用户意图的查询生成完整答案。

## 项目结构
- `app.py`：主应用程序，处理用户查询并返回结果。
- `vector_search.py`：实现向量搜索功能。
- `query_expander.py`：实现查询扩展功能。
- `config.py`：配置文件，包含API密钥和模型配置。

## 使用方法
1. 确保已安装所有依赖项：
   ```bash
   pip install -r requirements.txt
   ```

2. 在 `config.py` 中配置OpenAI API Key和Pinecone API Key。

3. 运行应用程序：
   ```bash
   python3 app.py
   ```

4. 使用命令行参数选择查询模式：
   ```bash
   python3 app.py --mode vector  # 基础向量检索模式，目前废弃
   python3 app.py --mode expand  # 查询扩展模式，根据用户query扩展出3个query,选择一个最合适的
   python3 app.py --mode enhance  # 增强RAG FUSION模式，目前废弃
   ```

## 配置说明
- `OPENAI_API_KEY`：OpenAI API密钥。
- `PINECONE_API_KEY`：Pinecone API密钥。
- `PINECONE_INDEX_NAME`：Pinecone索引名称。
- `DEFAULT_TEMPERATURE`：默认温度参数。
- `EMBADDING_MODEL`：嵌入模型名称。
- `LLM_MODEL`：LLM模型名称。
- `RERANK_MODEL`：重排序模型名称。

## 日志配置
- 日志级别设置为 `INFO`，格式为 `%(asctime)s - %(name)s - %(levelname)s - %(message)s`。

## 注意事项
- 确保在 `config.py` 中正确配置API密钥和模型名称。
- 确保Pinecone索引已创建并可用。

## 贡献
欢迎提交问题和改进建议！ 