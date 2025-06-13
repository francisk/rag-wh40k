from typing import List, Dict, Any
import logging
from vector_search import VectorSearch
from config import (
    DEFAULT_TEMPERATURE,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    LOG_FORMAT,
    LOG_LEVEL,
    LLM_MODEL,
    RERANK_MODEL
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from collections import defaultdict
import os
from openai import OpenAI

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)

# 设置日志格式
formatter = logging.Formatter(LOG_FORMAT)
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(console_handler)

class QueryExpander:
    def __init__(self, openai_api_key=OPENAI_API_KEY, temperature=0.7):
        """
        初始化查询扩展器
        
        Args:
            openai_api_key: OpenAI API key
            temperature: LLM的温度参数
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.temperature = temperature
        self.vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME
        )
        logger.info("QueryExpander初始化完成")
        self.cache = {}  # 用于缓存查询结果
        
    def expand_query(self, query: str) -> str:
        """
        扩展查询并选择最契合用户意图的查询
        
        Args:
            query: 原始查询
            
        Returns:
            str: 最契合用户意图的扩展查询
        """
        try:
            if not query.strip():
                return ""
                
            logger.info(f"开始扩展查询：{query}")
            
            # 第一步：生成多个查询变体
            expand_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的战锤40K规则专家。请根据用户的查询生成多个相关的查询变体。
                要求：
                1. 每个变体应该关注查询的不同方面
                2. 变体之间应该相互独立
                3. 变体应该足够具体，便于检索
                4. 保持原始查询的意图
                5. 返回3-5个变体
                6. 每个变体应该是一个完整的问句"""),
                ("user", f"请为以下查询生成多个变体：{query}")
            ])
            
            expand_response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": f"请为以下查询生成多个变体：{query}"}],
                temperature=self.temperature
            )
            expanded_queries = [q.strip() for q in expand_response.choices[0].message.content.split('\n') if q.strip()]
            logger.info(f"生成的查询变体：{expanded_queries}")
            
            # 第二步：选择最契合用户意图的查询
            select_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的战锤40K规则专家。请从多个查询变体中选择最契合用户原始意图的一个。
                要求：
                1. 仔细分析原始查询的意图
                2. 评估每个变体与原始意图的契合度
                3. 选择最准确、最全面的变体
                4. 如果变体都不够理想，可以适当修改选中的变体
                5. 只返回最终选择的查询，不要包含其他内容"""),
                ("user", f"""原始查询：{query}

查询变体：
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(expanded_queries)])}

请选择最契合原始查询意图的变体，并返回。""")
            ])
            
            select_response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": f"""原始查询：{query}

查询变体：
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(expanded_queries)])}

请选择最契合原始查询意图的变体，并返回。"""}],
                temperature=self.temperature
            )
            selected_query = select_response.choices[0].message.content.strip()
            logger.info(f"选择的最契合查询：{selected_query}")
            
            return selected_query
            
        except Exception as e:
            logger.error(f"查询扩展出错：{str(e)}")
            return query  # 如果出错，返回原始查询

    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用bge-reranker-v2-m3模型对结果进行重排序
        
        Args:
            query: 原始查询
            results: 需要重排序的结果列表
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 重排序后的结果列表
        """
        if not results:
            return []
            
        # 使用Pinecone的reranker进行重排序
        reranked_results = self.vector_search.index.query(
            vector=self.vector_search.embeddings.embed_query(query),
            top_k=min(top_k, len(results)),  # 确保不超过输入结果数量
            include_metadata=True,
            hybrid_search=True,
            alpha=0.5,  # 平衡向量搜索和关键词搜索的权重
            rerank_config={
                "model": RERANK_MODEL,
                "top_k": min(top_k, len(results))  # 确保不超过输入结果数量
            }
        )
        
        # 处理重排序结果
        processed_results = []
        for match in reranked_results.matches:
            if hasattr(match, 'metadata') and match.metadata:
                processed_results.append({
                    'text': match.metadata.get('text', ''),
                    'score': match.score
                })
            
        return processed_results

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            
        Returns:
            Dict[str, Any]: 包含搜索结果和整合答案的字典
        """
        try:
            if not query.strip():
                return {
                    "results": [],
                    "integrated_answer": "请输入有效的查询。"
                }
                
            logger.info(f"开始处理查询：{query}")
            
            # 扩展查询
            expanded_query = self.expand_query(query)
            logger.info(f"扩展后的查询：{expanded_query}")
            
            # 执行向量搜索
            results = self.vector_search.search(expanded_query)
            logger.info(f"向量搜索结果数量：{len(results)}")
            
            # 重排序结果
            reranked_results = self.rerank_results(expanded_query, results)
            logger.info(f"重排序后结果数量：{len(reranked_results)}")
            
            # 整合结果
            integrated_answer = self.vector_search.integrate_results(query, reranked_results)
            logger.info("最终答案生成完成")
            
            return {
                "results": reranked_results,
                "integrated_answer": integrated_answer
            }
            
        except Exception as e:
            logger.error(f"处理查询时出错：{str(e)}")
            return {
                "results": [],
                "integrated_answer": f"处理查询时出错：{str(e)}"
            }
    
    def _process_sub_query(self, sub_query: str) -> str:
        """
        处理单个子查询
        
        Args:
            sub_query: 子查询
            
        Returns:
            str: 子查询的回答
        """
        try:
            # 使用向量搜索获取相关文档
            search_results = self.vector_search.search(sub_query)
            
            # 验证搜索结果
            if not search_results:
                logger.warning(f"未找到与查询 '{sub_query}' 相关的资料")
                return "未找到相关规则信息"
            
            # 准备参考资料文本
            reference_text = ""
            for result in search_results:
                if isinstance(result, dict) and 'text' in result:
                    reference_text += f"- {result['text']}\n"
            
            if not reference_text.strip():
                logger.warning(f"搜索结果中没有有效的文本内容: {search_results}")
                return "未找到相关规则信息"
            
            # 使用LLM生成回答
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的战锤40K规则分析专家。请根据提供的参考资料，回答用户的问题。
                回答要准确、专业，并引用相关的规则来源。
                如果参考资料中没有相关信息，请直接说明"未找到相关规则信息"。"""),
                ("user", f"问题：{sub_query}\n\n参考资料：\n{reference_text}")
            ])
            
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": f"问题：{sub_query}\n\n参考资料：\n{reference_text}"}],
                temperature=self.temperature
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"处理子查询时出错: {str(e)}"
            logger.error(error_msg)
            return "处理查询时出现错误，请稍后重试"
    
    def _synthesize_answer(self, original_query: str, query_results: Dict[str, str]) -> str:
        """
        合成最终答案
        
        Args:
            original_query: 原始查询
            query_results: 查询结果字典
            
        Returns:
            str: 最终答案
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的战锤40K规则分析专家。请根据提供的查询结果，合成一个完整的答案。
            答案应该：
            1. 直接回答用户的问题
            2. 整合所有相关的信息，用具体的结果而不是模糊的概述来回答
            3. 保持逻辑性和连贯性
            4. 引用相关的规则来源
            5. 保持专业性和准确性"""),
            ("user", f"""原始问题：{original_query}

查询结果：
{chr(10).join([f"问题：{query}\n回答：{result}\n" for query, result in query_results.items()])}""")
        ])
        
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": f"""原始问题：{original_query}

查询结果：
{chr(10).join([f"问题：{query}\n回答：{result}\n" for query, result in query_results.items()])}"""}],
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()

    def expand_and_search(self, query):
        # 扩展查询
        expanded_query = self.expand_query(query)
        # 返回扩展后的查询
        return expanded_query 