from typing import Dict, List, Optional
from vector_search import VectorSearch
import logging
from config import (
    DEFAULT_TEMPERATURE,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        """
        初始化查询处理器
        
        Args:
            temperature: 生成温度参数
        """
        self.vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME
        )
        self.llm = ChatOpenAI(
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.cache = {}  # 用于缓存子查询结果
        
    def process_query(self, query: str) -> str:
        """
        处理用户查询的主方法
        
        Args:
            query: 用户输入的查询
            
        Returns:
            str: 最终的回答
        """
        try:
            # 检查空查询
            if not query or not query.strip():
                return "错误：请输入有效的查询内容"
            
            # 1. 拆解查询
            decomposition = self.decomposer.decompose_query(query)
            logger.info(f"查询拆解结果: {decomposition}")
            
            # 检查拆解结果是否为空
            if not any(decomposition.values()):
                return "错误：无法理解您的查询，请尝试重新表述"
            
            # 2. 处理每个子查询
            sub_results = {}
            
            # 处理核心概念
            for concept in decomposition['core_concepts']:
                if concept not in self.cache:
                    result = self._process_sub_query(concept)
                    if result:
                        self.cache[concept] = result
                    else:
                        logger.warning(f"无法获取概念 '{concept}' 的结果")
                sub_results[concept] = self.cache.get(concept, "未找到相关信息")
            
            # 处理分析步骤
            for step in decomposition['analysis_steps']:
                if step not in self.cache:
                    result = self._process_sub_query(step)
                    if result:
                        self.cache[step] = result
                    else:
                        logger.warning(f"无法获取步骤 '{step}' 的结果")
                sub_results[step] = self.cache.get(step, "未找到相关信息")
            
            # 处理关键规则
            for rule in decomposition['key_rules']:
                if rule not in self.cache:
                    result = self._process_sub_query(rule)
                    if result:
                        self.cache[rule] = result
                    else:
                        logger.warning(f"无法获取规则 '{rule}' 的结果")
                sub_results[rule] = self.cache.get(rule, "未找到相关信息")
            
            # 3. 合成最终答案
            final_answer = self._synthesize_answer(query, decomposition, sub_results)
            return final_answer
            
        except Exception as e:
            error_msg = f"错误：处理您的查询时出现问题 - {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _process_sub_query(self, sub_query: str) -> str:
        """
        处理单个子查询
        
        Args:
            sub_query: 子查询
            
        Returns:
            str: 子查询的回答
        """
        try:
            # 清理特殊字符
            cleaned_query = sub_query.replace('\\', '').replace('\\circ', '°')
            
            # 使用向量搜索获取相关文档
            search_results = self.vector_search.search(cleaned_query)
            
            # 验证搜索结果
            if not search_results:
                logger.warning(f"未找到与查询 '{cleaned_query}' 相关的资料")
                return "未找到相关规则信息"
            
            # 准备参考资料文本
            reference_text = ""
            for result in search_results:
                if isinstance(result, dict) and 'text' in result:
                    # 清理结果中的特殊字符
                    cleaned_text = result['text'].replace('\\', '').replace('\\circ', '°')
                    reference_text += f"- {cleaned_text}\n"
            
            if not reference_text.strip():
                logger.warning(f"搜索结果中没有有效的文本内容: {search_results}")
                return "未找到相关规则信息"
            
            # 使用LLM生成回答
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的战锤40K规则分析专家。请根据提供的参考资料，回答用户的问题。
                回答要准确、专业，并引用相关的规则来源。
                如果参考资料中没有相关信息，请直接说明"未找到相关规则信息"。"""),
                ("user", f"问题：{cleaned_query}\n\n参考资料：\n{reference_text}")
            ])
            
            response = self.llm.invoke(prompt.format_messages())
            return response.content
            
        except Exception as e:
            error_msg = f"处理子查询时出错: {str(e)}"
            logger.error(error_msg)
            return "处理查询时出现错误，请稍后重试"
    
    def _synthesize_answer(self, original_query: str, decomposition: Dict[str, List[str]], sub_results: Dict[str, str]) -> str:
        """
        合成最终答案
        
        Args:
            original_query: 原始查询
            decomposition: 查询拆解结果
            sub_results: 子查询结果
            
        Returns:
            str: 最终答案
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的战锤40K规则分析专家。请根据提供的子查询结果，合成一个完整的答案。
            答案应该：
            1. 直接回答用户的问题
            2. 包含所有相关的核心概念解释
            3. 按照分析步骤的逻辑组织内容
            4. 引用相关的规则来源
            5. 保持专业性和准确性"""),
            ("user", f"""原始问题：{original_query}

核心概念：
{chr(10).join([f"- {concept}: {sub_results.get(concept, '')}" for concept in decomposition['core_concepts']])}

分析步骤：
{chr(10).join([f"- {step}: {sub_results.get(step, '')}" for step in decomposition['analysis_steps']])}

关键规则：
{chr(10).join([f"- {rule}: {sub_results.get(rule, '')}" for rule in decomposition['key_rules']])}""")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear() 

    def search(self, query, top_k=5):
        # 直接使用用户的query
        results = self.vector_search.search(query, top_k=top_k)
        return results 