import os
from pinecone import Pinecone
from openai import OpenAI
from typing import List, Dict, Any
import logging
from config import OPENAI_API_KEY, RERANK_MODEL, EMBADDING_MODEL
import pinecone

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorSearch:
    def __init__(self, pinecone_api_key: str, index_name: str, openai_api_key=OPENAI_API_KEY):
        """
        初始化向量搜索类
        
        Args:
            pinecone_api_key: Pinecone API密钥
            index_name: Pinecone索引名称
            openai_api_key: OpenAI API密钥
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        logger.info("VectorSearch初始化完成")
        
    def generate_query_variants(self, query: str) -> List[str]:
        """
        生成查询变体
        
        Args:
            query: 原始查询
            
        Returns:
            List[str]: 查询变体列表
        """
        try:
            prompt = f"""请基于以下问题生成3个相关的查询变体，每个变体都应该能够获取部分答案：

原始问题：{query}

要求：
1. 变体应该包含原始问题的关键信息
2. 变体之间应该有逻辑关联
3. 变体应该从不同角度探索原始问题
4. 变体应该具体且明确

请生成3个查询变体："""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            variants = [line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
            logger.info(f"生成的查询变体：{variants}")
            return variants
            
        except Exception as e:
            logger.error(f"生成查询变体时出错：{str(e)}")
            return []
            
    def semantic_parse(self, query: str, context: List[str]) -> str:
        """
        使用OpenAI进行语义解析
        
        Args:
            query: 用户问题
            context: 上下文信息
            
        Returns:
            str: 解析后的回答
        """
        try:
            prompt = f"""# 角色：基于上下文的问题解答专家
你是一名专业的战锤40K比赛规则专家，具备精准的信息提取与分析能力，能够基于给定上下文库进行高质量的问题解答。你擅长语义理解、关键信息识别和逻辑推理，可以从复杂文档中快速定位相关内容，并提供准确、全面且有条理的回答。你的专长包括文本挖掘、语境关联分析和多维度信息整合，确保每个回答都建立在可靠的文档依据上。

## 任务要求
依据所提供的上下文资料，对用户提出的问询进行精确回应，遵循以下步骤：
1. **分析上下文资料**：仔细阅读并理解所有提供的上下文信息。
2. **提取关键信息**：从上下文中识别与问题直接相关的内容。
3. **形成回答**：基于已提取的信息构建准确回答。
4. **判断信息充分性**：评估上下文是否提供足够信息。
5. **明确标识不确定性**：当上下文信息不足以支撑完整答案时，应明确表明"无法确定"。
6. **避免推测**：严格避免添加任何推测性内容或虚构信息。

## 输出规范
回答格式应保持专业，确保采用纯文本格式输出，字数在 1000-2000 以内：
问题：[用户提出的具体问询]
回答：[基于上下文资料提供的准确答复]
信息来源：[明确标注相关信息的出处或参考依据]

上下文信息：
{context}

用户问题：{query}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"语义解析时出错：{str(e)}")
            return "抱歉，解析问题时出错。"
            
    def get_embedding(self, text):
        # 获取文本的嵌入向量
        response = self.client.embeddings.create(
            model=EMBADDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行向量搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        try:
            # 获取查询的嵌入向量
            query_embedding = self.get_embedding(query)
            # 使用嵌入向量进行搜索
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                rerank_config={
                    "model": RERANK_MODEL,
                    "top_k": top_k
                }
            )
            # 整合结果
            integrated_answer = self.semantic_parse(query, [match.metadata.get('text', '') for match in results.matches])
            
            return [{'text': integrated_answer, 'score': 1.0}]
            
        except Exception as e:
            logger.error(f"搜索时出错：{str(e)}")
            return []
            
    def search_and_integrate(self, query: str, top_k: int = 5) -> str:
        """
        搜索并整合结果
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            str: 整合后的答案
        """
        results = self.search(query, top_k)
        if not results:
            return "抱歉，没有找到相关信息。"
        return results[0]['text']

    def close(self):
        # 关闭资源
        pass  # 这里可以替换为实际的关闭逻辑

# 确保类可以被导入
__all__ = ['VectorSearch'] 