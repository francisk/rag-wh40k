import unittest
import os
from vector_search import VectorSearch
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY

class TestVectorSearch(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # 设置环境变量
        self.vector_search = VectorSearch(pinecone_api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
        self.test_query = "游侠的技能是什么？"
        
    def test_generate_query_variants(self):
        """测试查询变体生成功能"""
        variants = self.vector_search.generate_query_variants(self.test_query)
        self.assertIsInstance(variants, list)
        self.assertGreater(len(variants), 0)
        for variant in variants:
            self.assertIsInstance(variant, str)
            self.assertTrue(len(variant) > 0)
            
    def test_semantic_parse(self):
        """测试语义解析功能"""
        context = ["游侠具有隐身技能", "游侠可以狙击敌人", "游侠移动速度很快"]
        answer = self.vector_search.semantic_parse(self.test_query, context)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        
    def test_search(self):
        """测试搜索功能"""
        results = self.vector_search.search(self.test_query, top_k=5)
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIn('text', result)
            self.assertIn('score', result)
            
    def test_search_and_integrate(self):
        """测试搜索并整合功能"""
        integrated_answer = self.vector_search.search_and_integrate(self.test_query, top_k=5)
        self.assertIsInstance(integrated_answer, str)
        self.assertGreater(len(integrated_answer), 0)

if __name__ == '__main__':
    unittest.main() 