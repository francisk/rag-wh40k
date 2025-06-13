import unittest
from query_processor import QueryProcessor
import logging
from config import DEFAULT_TEMPERATURE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestQueryProcessor(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.processor = QueryProcessor(temperature=DEFAULT_TEMPERATURE)
        
    def test_simple_query(self):
        """测试简单查询"""
        query = "哪些人物单位可以使用手雷技能"
        logger.info(f"\n测试简单查询: {query}")
        
        # 处理查询
        response = self.processor.process_query(query)
        
        # 验证结果
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        logger.info(f"查询结果:\n{response}")
        
    def test_complex_query(self):
        """测试复杂查询"""
        query = "星际战士和混沌星际战士在近战中的优势和劣势是什么？"
        logger.info(f"\n测试复杂查询: {query}")
        
        # 处理查询
        response = self.processor.process_query(query)
        
        # 验证结果
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        logger.info(f"查询结果:\n{response}")
        
    def test_cache_mechanism(self):
        """测试缓存机制"""
        query = "载具在近战范围内可以使用警戒射击技能吗？"
        logger.info(f"\n测试缓存机制: {query}")
        
        # 第一次查询
        response1 = self.processor.process_query(query)
        logger.info("第一次查询结果:")
        logger.info(response1)
        
        # 第二次查询（应该使用缓存）
        response2 = self.processor.process_query(query)
        logger.info("第二次查询结果:")
        logger.info(response2)
        
        # 验证结果
        self.assertEqual(response1, response2)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空查询
        query = ""
        logger.info("\n测试空查询")
        response = self.processor.process_query(query)
        self.assertIsInstance(response, str)
        self.assertTrue("错误" in response.lower())
        
        # 测试清除缓存
        self.processor.clear_cache()
        self.assertEqual(len(self.processor.cache), 0)
        
    def test_multilingual_query(self):
        """测试多语言查询"""
        query = "What are the advantages and disadvantages of Space Marines vs Chaos Space Marines in melee combat?"
        logger.info(f"\n测试多语言查询: {query}")
        
        # 处理查询
        response = self.processor.process_query(query)
        
        # 验证结果
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        logger.info(f"查询结果:\n{response}")

if __name__ == '__main__':
    unittest.main() 