import unittest
import streamlit as st
from app import main, query_processor
import logging
from config import DEFAULT_TEMPERATURE

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestApp(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.query_processor = query_processor

    def test_simple_query(self):
        """测试简单查询"""
        logger.info("\n测试简单查询: 哪些人物单位可以使用手雷技能")
        response = self.query_processor.process_query("哪些人物单位可以使用手雷技能")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        logger.info(f"查询结果:\n{response}")

    def test_complex_query(self):
        """测试复杂查询"""
        logger.info("\n测试复杂查询: 星际战士和混沌星际战士在近战中的优势和劣势是什么？")
        response = self.query_processor.process_query("星际战士和混沌星际战士在近战中的优势和劣势是什么？")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        logger.info(f"查询结果:\n{response}")

    def test_empty_query(self):
        """测试空查询"""
        logger.info("\n测试空查询")
        response = self.query_processor.process_query("")
        self.assertIsNotNone(response)
        self.assertIn("错误", response)
        logger.info(f"查询结果:\n{response}")

    def test_multilingual_query(self):
        """测试多语言查询"""
        logger.info("\n测试多语言查询: What are the advantages and disadvantages of Space Marines vs Chaos Space Marines in melee combat?")
        response = self.query_processor.process_query("What are the advantages and disadvantages of Space Marines vs Chaos Space Marines in melee combat?")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        logger.info(f"查询结果:\n{response}")

if __name__ == '__main__':
    unittest.main() 