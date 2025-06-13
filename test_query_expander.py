import unittest
from query_expander import QueryExpander
from config import DEFAULT_TEMPERATURE

class TestQueryExpander(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.expander = QueryExpander(temperature=DEFAULT_TEMPERATURE)
        self.test_query = "游侠的技能是什么？"
        
    def test_expand_query(self):
        """测试查询扩展功能"""
        # 测试基本扩展功能
        expanded_query = self.expander.expand_query(self.test_query)
        self.assertIsInstance(expanded_query, str)
        self.assertGreater(len(expanded_query), 0)
        
        # 测试空查询处理
        empty_result = self.expander.expand_query("")
        self.assertEqual(empty_result, "")
        
        # 测试查询是否包含原始意图
        self.assertIn("游侠", expanded_query.lower())
        self.assertIn("技能", expanded_query.lower())
        
    def test_rerank_results(self):
        """测试结果重排序功能"""
        # 准备测试数据
        test_results = [
            {"text": "游侠具有隐身技能", "score": 0.8},
            {"text": "游侠可以狙击敌人", "score": 0.9},
            {"text": "游侠移动速度很快", "score": 0.7}
        ]
        
        # 测试重排序
        reranked = self.expander.rerank_results(self.test_query, test_results)
        self.assertIsInstance(reranked, list)
        self.assertEqual(len(reranked), len(test_results))
        
        # 验证结果格式
        for result in reranked:
            self.assertIn("text", result)
            self.assertIn("score", result)
            
    def test_process_query(self):
        """测试完整的查询处理流程"""
        try:
            # 测试正常查询
            result = self.expander.process_query(self.test_query)
            self.assertIsInstance(result, dict)
            self.assertIn("results", result)
            self.assertIn("integrated_answer", result)
            
            # 测试空查询
            empty_result = self.expander.process_query("")
            self.assertIsInstance(empty_result, dict)
            self.assertIn("results", empty_result)
            self.assertIn("integrated_answer", empty_result)
            
        except Exception as e:
            self.fail(f"处理查询时出错：{str(e)}")

if __name__ == '__main__':
    unittest.main() 