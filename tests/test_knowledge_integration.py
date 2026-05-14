import unittest
from advanced_features.knowledge_integration import KnowledgeIntegration

class TestKnowledgeIntegration(unittest.TestCase):
    def test_query(self):
        ki = KnowledgeIntegration()
        result = ki.query('wikipedia', 'AI')
        self.assertEqual(result['source'], 'wikipedia')
        self.assertEqual(result['query'], 'AI')
        self.assertIn('Simulated answer', result['result'])

if __name__ == "__main__":
    unittest.main()