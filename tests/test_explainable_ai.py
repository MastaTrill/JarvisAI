import unittest
from advanced_features.explainable_ai import ExplainableAIDashboard

class TestExplainableAIDashboard(unittest.TestCase):
    def test_explain(self):
        dashboard = ExplainableAIDashboard()
        model = {'feature_names': ['a', 'b', 'c'], 'weights': [0.5, -1.2, 0.0]}
        data = [1, 2, 3]
        result = dashboard.explain(model, data)
        self.assertEqual(result, {'a': 0.5, 'b': 1.2, 'c': 0.0})

if __name__ == "__main__":
    unittest.main()