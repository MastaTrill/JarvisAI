import unittest
from advanced_features.orchestrator import AdvancedFeaturesOrchestrator

class TestAdvancedFeaturesOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orch = AdvancedFeaturesOrchestrator()

    def test_multimodal(self):
        result = self.orch.run_multimodal_demo()
        self.assertIn('fusion', result)

    def test_federated_learning(self):
        result = self.orch.run_federated_learning_demo()
        self.assertIn('weights', result)

    def test_explainable_ai(self):
        result = self.orch.run_explainable_ai_demo()
        self.assertIn('a', result)

    def test_self_healing(self):
        result = self.orch.run_self_healing_demo()
        self.assertEqual(result['status'], 'recovered')

    def test_workflow_automation(self):
        result = self.orch.run_workflow_automation_demo()
        self.assertTrue(result)

    def test_nlu(self):
        result = self.orch.run_nlu_demo()
        self.assertIn('reasoning', result)

    def test_knowledge_integration(self):
        result = self.orch.run_knowledge_integration_demo()
        self.assertIn('result', result)

    def test_live_data_viz(self):
        result = self.orch.run_live_data_viz_demo()
        self.assertEqual(result['mean'], 3.0)

    def test_quantum_optimization(self):
        result = self.orch.run_quantum_optimization_demo()
        self.assertEqual(result['solution'], 2)

if __name__ == "__main__":
    unittest.main()
