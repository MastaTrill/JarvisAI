import unittest
from advanced_features.quantum_optimization import QuantumOptimization

class TestQuantumOptimization(unittest.TestCase):
    def test_optimize(self):
        qo = QuantumOptimization()
        problem = {'objective': lambda x: (x-2)**2}
        result = qo.optimize(problem)
        self.assertEqual(result['solution'], 2)
        self.assertEqual(result['value'], 0)

if __name__ == "__main__":
    unittest.main()