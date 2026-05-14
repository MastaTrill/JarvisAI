import unittest
from advanced_features.federated_learning import FederatedLearning

class TestFederatedLearning(unittest.TestCase):
    def test_train_average(self):
        fl = FederatedLearning()
        clients = [
            {'weights': [1.0, 2.0, 3.0]},
            {'weights': [2.0, 3.0, 4.0]},
            {'weights': [3.0, 4.0, 5.0]},
        ]
        model = {'weights': [0.0, 0.0, 0.0]}
        result = fl.train(clients, model)
        self.assertEqual(result['weights'], [2.0, 3.0, 4.0])

if __name__ == "__main__":
    unittest.main()