import unittest
from advanced_features.live_data_viz import LiveDataVisualization

class TestLiveDataVisualization(unittest.TestCase):
    def test_visualize(self):
        viz = LiveDataVisualization()
        data = [1, 2, 3, 4, 5]
        result = viz.visualize(data)
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['mean'], 3.0)

if __name__ == "__main__":
    unittest.main()