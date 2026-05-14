import unittest
from advanced_features.nlu_advanced import AdvancedNLU

class TestAdvancedNLU(unittest.TestCase):
    def test_understand(self):
        nlu = AdvancedNLU()
        context = {'memory': ['previous statement', 'last statement']}
        text = 'What is the weather?'
        result = nlu.understand(text, context)
        self.assertEqual(result['input'], text)
        self.assertEqual(result['context_used'], 'last statement')
        self.assertIn('Processed', result['reasoning'])

if __name__ == "__main__":
    unittest.main()