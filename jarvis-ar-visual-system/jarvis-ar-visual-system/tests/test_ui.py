import unittest
from src.ui.interface_manager import InterfaceManager
from src.ui.hud_elements import HealthBar, ScoreDisplay
from src.ui.gesture_controls import GestureControls

class TestInterfaceManager(unittest.TestCase):
    def setUp(self):
        self.interface_manager = InterfaceManager()

    def test_add_hud_element(self):
        health_bar = HealthBar()
        self.interface_manager.add_hud_element(health_bar)
        self.assertIn(health_bar, self.interface_manager.hud_elements)

    def test_remove_hud_element(self):
        health_bar = HealthBar()
        self.interface_manager.add_hud_element(health_bar)
        self.interface_manager.remove_hud_element(health_bar)
        self.assertNotIn(health_bar, self.interface_manager.hud_elements)

class TestHealthBar(unittest.TestCase):
    def setUp(self):
        self.health_bar = HealthBar()

    def test_initial_health(self):
        self.assertEqual(self.health_bar.health, 100)

    def test_update_health(self):
        self.health_bar.update_health(80)
        self.assertEqual(self.health_bar.health, 80)

class TestScoreDisplay(unittest.TestCase):
    def setUp(self):
        self.score_display = ScoreDisplay()

    def test_initial_score(self):
        self.assertEqual(self.score_display.score, 0)

    def test_update_score(self):
        self.score_display.update_score(10)
        self.assertEqual(self.score_display.score, 10)

class TestGestureControls(unittest.TestCase):
    def setUp(self):
        self.gesture_controls = GestureControls()

    def test_recognize_gesture(self):
        gesture = "swipe_left"
        self.assertTrue(self.gesture_controls.recognize_gesture(gesture))

if __name__ == '__main__':
    unittest.main()