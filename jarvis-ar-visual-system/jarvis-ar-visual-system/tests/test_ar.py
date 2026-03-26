import unittest
from src.ar.camera_tracking import CameraTracker
from src.ar.object_detection import ObjectDetector
from src.ar.pose_estimation import PoseEstimator

class TestCameraTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = CameraTracker()

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)

    def test_track_camera(self):
        position, orientation = self.tracker.track()
        self.assertIsInstance(position, tuple)
        self.assertIsInstance(orientation, tuple)

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetector()

    def test_initialization(self):
        self.assertIsNotNone(self.detector)

    def test_detect_objects(self):
        objects = self.detector.detect()
        self.assertIsInstance(objects, list)

class TestPoseEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = PoseEstimator()

    def test_initialization(self):
        self.assertIsNotNone(self.estimator)

    def test_estimate_pose(self):
        pose = self.estimator.estimate()
        self.assertIsInstance(pose, dict)

if __name__ == '__main__':
    unittest.main()