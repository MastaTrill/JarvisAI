from typing import Tuple
import numpy as np
import logging

class CameraTracker:
    """
    A class to manage camera position and orientation tracking.

    Attributes:
        position (np.ndarray): The current position of the camera.
        orientation (np.ndarray): The current orientation of the camera.
    """

    def __init__(self, initial_position: Tuple[float, float, float], initial_orientation: Tuple[float, float, float]):
        """
        Initializes the CameraTracker with a given position and orientation.

        Args:
            initial_position (Tuple[float, float, float]): The initial position of the camera.
            initial_orientation (Tuple[float, float, float]): The initial orientation of the camera.
        """
        self.position = np.array(initial_position, dtype=np.float32)
        self.orientation = np.array(initial_orientation, dtype=np.float32)
        logging.info(f"CameraTracker initialized with position: {self.position}, orientation: {self.orientation}")

    def update_position(self, new_position: Tuple[float, float, float]):
        """
        Updates the camera's position.

        Args:
            new_position (Tuple[float, float, float]): The new position of the camera.
        """
        self.position = np.array(new_position, dtype=np.float32)
        logging.debug(f"Camera position updated to: {self.position}")

    def update_orientation(self, new_orientation: Tuple[float, float, float]):
        """
        Updates the camera's orientation.

        Args:
            new_orientation (Tuple[float, float, float]): The new orientation of the camera.
        """
        self.orientation = np.array(new_orientation, dtype=np.float32)
        logging.debug(f"Camera orientation updated to: {self.orientation}")

    def get_camera_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the current state of the camera.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The current position and orientation of the camera.
        """
        return self.position, self.orientation

    def reset(self):
        """
        Resets the camera position and orientation to the initial state.
        """
        self.position = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)
        logging.info("CameraTracker reset to initial state.")