from abc import ABC, abstractmethod
import numpy as np

class BaseSegmentationModel(ABC):
    """
    Abstract Base Class for segmentation models.
    Follows the Interface Segregation and Dependency Inversion principles.
    """
    
    @abstractmethod
    def load_model(self, checkpoint_path: str):
        """Load the model weights from the given path."""
        pass

    @abstractmethod
    def predict_image(self, image: np.ndarray, prompts: dict = None, confidence_threshold: float = 0.5) -> list:
        """
        Predict masks for a single image.
        :param image: Input image as a numpy array (RGB).
        :param prompts: Dictionary containing prompts (points, boxes, text).
        :param confidence_threshold: Float between 0 and 1 for filtering results.
        :return: List of segmentation results.
        """
        pass

    @abstractmethod
    def predict_video(self, video_path: str, output_path: str, prompts: dict = None, confidence_threshold: float = 0.5):
        """
        Process a video and save/return the segmented video.
        :param video_path: Path to input video file.
        :param output_path: Path to save the processed video.
        :param prompts: Prompts for the first frame or tracking.
        :param confidence_threshold: Confidence threshold for detection.
        """
        pass
