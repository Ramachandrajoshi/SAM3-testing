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
    def predict_image(self, image: np.ndarray, prompts: dict = None) -> list:
        """
        Predict masks for a single image.
        :param image: Input image as a numpy array (RGB).
        :param prompts: Dictionary containing prompts (points, boxes, text).
        :return: List of segmentation results.
        """
        pass

    @abstractmethod
    def predict_video(self, video_path: str, output_path: str, prompts: dict = None):
        """
        Process a video and save/return the segmented video.
        :param video_path: Path to input video file.
        :param output_path: Path to save the processed video.
        :param prompts: Prompts for the first frame or tracking.
        """
        pass
