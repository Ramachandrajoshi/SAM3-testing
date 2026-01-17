import cv2
import numpy as np
from models.base_model import BaseSegmentationModel
from utils.visualization import draw_results

class SAM3Analyzer:
    """
    Core Logic class that coordinates between the UI and the Model.
    Adheres to the Single Responsibility Principle.
    """
    
    def __init__(self, model: BaseSegmentationModel):
        self.model = model

    def analyze_image(self, image: np.ndarray, prompts: dict = None):
        """
        Analyze a single image and return the annotated result.
        """
        # 1. Get predictions from model
        masks = self.model.predict_image(image, prompts)
        
        # 2. Draw results
        if len(masks) > 0:
            annotated_image = draw_results(image, masks)
        else:
            annotated_image = image
            
        return annotated_image

    def analyze_video_frame(self, frame: np.ndarray, prompts: dict = None):
        """
        Analyze a single frame from a video stream/camera.
        """
        # Similar to analyze_image but optimized for video tracking if needed
        return self.analyze_image(frame, prompts)

    def process_video_file(self, input_path: str, output_path: str, progress_callback=None):
        """
        Process a full video file.
        """
        return self.model.predict_video(input_path, output_path)
