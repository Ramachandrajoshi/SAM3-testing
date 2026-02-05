import cv2
import numpy as np
from models.base_model import BaseSegmentationModel
from utils.visualization import draw_results

class NSPVisualAnalysisSystemAnalyzer:
    """
    Core Logic class that coordinates between the UI and the Model.
    Adheres to the Single Responsibility Principle.
    """
    
    def __init__(self, model: BaseSegmentationModel):
        self.model = model

    def analyze_image(self, image: np.ndarray, prompts: dict = None, confidence_threshold: float = 0.5):
        """
        Analyze a single image and return the annotated result and object count.
        """
        # 1. Get predictions from model
        masks = self.model.predict_image(image, prompts, confidence_threshold=confidence_threshold)
        count = len(masks)
        
        # 2. Draw results
        if count > 0:
            labels = [f"ID:{i+1}" for i in range(count)]
            annotated_image = draw_results(image, masks, labels=labels)
            # Add text overlay for count
            cv2.putText(annotated_image, f"Count: {count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            annotated_image = image
            
        return annotated_image, count

    def analyze_video_frame(self, frame: np.ndarray, prompts: dict = None, confidence_threshold: float = 0.5):
        """
        Analyze a single frame from a video stream/camera.
        """
        # Similar to analyze_image but optimized for video tracking if needed
        return self.analyze_image(frame, prompts, confidence_threshold=confidence_threshold)

    def process_video_file(self, input_path: str, output_path: str, prompts: dict = None, confidence_threshold: float = 0.5, progress_callback=None):
        """
        Process a full video file.
        """
        count = self.model.predict_video(input_path, output_path, prompts, confidence_threshold)
        return count
