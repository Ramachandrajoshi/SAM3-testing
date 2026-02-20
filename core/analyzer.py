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
    
    def analyze_image(self, image: np.ndarray, prompts: dict = None, input_size: int = None, skip_frames: int = 1):
        """
        Analyze a single image or frame and return the annotated result and object count.
        Parameters:
            image: Input image as a numpy array (RGB).
            prompts: Dictionary containing prompts (points, boxes, text).
            input_size: If provided, resize the image to this size (e.g., (256,256)).
            skip_frames: If provided, skip this many frames (for live stream optimization).
        """
        # Resize image for memory efficiency (critical for VRAM constraint)
        if input_size is not None:
            image = cv2.resize(image, (input_size, input_size))
        
        # 1. Get predictions from model
        masks = self.model.predict_image(image, prompts)
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
    
    def analyze_video_frame(self, frame: np.ndarray, prompts: dict = None, input_size: int = None, skip_frames: int = 1):
        """
        Analyze a single frame from a video stream/camera.
        """
        # Apply same optimizations as analyze_image
        if input_size is not None:
            frame = cv2.resize(frame, (input_size, input_size))
        
        return self.analyze_image(frame, prompts, input_size, skip_frames)
    
    def process_video_file(self, input_path: str, output_path: str, progress_callback=None):
        """
        Process a full video file.
        """
        count = self.model.predict_video(input_path, output_path)
        return count
