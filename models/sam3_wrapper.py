import torch
import numpy as np
from .base_model import BaseSegmentationModel
from PIL import Image

# Use the actual sam3 library
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False

class SAM3Wrapper(BaseSegmentationModel):
    """
    Implementation of BaseSegmentationModel for Meta's SAM3.
    """
    
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load_model(self, checkpoint_path: str, model_type: str = "sam3_h"):
        """
        Load SAM3 model weights.
        """
        print(f"Loading SAM3 model from {checkpoint_path}...")
        if HAS_SAM3:
            # build_sam3_image_model handles the construction and weight loading
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                device=self.device,
                enable_inst_interactivity=True
            )
            self.processor = Sam3Processor(self.model, device=self.device)
        else:
            print("Warning: SAM3 library not found. Running in mock mode.")

    def predict_image(self, image: np.ndarray, prompts: dict = None) -> list:
        """
        Segment objects in an image using SAM3.
        """
        if not self.model and HAS_SAM3:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not HAS_SAM3:
            return []

        # Convert numpy array to PIL Image for the processor
        pil_img = Image.fromarray(image.astype('uint8'), 'RGB')
        state = self.processor.set_image(pil_img)
        
        results = []
        
        if prompts:
            # Handle text prompts
            if prompts.get("text"):
                state = self.processor.set_text_prompt(prompts["text"], state)
                if "masks" in state and state["masks"].numel() > 0:
                    # state["masks"] is Bx1xHxW or similar
                    masks = state["masks"].cpu().numpy()
                    # Return all masks found
                    results = [m[0] for m in masks]
            
            # Handle point or box prompts (interactive)
            # Note: For now, if both text and interactive are provided, text takes precedence
            elif prompts.get("points") is not None or prompts.get("boxes") is not None:
                masks, scores, logits = self.model.predict_inst(
                    state,
                    point_coords=prompts.get("points"),
                    point_labels=prompts.get("labels"),
                    box=prompts.get("boxes"),
                    multimask_output=True
                )
                best_idx = np.argmax(scores)
                results = [masks[best_idx]]
        
        return results

    def predict_video(self, video_path: str, output_path: str, prompts: dict = None):
        """
        Track and segment objects in a video using SAM3.
        """
        print(f"Processing video: {video_path}")
        # Placeholder for video tracking logic
        pass
