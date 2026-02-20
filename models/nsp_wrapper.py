import torch
import numpy as np
from .base_model import BaseSegmentationModel
from PIL import Image

# Use the actual segmentation library
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False

class NSPVisualAnalysisSystemWrapper(BaseSegmentationModel):
    """
    Implementation of BaseSegmentationModel for NSP Visual Analysis System.
    """
    
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load_model(self, checkpoint_path: str, model_type: str = "nsp_h"):
        """
        Load NSP Visual Analysis System model weights.
        """
        print(f"Loading NSP Visual Analysis System model from {checkpoint_path}...")
        if HAS_LIBRARY:
            # build_sam3_image_model handles the construction and weight loading
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                device=self.device,
                enable_inst_interactivity=True
            )
            self.processor = Sam3Processor(self.model, device=self.device)
        else:
            print("Warning: NSP Visual Analysis System library not found. Running in mock mode.")

    def predict_image(self, image: np.ndarray, prompts: dict = None) -> list:
        """
        Segment objects in an image using NSP Visual Analysis System.
        """
        if not self.model and HAS_LIBRARY:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not HAS_LIBRARY:
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

    def predict_all(self, image: np.ndarray, grid_n: int = 12,
                    iou_threshold: float = 0.70,
                    min_area_ratio: float = 0.0005,
                    max_area_ratio: float = 0.25) -> list:
        """
        Auto-segment ALL objects via a grid of center-point prompts.
        Returns a deduplicated list of binary H×W boolean masks,
        sorted by area ascending (small objects on top in overlay).
        """
        if not HAS_LIBRARY:
            return []
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        h, w      = image.shape[:2]
        total_px  = h * w
        min_area  = int(total_px * min_area_ratio)
        max_area  = int(total_px * max_area_ratio)

        pil_img   = Image.fromarray(image.astype("uint8"), "RGB")
        state     = self.processor.set_image(pil_img)

        kept_masks: list = []

        step_y = h / (grid_n + 1)
        step_x = w / (grid_n + 1)

        for row in range(1, grid_n + 1):
            for col in range(1, grid_n + 1):
                cy = np.array([[col * step_x, row * step_y]], dtype=float)
                cl = np.array([1], dtype=int)
                try:
                    masks, scores, _ = self.model.predict_inst(
                        state,
                        point_coords=cy,
                        point_labels=cl,
                        box=None,
                        multimask_output=True,
                    )
                    best = masks[int(np.argmax(scores))].astype(bool)
                except Exception:
                    continue

                area = int(best.sum())
                if area < min_area or area > max_area:
                    continue          # skip tiny noise and large background blobs

                # IoU deduplication
                duplicate = False
                for existing in kept_masks:
                    inter = int((best & existing).sum())
                    union = int((best | existing).sum())
                    if union > 0 and inter / union > iou_threshold:
                        duplicate = True
                        break
                if not duplicate:
                    kept_masks.append(best)

        # Sort by area ascending so small objects paint on top in the overlay
        kept_masks.sort(key=lambda m: int(m.sum()))
        return kept_masks

    def predict_video(self, video_path: str, output_path: str, prompts: dict = None):
        """
        Track and segment objects in a video using NSP Visual Analysis System.
        """
        print(f"Processing video: {video_path}")
        # Placeholder for video tracking logic
        # In a real implementation, this would return the total number of unique objects tracked
        return 0
