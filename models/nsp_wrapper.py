import torch
import numpy as np
import cv2
import traceback
from .base_model import BaseSegmentationModel
from PIL import Image
from utils.video_utils import VideoHandler
from utils.visualization import draw_results

# Use the actual segmentation library
try:
    from sam3.model_builder import (
        build_sam3_image_model, 
        build_sam3_video_model, 
        build_sam3_video_predictor
    )
    from sam3.model.sam3_image_processor import Sam3Processor
    # Import predictor for type checking/fallback, but prefer builder
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False
    build_sam3_image_model = None
    build_sam3_video_model = None
    build_sam3_video_predictor = None
    Sam3VideoPredictor = None
    Sam3Processor = None

class NSPVisualAnalysisSystemWrapper(BaseSegmentationModel):
    """
    Implementation of BaseSegmentationModel for NSP Visual Analysis System.
    """
    
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.video_predictor = None

    def load_model(self, checkpoint_path: str, model_type: str = "nsp_h"):
        """
        Load NSP Visual Analysis System model weights.
        """
        print(f"Loading NSP Visual Analysis System model from {checkpoint_path}...")
        print(f"Environment: CUDA available={torch.cuda.is_available()}, Device={self.device}")
        
        if HAS_LIBRARY:
            try:
                # 1. Initialize Image Model
                self.model = build_sam3_image_model(
                    checkpoint_path=checkpoint_path,
                    device=self.device,
                    enable_inst_interactivity=True
                )
                if self.model:
                    self.model.eval()
                    self.processor = Sam3Processor(self.model, device=self.device)
                    print("Image model loaded and set to eval mode.")
                
                # 2. Initialize Video Predictor (if possible)
                if build_sam3_video_predictor:
                    try:
                        print("Initializing native video predictor...")
                        # In SAM3, build_sam3_video_predictor is the recommended way to initialize
                        # It handles model loading and GPU distribution internally
                        gpus = [0] if "cuda" in str(self.device).lower() else None
                        self.video_predictor = build_sam3_video_predictor(
                            checkpoint_path=checkpoint_path,
                            gpus_to_use=gpus
                        )
                        print("Native video predictor initialized successfully.")
                    except Exception as ve:
                        print(f"Warning: Could not initialize native video predictor.")
                        print(f"Reason: {ve}")
                else:
                    print("Note: Native video components (build_sam3_video_predictor) not available.")
                    
            except Exception as e:
                print(f"Error during model loading: {e}")
                traceback.print_exc()
        else:
            print("Warning: NSP Visual Analysis System library not found. Running in mock mode.")

    def predict_image(self, image: np.ndarray, prompts: dict = None, confidence_threshold: float = 0.5) -> list:
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
                    # Filter by scores if available in state
                    masks = state["masks"].cpu().numpy()
                    if "scores" in state:
                        scores = state["scores"].cpu().numpy()
                        results = [masks[i][0] for i, score in enumerate(scores) if score >= confidence_threshold]
                    else:
                        # Fallback if library doesn't provide scores for text
                        results = [m[0] for m in masks]
            
            # Handle point or box prompts (interactive)
            elif prompts.get("points") is not None or prompts.get("boxes") is not None:
                masks, scores, logits = self.model.predict_inst(
                    state,
                    point_coords=prompts.get("points"),
                    point_labels=prompts.get("labels"),
                    box=prompts.get("boxes"),
                    multimask_output=True
                )
                
                # Filter results by confidence threshold
                results = [masks[i] for i, score in enumerate(scores) if score >= confidence_threshold]
                
                # If everything filtered out but we asked for something, maybe return best match anyway?
                # For now, stick to strict threshold.
        
        return results

    def predict_video(self, video_path: str, output_path: str, prompts: dict = None, confidence_threshold: float = 0.5):
        """
        Track and segment objects in a video using NSP Visual Analysis System.
        """
        print(f"Processing video: {video_path}")
        
        if not HAS_LIBRARY:
            print("Running in mock mode for video.")
            return 0

        # NATIVE VIDEO PREDICTOR PATH (Faster)
        if self.video_predictor is not None:
            print("Using native SAM3 video predictor for optimized tracking...")
            return self._predict_video_native(video_path, output_path, prompts, confidence_threshold)

        # FALLBACK: FRAME-BY-FRAME PATH (Slower)
        print("Falling back to frame-by-frame processing...")
        return self._predict_video_frame_by_frame(video_path, output_path, prompts, confidence_threshold)

    def _predict_video_native(self, video_path: str, output_path: str, prompts: dict, confidence_threshold: float):
        """Optimized video tracking using SAM3's native session-based predictor."""
        import uuid
        session_id = str(uuid.uuid4())
        
        try:
            # 1. Start session for the video
            print(f"Starting tracking session: {session_id}")
            self.video_predictor.start_session(video_path, session_id=session_id)
            
            # 2. Add prompt to the first frame (frame_idx=0)
            if prompts and prompts.get("text"):
                print(f"Adding text prompt to session: '{prompts['text']}'")
                self.video_predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    text=prompts["text"]
                )
            elif prompts and (prompts.get("points") is not None or prompts.get("boxes") is not None):
                print("Adding interactive prompt to session...")
                self.video_predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    points=prompts.get("points"),
                    point_labels=prompts.get("labels"),
                    bounding_boxes=prompts.get("boxes")
                )
            else:
                print("No prompts provided for tracking.")
                return 0

            # 3. Propagate through the video
            fps, width, height, total_frames = VideoHandler.get_video_properties(video_path)
            writer = VideoHandler.create_video_writer(output_path, fps, width, height)
            cap = cv2.VideoCapture(video_path)
            
            print(f"Propagating prompts through {total_frames} frames...")
            
            # Use 'forward' direction starting from frame 0
            # propagate_in_video likely returns a generator of results
            results_generator = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=total_frames
            )

            for out_frame_idx, out_obj_ids, out_mask_logits in results_generator:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # out_mask_logits typically contains masks for all objects in the frame
                masks = []
                for i in range(len(out_obj_ids)):
                    # Thresholding logits (usually > 0 in SAM)
                    mask = (out_mask_logits[i] > 0).cpu().numpy()
                    if mask.ndim == 3: # If [1, H, W]
                        mask = mask[0]
                    masks.append(mask)
                
                # Draw results
                labels = [f"ID:{oid+1}" for oid in out_obj_ids]
                annotated_frame = draw_results(frame_rgb, masks, labels=labels)
                
                # Add frame info
                cv2.putText(annotated_frame, f"Frame: {out_frame_idx}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Final conversion and write
                frame_bgr = cv2.cvtColor(annotated_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                
                if out_frame_idx % 30 == 0:
                    print(f"Tracked frame {out_frame_idx}/{total_frames}...")

            cap.release()
            writer.release()
            return 1 # Simplified return as we are tracking a prompt
            
        except Exception as e:
            print(f"Error during native video tracking: {e}")
            traceback.print_exc()
            return 0
        finally:
            print(f"Closing tracking session: {session_id}")
            self.video_predictor.close_session(session_id)

    def _predict_video_frame_by_frame(self, video_path: str, output_path: str, prompts: dict, confidence_threshold: float):
        """Fallback frame-by-frame processing."""
        fps, width, height, total_frames = VideoHandler.get_video_properties(video_path)
        cap = cv2.VideoCapture(video_path)
        writer = VideoHandler.create_video_writer(output_path, fps, width, height)
        
        object_count = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = self.predict_image(frame_rgb, prompts, confidence_threshold=confidence_threshold)
            object_count = max(object_count, len(masks))
            
            if len(masks) > 0:
                labels = [f"ID:{i+1}" for i in range(len(masks))]
                annotated_frame = draw_results(frame_rgb, masks, labels=labels)
                cv2.putText(annotated_frame, f"Objects: {len(masks)}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                annotated_frame = frame_rgb
                
            frame_bgr = cv2.cvtColor(annotated_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        writer.release()
        return object_count
