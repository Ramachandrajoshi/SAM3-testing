import numpy as np
from models.sam3_wrapper import SAM3Wrapper
import torch

def test_sam3():
    print("Initializing SAM3Wrapper...")
    wrapper = SAM3Wrapper()
    
    checkpoint_path = "data/checkpoints/sam3.pt"
    print(f"Loading model from {checkpoint_path}...")
    wrapper.load_model(checkpoint_path)
    
    # Create a dummy image
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[100:300, 100:300, :] = 255 # A white square
    
    print("Testing text prompt prediction...")
    prompts = {"text": "a white square"}
    results = wrapper.predict_image(image, prompts)
    print(f"Number of masks found (text): {len(results)}")

    print("Testing box prompt prediction...")
    # Box format: [x0, y0, x1, y1] for SAM2Predictor usually
    # But wait, SAM3InteractiveImagePredictor might expect something else?
    # Let's try [100, 100, 300, 300]
    prompts = {"boxes": np.array([100, 100, 300, 300])}
    results = wrapper.predict_image(image, prompts)
    
    print(f"Number of masks found (box): {len(results)}")
    if len(results) > 0:
        print(f"Mask shape: {results[0].shape}")
        print("Success!")
    else:
        print("No masks found, but no crash either.")

if __name__ == "__main__":
    test_sam3()
