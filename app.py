import gradio as gr
import cv2
import numpy as np
from core.analyzer import NSPVisualAnalysisSystemAnalyzer
from models.nsp_wrapper import NSPVisualAnalysisSystemWrapper
import os

# Initialize components
model = NSPVisualAnalysisSystemWrapper()
# Load the actual downloaded weights
model.load_model("data/checkpoints/nsp.pt")
analyzer = NSPVisualAnalysisSystemAnalyzer(model)

def process_image(input_img, text_prompt):
    """Gradio interface function for image analysis."""
    if input_img is None:
        return None, 0
    
    # Convert Gradio image (RGB) to numpy array if it isn't already
    image_np = np.array(input_img)
    
    # Simple prompt dictionary (can be expanded)
    prompts = {"text": text_prompt} if text_prompt else None
    
    # Core logic call
    result, count = analyzer.analyze_image(image_np, prompts)
    return result, count

def process_video(video_path):
    """Gradio interface function for video analysis."""
    if video_path is None:
        return None, 0
    
    output_path = "output_processed_video.mp4"
    # Core logic call
    count = analyzer.process_video_file(video_path, output_path)
    
    return output_path, count

# Gradio Blocks UI
with gr.Blocks(title="NSP Visual Analysis System") as demo:
    gr.Markdown("# 🚀 NSP Visual Analysis System - Video & Image Analysis")
    gr.Markdown("Analyze and segment objects in images, video files, or live camera feeds using the NSP Visual Analysis System.")
    
    with gr.Tabs():
        # Image Tab
        with gr.TabItem("🖼️ Image Analysis"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Upload Image", type="numpy")
                    text_input = gr.Textbox(label="Text Prompt (Optional)", placeholder="e.g., 'the red car'")
                    img_button = gr.Button("Analyze Image")
                with gr.Column():
                    img_output = gr.Image(label="Result")
                    img_count = gr.Number(label="Object Count", precision=0)
            
            img_button.click(process_image, inputs=[img_input, text_input], outputs=[img_output, img_count])

        # Video Tab
        with gr.TabItem("📽️ Video Analysis"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    video_button = gr.Button("Process Video")
                with gr.Column():
                    video_output = gr.Video(label="Processed Video")
                    video_count = gr.Number(label="Total Objects Tracked", precision=0)
            
            video_button.click(process_video, inputs=video_input, outputs=[video_output, video_count])

        # Camera Tab (Note: Gradio's webcam input works frame-by-frame)
        with gr.TabItem("📸 Live Camera"):
            gr.Markdown("Note: Live processing may depend on your GPU capability.")
            with gr.Row():
                with gr.Column():
                    cam_input = gr.Image(sources="webcam", streaming=True, label="Live Feed")
                with gr.Column():
                    cam_output = gr.Image(label="Segmented Feed")
                    cam_count = gr.Number(label="Objects in Frame", precision=0)
            
            cam_input.stream(process_image, inputs=[cam_input, gr.State(None)], outputs=[cam_output, cam_count])

    gr.Markdown("### Instructions")
    gr.Markdown("""
    1. **Images**: Upload an image and optionally provide a text prompt to segment specific objects.
    2. **Videos**: Upload a video file. The model will track and segment objects throughout the sequence.
    3. **Camera**: Use your webcam for real-time segmentation.
    
    *Developed with SOLID + DRY principles for easy maintenance and fine-tuning.*
    """)

if __name__ == "__main__":
    # Launch Gradio app
    demo.launch(share=False)
