# NSP Visual Analysis System

This project provides a robust framework for object segmentation in images and videos using the NSP Visual Analysis System (based on Segment Anything Model 3).

## Features
- **Image Analysis**: Segment objects in static images with text or point prompts.
- **Video Analysis**: Process video files with object tracking and segmentation.
- **Live Camera Feed**: Real-time segmentation from webcam input.
- **Clean Architecture**: Follows SOLID and DRY principles for easy maintenance and fine-tuning.
- **Gradio UI**: User-friendly web interface.

## Project Structure
- `app.py`: Gradio frontend.
- `core/`: High-level analysis logic (NSPVisualAnalysisSystemAnalyzer).
- `models/`: Model wrappers and base classes.
- `utils/`: Video processing and visualization utilities.
- `data/checkpoints/`: Directory for model weights.

## Setup Instructions

### 1. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Required Library
Follow the instructions for installing the underlying segmentation engine.
Typically:
```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

### 4. Download Model Weights
Place the model checkpoints in `data/checkpoints/` as `nsp.pt`.

### 5. Run the Application
```bash
python app.py
```

## Maintenance & Fine-tuning
- To use a different model, implement the `BaseSegmentationModel` interface in `models/`.
- Fine-tuning logic can be added to the `models/` directory following the architecture.
- Visualization styles can be customized in `utils/visualization.py`.
