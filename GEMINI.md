# GEMINI Context: SAM3 Video & Image Analysis

This document provides essential context for Gemini to understand and interact with this project efficiently.

## Project Overview
This project is a Python-based framework for object segmentation and tracking in images and videos, leveraging Meta's **Segment Anything Model 3 (SAM3)**. It provides a Gradio-powered web interface for interactive analysis.

### Core Technologies
- **Language:** Python 3.12+
- **Deep Learning:** PyTorch, TorchVision
- **Computer Vision:** OpenCV (cv2), Pillow
- **UI Framework:** Gradio
- **Model:** Meta SAM3 (Segment Anything Model 3)

### Architecture
The project follows clean architecture principles (SOLID & DRY):
- **Frontend (`app.py`):** Gradio-based web interface with tabs for Image, Video, and Live Camera analysis.
- **Core Logic (`core/analyzer.py`):** The `SAM3Analyzer` class coordinates between the UI and the model wrapper, handling high-level analysis tasks.
- **Models (`models/`):** 
    - `base_model.py`: Defines the `BaseSegmentationModel` abstract interface.
    - `sam3_wrapper.py`: Implements the SAM3-specific logic.
- **Utilities (`utils/`):** 
    - `video_utils.py`: Handles video reading, writing, and property extraction.
    - `visualization.py`: Contains logic for drawing masks and results on images.
- **Data (`data/checkpoints/`):** Intended storage for SAM3 model weights (e.g., `sam3_h.pth`).

## Building and Running

### Prerequisites
1. **SAM3 Library:** Must be installed from the official Meta repository.
   ```bash
   pip install git+https://github.com/facebookresearch/sam3.git
   ```
2. **Model Weights:** Download SAM3 checkpoints and place them in `data/checkpoints/`.

### Commands
- **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **Run the Application:**
  ```bash
  python app.py
  ```
- **Tests:** (TODO: Implement unit tests for analyzer and wrappers)

## Development Conventions
- **Interfaces:** Always extend `BaseSegmentationModel` when adding support for new models.
- **Typing:** Use type hints for function signatures.
- **Image Format:** Images are primarily handled as NumPy arrays in **RGB** format (Gradio standard), while OpenCV utilities might convert to **BGR** internally for processing.
- **Error Handling:** Check for SAM3 library availability (see `models/sam3_wrapper.py`) to support mock/CPU environments.
- **Single Responsibility:** Keep UI logic in `app.py` and processing logic in `core/` or `models/`.

## Key Files
- `app.py`: Entry point for the Gradio UI.
- `core/analyzer.py`: Main analysis coordinator.
- `models/sam3_wrapper.py`: Integration point for Meta's SAM3.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation for humans.
