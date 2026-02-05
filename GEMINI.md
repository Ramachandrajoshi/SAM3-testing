# GEMINI Context: NSP Visual Analysis System

This document provides essential context for Gemini to understand and interact with this project efficiently.

## Project Overview
This project is a Python-based framework for object segmentation and tracking in images and videos, leveraging the **NSP Visual Analysis System** (based on Segment Anything Model 3). It provides a Gradio-powered web interface for interactive analysis.

### Core Technologies
- **Language:** Python 3.12+
- **Deep Learning:** PyTorch, TorchVision
- **Computer Vision:** OpenCV (cv2), Pillow
- **UI Framework:** Gradio
- **Model:** NSP Visual Analysis System

### Architecture
The project follows clean architecture principles (SOLID & DRY):
- **Frontend (`app.py`):** Gradio-based web interface with tabs for Image, Video, and Live Camera analysis.
- **Core Logic (`core/analyzer.py`):** The `NSPVisualAnalysisSystemAnalyzer` class coordinates between the UI and the model wrapper, handling high-level analysis tasks.
- **Models (`models/`):** 
    - `base_model.py`: Defines the `BaseSegmentationModel` abstract interface.
    - `nsp_wrapper.py`: Implements the NSP Visual Analysis System-specific logic.
- **Utilities (`utils/`):** 
    - `video_utils.py`: Handles video reading, writing, and property extraction.
    - `visualization.py`: Contains logic for drawing masks and results on images.
- **Data (`data/checkpoints/`):** Intended storage for model weights (e.g., `nsp.pt`).

## Building and Running

### Prerequisites
1. **Required Library:** Must be installed from the official repository.
   ```bash
   pip install git+https://github.com/facebookresearch/sam3.git
   ```
2. **Model Weights:** Download checkpoints and place them in `data/checkpoints/` as `nsp.pt`.

### Commands
- **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **Run the Application:**
  ```bash
  python app.py
  ```
- **Tests:** Run `python test_nsp_setup.py` to verify installation.

## Development Conventions
- **Interfaces:** Always extend `BaseSegmentationModel` when adding support for new models.
- **Typing:** Use type hints for function signatures.
- **Image Format:** Images are primarily handled as NumPy arrays in **RGB** format (Gradio standard), while OpenCV utilities might convert to **BGR** internally for processing.
- **Error Handling:** Check for library availability (see `models/nsp_wrapper.py`) to support mock/CPU environments.
- **Single Responsibility:** Keep UI logic in `app.py` and processing logic in `core/` or `models/`.

## Key Files
- `app.py`: Entry point for the Gradio UI.
- `core/analyzer.py`: Main analysis coordinator.
- `models/nsp_wrapper.py`: Integration point for the NSP Visual Analysis System.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation for humans.
