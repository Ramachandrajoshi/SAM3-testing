import json
import zipfile
import tempfile
from pathlib import Path
import gradio as gr
import cv2
import numpy as np
from core.analyzer import NSPVisualAnalysisSystemAnalyzer
from models.nsp_wrapper import NSPVisualAnalysisSystemWrapper
import os

# Initialize wrapper, load weights, then pass to analyzer
_model_load_error = None
try:
    _wrapper = NSPVisualAnalysisSystemWrapper()
    _wrapper.load_model("data/checkpoints/nsp.pt")
    analyzer = NSPVisualAnalysisSystemAnalyzer(model=_wrapper)
except Exception as e:
    _model_load_error = str(e)
    analyzer = None
    print(f"CRITICAL ERROR: Model failed to load — {e}")

# Configuration for real-time pipeline (6GB VRAM constraint)
INPUT_SIZE = 256  # Downscale to reduce VRAM usage
SKIP_FRAMES = 2   # Process every 2 frames to manage latency/VRAM


def _mask_from_editor_layers(editor_data: dict) -> np.ndarray:
    """Combine all painted ImageEditor layers into a single boolean mask."""
    layers = (editor_data or {}).get("layers", [])
    combined = None
    for layer in layers:
        if layer is None:
            continue
        # Each layer is an RGBA ndarray; alpha channel shows painted pixels
        alpha = layer[:, :, 3] if (layer.ndim == 3 and layer.shape[2] == 4) else layer
        m = alpha > 10          # threshold out noise
        combined = m if combined is None else (combined | m)
    return combined             # None if nothing was drawn


def _prompts_from_mask(mask: np.ndarray) -> dict:
    """Convert a drawn mask into a bounding-box + centroid-point SAM3 prompt."""
    if mask is None or not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    # Small padding
    h, w   = mask.shape
    pad    = max(5, int(min(h, w) * 0.01))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    # Centroid of the painted strokes
    ys, xs = np.where(mask)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    return {
        "boxes":  np.array([x1, y1, x2, y2], dtype=float),
        "points": np.array([[cx, cy]], dtype=float),
        "labels": np.array([1], dtype=int),
    }


def process_drawn_image(editor_data, text_prompt):
    """Segment the region the user painted on, or use a text prompt."""
    if editor_data is None:
        return None, 0, "Upload an image first."
    if analyzer is None:
        return None, 0, f"Model not loaded: {_model_load_error}"

    bg = editor_data.get("background")
    if bg is None:
        return None, 0, "No background image in editor."

    # ImageEditor background is RGBA — convert to RGB
    if bg.ndim == 3 and bg.shape[2] == 4:
        image_np = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)
    else:
        image_np = bg.copy()

    # Priority: text > drawn area > auto-segment
    prompts     = None
    prompt_info = "none (auto-segment)"
    if text_prompt and text_prompt.strip():
        prompts     = {"text": text_prompt.strip()}
        prompt_info = f"text: '{text_prompt.strip()}'"
    else:
        mask    = _mask_from_editor_layers(editor_data)
        prompts = _prompts_from_mask(mask)
        if prompts is not None:
            bb          = prompts["boxes"].astype(int)
            prompt_info = f"drawn area → box [{bb[0]},{bb[1]},{bb[2]},{bb[3]}]"
        # else: prompts stays None → auto-segment

    try:
        result, count = analyzer.analyze_image(
            image_np, prompts, input_size=INPUT_SIZE, skip_frames=SKIP_FRAMES
        )
    except Exception as e:
        return image_np.copy(), 0, f"Analysis error: {e}"

    return result, count, f"Found {count} object(s) · {prompt_info}"


def _parse_box(box_text: str):
    """Parse 'x1,y1,x2,y2' into [x1,y1,x2,y2] array."""
    if not box_text or not box_text.strip():
        return None
    parts = box_text.strip().split(",")
    if len(parts) == 4:
        try:
            return np.array([float(p.strip()) for p in parts])
        except ValueError:
            pass
    return None


def process_image(original_img, text_prompt, points_list, box_text):
    """Run segmentation on original_img using whichever prompt is provided."""
    if original_img is None:
        return None, 0, "No image provided."
    if analyzer is None:
        return None, 0, f"Model not loaded: {_model_load_error}"

    image_np = np.array(original_img)

    # Priority: text > click-points > box
    prompts = None
    prompt_info = "none (auto-segment)"
    if text_prompt and text_prompt.strip():
        prompts = {"text": text_prompt.strip()}
        prompt_info = f"text: '{text_prompt.strip()}'"
    elif points_list:
        coords = np.array(points_list, dtype=float)
        prompts = {
            "points": coords,
            "labels": np.ones(len(points_list), dtype=int),
        }
        prompt_info = f"{len(points_list)} click point(s)"
        box = _parse_box(box_text)
        if box is not None:
            prompts["boxes"] = box
            prompt_info += " + box"
    else:
        box = _parse_box(box_text)
        if box is not None:
            prompts = {"boxes": box}
            prompt_info = f"box: {box_text}"

    try:
        result, count = analyzer.analyze_image(
            image_np, prompts, input_size=INPUT_SIZE, skip_frames=SKIP_FRAMES
        )
    except Exception as e:
        return original_img.copy(), 0, f"Analysis error: {e}"

    return result, count, f"Found {count} object(s) · prompt: {prompt_info}"


def process_video(video_path):
    """Gradio interface function for video analysis."""
    if video_path is None:
        return None, 0
    if analyzer is None:
        return None, 0

    output_path = "output_processed_video.mp4"
    try:
        count = analyzer.process_video_file(video_path, output_path)
    except Exception as e:
        print(f"Video processing error: {e}")
        return None, 0

    return output_path, count


def process_camera_frame(frame, frame_count):
    """Process a single frame from the webcam stream."""
    if frame is None or analyzer is None:
        return frame, 0, frame_count

    # Skip frames to manage VRAM/latency
    if frame_count % SKIP_FRAMES != 0:
        return frame, 0, frame_count + 1

    image_np = np.array(frame)
    try:
        result, count = analyzer.analyze_image(
            image_np, prompts=None, input_size=INPUT_SIZE, skip_frames=SKIP_FRAMES
        )
    except Exception as e:
        print(f"Camera frame error: {e}")
        return frame, 0, frame_count + 1

    return result, count, frame_count + 1


# ─────────────────────────────────────────────────────────────────────────────
# Segment & Label helpers
# ─────────────────────────────────────────────────────────────────────────────

# Fixed palette (RGB) – up to 30 segments
_SEG_PALETTE = np.array([
    [230,  25,  75], [ 60, 180,  75], [255, 225,  25], [  0, 130, 200],
    [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
    [210, 245,  60], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [170, 110,  40], [255, 250, 200], [128,   0,   0], [170, 255, 195],
    [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128],
    [255,  80,  80], [ 80, 255,  80], [ 80,  80, 255], [255, 200,  80],
    [200,  80, 255], [ 80, 200, 200], [160, 120,  80], [120, 160,  80],
    [ 80, 120, 160], [200, 120, 120],
], dtype=np.uint8)


def _draw_segments_overlay(
        image: np.ndarray,
        masks: list,
        selected_idx: int = -1,
        labels: dict = None,   # {mask_idx: str}
        alpha_normal: float = 0.40,
        alpha_selected: float = 0.70,
) -> np.ndarray:
    """Draw all mask overlays on image; highlight selected_idx."""
    overlay = image.copy().astype(np.float32)
    labels  = labels or {}
    h, w    = image.shape[:2]

    for i, mask in enumerate(masks):
        if mask is None or not mask.any():
            continue
        color = _SEG_PALETTE[i % len(_SEG_PALETTE)].astype(np.float32)
        alpha = alpha_selected if i == selected_idx else alpha_normal
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + alpha * color[c],
                overlay[:, :, c],
            )
        # Contour
        mask_u8  = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        thickness = 3 if i == selected_idx else 1
        border_col = (255, 255, 255) if i == selected_idx else tuple(color.tolist())
        cv2.drawContours(overlay, contours, -1,
                         (int(border_col[0]), int(border_col[1]), int(border_col[2])),
                         thickness)
        # Index label at centroid
        ys, xs = np.where(mask)
        cx, cy  = int(np.median(xs)), int(np.median(ys))
        txt     = f"{i+1}"
        if i in labels:
            txt += f":{labels[i]}"
        cv2.putText(overlay, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _crop_masked_object(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a tight crop of the object on a white background."""
    masked = np.full_like(image, 255)   # white background
    masked[mask] = image[mask]
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return masked
    pad = 10
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(image.shape[0], int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(image.shape[1], int(xs.max()) + pad + 1)
    return masked[y1:y2, x1:x2]


def segment_all_objects(image_np, grid_n, max_size_pct):
    """Run auto-segmentation and return overlay image + initial state tuple."""
    if image_np is None:
        return None, None, [], {}, "Upload an image first."
    if _wrapper is None:
        return None, None, [], {}, f"Model not loaded: {_model_load_error}"
    try:
        masks = _wrapper.predict_all(
            image_np,
            grid_n=int(grid_n),
            max_area_ratio=float(max_size_pct) / 100.0,
        )
    except Exception as e:
        return None, None, [], {}, f"Segmentation error: {e}"
    if not masks:
        return image_np.copy(), image_np.copy(), [], {}, \
            "No segments found. Try lowering Max Segment Size or increasing Grid Density."
    overlay = _draw_segments_overlay(image_np, masks)
    status  = (
        f"{len(masks)} segment(s) found. "
        "Click a coloured region to select it, then add a label below."
    )
    return overlay, image_np.copy(), masks, {}, status


def select_segment(orig_img, masks, labels_dict, evt: gr.SelectData):
    """Find which mask the user clicked and update the overlay + preview."""
    if orig_img is None or not masks:
        return None, None, -1, "No segments loaded."
    x, y    = evt.index          # col, row
    sel_idx = -1
    for i, mask in enumerate(masks):
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            sel_idx = i
            break
    if sel_idx == -1:
        # Clicked on background – find nearest mask centroid
        min_d = float("inf")
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            d = ((np.median(xs) - x) ** 2 + (np.median(ys) - y) ** 2) ** 0.5
            if d < min_d:
                min_d, sel_idx = d, i
    overlay = _draw_segments_overlay(orig_img, masks, selected_idx=sel_idx,
                                     labels=labels_dict)
    preview = _crop_masked_object(orig_img, masks[sel_idx]) if sel_idx >= 0 else orig_img
    existing_label = labels_dict.get(sel_idx, "")
    status  = (
        f"Selected segment {sel_idx + 1}"
        + (f" (labelled: '{existing_label}'" + ")" if existing_label else
           " — type a label and click Add.")
    )
    return overlay, preview, sel_idx, status


def add_label(orig_img, masks, labels_dict, sel_idx, label_text):
    """Attach a label to the selected mask and refresh the overlay."""
    if sel_idx < 0 or not masks:
        return None, labels_dict, _labels_to_df(labels_dict), "No segment selected."
    labels_dict          = dict(labels_dict)   # copy so Gradio detects change
    labels_dict[sel_idx] = label_text.strip()
    overlay  = _draw_segments_overlay(orig_img, masks, selected_idx=sel_idx,
                                       labels=labels_dict)
    n_labelled = sum(1 for v in labels_dict.values() if v)
    return overlay, labels_dict, _labels_to_df(labels_dict), \
        f"{n_labelled}/{len(masks)} segments labelled."


def _labels_to_df(labels_dict: dict) -> list:
    """Convert {idx: label} to list-of-lists for gr.Dataframe."""
    rows = []
    for idx in sorted(labels_dict.keys()):
        rows.append([idx + 1, labels_dict[idx]])
    return rows if rows else [["—", "—"]]


def export_annotations(orig_img, masks, labels_dict):
    """Save annotated masks + JSON to a zip and return the path."""
    if orig_img is None or not masks:
        return None, "Nothing to export."
    tmp_dir  = Path(tempfile.mkdtemp())
    masks_dir = tmp_dir / "masks"
    masks_dir.mkdir()
    records   = []
    for i, mask in enumerate(masks):
        mask_u8  = (mask.astype(np.uint8) * 255)
        fname    = f"mask_{i:03d}.png"
        cv2.imwrite(str(masks_dir / fname), mask_u8)
        ys, xs   = np.where(mask)
        bbox     = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    if len(ys) else [])
        records.append({
            "id":         i + 1,
            "label":      labels_dict.get(i, ""),
            "mask_file":  f"masks/{fname}",
            "bbox_xyxy":  bbox,
            "area_px":    int(mask.sum()),
        })
    # Save original image
    orig_bgr = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(tmp_dir / "original.jpg"), orig_bgr)
    # Save JSON
    json_path = tmp_dir / "annotations.json"
    json_path.write_text(
        json.dumps({"image": "original.jpg", "segments": records}, indent=2)
    )
    # Zip everything
    zip_path = tmp_dir / "annotations.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in tmp_dir.rglob("*"):
            if f != zip_path and f.is_file():
                zf.write(f, f.relative_to(tmp_dir))
    n_labelled = sum(1 for r in records if r["label"])
    return str(zip_path), (
        f"Exported {len(masks)} masks ({n_labelled} labelled) → annotations.zip"
    )


with gr.Blocks(title="NSP Visual Analysis System - Real-Time Pipeline") as demo:
    gr.Markdown("# NSP Visual Analysis System - Real-Time Object Analysis")
    gr.Markdown(
        "Analyze and annotate objects in images, videos, and live webcam streams "
        "using SAM3 with 6 GB VRAM optimizations."
    )

    if _model_load_error:
        gr.Markdown(
            f"**ERROR: Model failed to load — `{_model_load_error}`. "
            "Verify `data/checkpoints/nsp.pt` exists.**"
        )

    with gr.Tabs():

        # ── Image Analysis & Annotation Tab ─────────────────────────────────
        with gr.TabItem("Image Analysis & Annotation"):
            gr.Markdown(
                "**How to use**\n"
                "1. Upload an image.\n"
                "2. Select the **brush tool** (pencil icon) and **paint over the region** "
                "you want segmented — any colour is fine.\n"
                "3. Press **Analyze & Annotate**. The model segments the precise object "
                "inside your painted area.\n"
                "4. Use the **eraser** or the ↩ undo button to fix strokes before analysing.\n\n"
                "Optionally type a **text prompt** instead — it overrides the painted area."
            )
            with gr.Row():
                # ── Left column: drawing canvas + controls ──
                with gr.Column(scale=1):
                    img_editor = gr.ImageEditor(
                        label="Upload · then paint the region to segment",
                        type="numpy",
                        brush=gr.Brush(colors=["#FF4444", "#44FF44", "#4488FF"],
                                       default_size=18),
                        eraser=gr.Eraser(default_size=20),
                        layers=False,
                    )
                    text_input = gr.Textbox(
                        label="Text Prompt (optional — overrides drawing)",
                        placeholder="e.g. red car, blue chair",
                    )
                    img_button = gr.Button("Analyze & Annotate", variant="primary")

                # ── Right column: results ──
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Annotated Result")
                    img_count  = gr.Number(label="Object Count", precision=0)
                    img_status = gr.Textbox(label="Status", interactive=False)

            img_button.click(
                process_drawn_image,
                inputs=[img_editor, text_input],
                outputs=[img_output, img_count, img_status],
            )

        # ── Segment & Label Tab ──────────────────────────────────────────────
        with gr.TabItem("Segment & Label"):
            gr.Markdown(
                "**Workflow**\n"
                "1. Upload an image and press **Segment All** — every object is found automatically.\n"
                "2. **Click any coloured region** to select that segment (it brightens).\n"
                "3. Type a label and press **Add Label**.\n"
                "4. Repeat for as many segments as needed.\n"
                "5. Press **Export** to download a ZIP with all masks (PNG) + `annotations.json`."
            )

            # ── State ──
            sl_orig_state    = gr.State(None)
            sl_masks_state   = gr.State([])
            sl_labels_state  = gr.State({})
            sl_selidx_state  = gr.State(-1)

            with gr.Row():
                # Left: upload + controls
                with gr.Column(scale=1):
                    sl_upload = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        interactive=True,
                    )
                    with gr.Accordion("Segmentation Settings", open=False):
                        sl_grid_slider = gr.Slider(
                            minimum=4, maximum=20, value=12, step=1,
                            label="Grid Density (higher = more segments detected)",
                        )
                        sl_maxsize_slider = gr.Slider(
                            minimum=5, maximum=60, value=25, step=1,
                            label="Max Segment Size % of image (lower = ignore background)",
                        )
                    sl_seg_btn = gr.Button("Segment All", variant="primary")
                    sl_status  = gr.Textbox(
                        label="Status", interactive=False,
                        value="Upload an image, then press Segment All."
                    )

                # Middle: segmented overlay (click to select)
                with gr.Column(scale=1):
                    sl_overlay = gr.Image(
                        label="Segmented — click to select",
                        type="numpy",
                        interactive=True,
                    )

                # Right: selected object preview + labelling
                with gr.Column(scale=1):
                    sl_preview = gr.Image(
                        label="Selected Segment Preview",
                        type="numpy",
                        interactive=False,
                    )
                    sl_label_input = gr.Textbox(
                        label="Label for selected segment",
                        placeholder="e.g. car, person, tree …",
                    )
                    with gr.Row():
                        sl_add_btn    = gr.Button("Add Label", variant="primary")
                        sl_export_btn = gr.Button("Export")
                    sl_annotations_df = gr.Dataframe(
                        headers=["#", "Label"],
                        label="Annotations",
                        column_count=2,
                        interactive=False,
                    )
                    sl_download = gr.File(label="Download Annotations ZIP")

            # ── Event wiring ──

            sl_seg_btn.click(
                segment_all_objects,
                inputs=[sl_upload, sl_grid_slider, sl_maxsize_slider],
                outputs=[sl_overlay, sl_orig_state, sl_masks_state,
                         sl_labels_state, sl_status],
            )

            # Click on overlay → select segment
            sl_overlay.select(
                select_segment,
                inputs=[sl_orig_state, sl_masks_state, sl_labels_state],
                outputs=[sl_overlay, sl_preview, sl_selidx_state, sl_status],
            )

            # Add label
            sl_add_btn.click(
                add_label,
                inputs=[sl_orig_state, sl_masks_state, sl_labels_state,
                        sl_selidx_state, sl_label_input],
                outputs=[sl_overlay, sl_labels_state,
                         sl_annotations_df, sl_status],
            )

            # Export
            sl_export_btn.click(
                export_annotations,
                inputs=[sl_orig_state, sl_masks_state, sl_labels_state],
                outputs=[sl_download, sl_status],
            )
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    video_button = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Processed Video")
                    video_track_total = gr.Number(label="Total Objects Tracked", precision=0)

            video_button.click(
                process_video,
                inputs=video_input,
                outputs=[video_output, video_track_total]
            )

        # ── Live Camera Tab ──────────────────────────────────────────────────
        with gr.TabItem("Live Camera - Real-Time Stream"):
            gr.Markdown(
                "Real-time segmentation from your webcam. "
                f"Frames are resized to {INPUT_SIZE}×{INPUT_SIZE} and every "
                f"{SKIP_FRAMES} frames are skipped to reduce VRAM pressure."
            )
            cam_input = gr.Image(
                sources="webcam",
                streaming=True,
                type="numpy",
                label="Live Feed"
            )
            with gr.Row():
                cam_output = gr.Image(label="Segmented Feed")
                cam_count = gr.Number(label="Objects in Frame", precision=0)

            frame_count_state = gr.State(0)
            cam_input.stream(
                process_camera_frame,
                inputs=[cam_input, frame_count_state],
                outputs=[cam_output, cam_count, frame_count_state]
            )

    # ── Technical Specifications ─────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### Technical Specifications")
    gr.Markdown(
        "**Real-Time Pipeline** | Frame resize to 256×256 · Frame skipping (every 2 frames) "
        "· GPU-optimised SAM3 processing · Live webcam streaming\n\n"
        "**VRAM Optimisation** | Input resizing reduces footprint ~70% · "
        "Frame skipping prevents memory accumulation · Quantisation-ready (INT8 pending)"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
