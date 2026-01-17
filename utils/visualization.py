import numpy as np
import cv2

def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply a semi-transparent mask to an image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def draw_results(image: np.ndarray, masks: list, boxes: list = None, labels: list = None):
    """
    Draw segmentation masks and bounding boxes on the image.
    """
    annotated_image = image.copy()
    
    for i, mask in enumerate(masks):
        # Generate a random color for each mask
        color = np.random.random(3)
        annotated_image = apply_mask(annotated_image, mask, color)
        
        if boxes is not None and i < len(boxes):
            box = boxes[i]
            cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), (0, 255, 0), 2)
            
        if labels is not None and i < len(labels):
            label = labels[i]
            cv2.putText(annotated_image, str(label), (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return annotated_image
