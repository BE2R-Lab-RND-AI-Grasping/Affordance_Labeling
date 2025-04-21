import importlib
import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from src.object_detection import get_bounding_boxes_dino
from src import bbox_utils
from src import hand_search
from src.segmentation_utils import get_masks_sam


def process_scene_image(image_path: Path, object_class: str):
    scene_image = cv2.imread(image_path)
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)
    bboxes, scores = get_bounding_boxes_dino(scene_image, object_class)
    if len(bboxes) == 0:
        print(f"No bounding boxes found for {object_class} in {image_path}.")
        return None, None
    # Select the best bounding box based on the score
    best_bbox = bboxes[np.argmax(np.asarray(scores))]
    hand_bbox = hand_search.get_hand_bbox(scene_image, best_bbox, vis=True)
    if hand_bbox is None:
        print(f"No hand bounding box found for {object_class} in {image_path}.")
        return None, None
    hand_bbox = np.round(hand_bbox).astype(int)
    # Get the mask using SAM
    masks, scores = get_masks_sam(scene_image, list(best_bbox))
    mask = masks[0][scores.argmax()]
    rgb_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb_image[mask == 1] = [128, 128, 128]
    return rgb_image, hand_bbox

def process_scenes(scene_dir:Path, object_class):
    scene_file_list = list(scene_dir.glob("*.jpg"))
    for file in scene_file_list:
        image, hand_bbox = process_scene_image(file, object_class)
        if image is None:
            print(f"Skipping {file} due to no bounding boxes found.")
            continue
        # Save the processed image and hand bounding box
        np.savez(file.with_suffix(".npz"),image=image, hand_bbox=hand_bbox)


