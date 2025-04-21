import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import cv2
from copy import deepcopy
from src import bbox_utils
_model = None


def get_model():
    global _model
    if _model is None:
        mp_hands = mp.solutions.hands
        _model = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                min_detection_confidence=0.4, min_tracking_confidence=0.3)
    return _model


def get_hand_bbox(image, image_bbox, vis=False):
    """Return hand bounding box. 

    Args:
        image (PIL image): The input PIL image.

    Returns:
        ndarray: hand bounding box
    """

    hands = get_model()
    image = deepcopy(image)
    results_mediapipe = hands.process(image)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    # landmarks are the 21 points on the hand
    mean_distance = np.inf
    if results_mediapipe.multi_hand_landmarks:
        for hand_landmarks in results_mediapipe.multi_hand_landmarks:
            # Get bounding box coordinates
            x_coords = [lm.x * image.shape[1]
                        for lm in hand_landmarks.landmark]
            y_coords = [lm.y * image.shape[0]
                        for lm in hand_landmarks.landmark]
            distance = bbox_utils.distance_to_bbox(x_coords, y_coords, image_bbox).mean()
            # the final hand_bbox is the one with the smallest distance to the image_bbox
            if distance < mean_distance:
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                mean_distance = distance
            if vis:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Landmark color
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)   # Connection color
                )
    
        if vis:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("1. Hand Detection (MediaPipe)", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # vis_image = Image.fromarray(image)
            # draw = ImageDraw.Draw(vis_image)
            # box = [x_min, y_min, x_max, y_max]
            # draw.rectangle(box, outline="red", width=3)
            # vis_image.show()
        return np.asarray([x_min, y_min, x_max, y_max])
    else:
        print("No hands detected.")
        return None

    
