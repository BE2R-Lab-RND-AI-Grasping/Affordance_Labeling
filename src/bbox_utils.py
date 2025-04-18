import cv2
import numpy as np

def add_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """Visualize the bounding box on the image.

    Args:
        image (np.ndarray): The input image.
        bbox (list): The bounding box coordinates [x_min, y_min, x_max, y_max].
        color (tuple): Color of the bounding box in BGR format.
        thickness (int): Thickness of the bounding box lines.
    """
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)


def crop_bbox_cv2(image, bbox)->np.ndarray: 
    """
    Crop objects from the image based on bounding boxes.
    Args:
    - image: The input numpy array.
    - boxes: Bounding boxes (tensor) in the format [x_min, y_min, x_max, y_max].

    Returns:
    - list of cropped object regions.
    """

    x_min, y_min, x_max, y_max = bbox
    cropped_object = image[y_min:y_max+1, x_min:x_max+1]
    return cropped_object

def add_bbox_to_image(image:np.ndarray, bbox, color=np.array([200,200,200])):
    # Create a copy of the image to avoid modifying the original
    painted_image = image.copy()
    # Get the bounding box coordinates
    x1, y1, x2, y2 = bbox
    painted_image[y1:y2+1, x1:x2+1] = color
    # Paint the area inside the bounding box with the specified color

    return painted_image

def get_total_bounding_box(image:np.ndarray, background_white=False, vis=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert threshold: detect gray object on white background
    # Threshold the image to separate foreground from background
    if background_white: 
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine all contours into one large set of points
    all_points = np.concatenate(contours)  # Concatenate all points of the contours

    x, y, w, h = cv2.boundingRect(all_points)
    bbox = [x, y, x+w, y+h]  # [x_min, y_min, x_max, y_max]
    # Draw bounding boxes around the detected objects
    if vis:
        vis_image = image.copy()
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detected Objects", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() # Convert points to integer

    return bbox
