"""Functions for object search in an image"""
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image


def detect_object_with_opencv(image, background_white=False, vis=False):
    """OpenCV method for detecting objects from the color.
    This function works with renders obtained from the open3d visualizer. 
    The model is gray and the background is white.

    Args:
        image (ndarray): image loaded with opencv
        vis (bool, optional): visualization. Defaults to False.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate foreground from background
    if background_white: 
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image, the white regions are the objects and the black regions are the background
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found")
        return None
    if len(contours) > 1:
        print("Multiple contours found")
        return None

    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)
    bbox = [x, y, x+w, y+h]  # [x_min, y_min, x_max, y_max]
    # Draw bounding boxes around the detected objects
    if vis:
        vis_image = image.copy()
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detected Objects", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bbox


_model = None
_transform = None


def get_model_and_preprocess():
    """Model and transform getter. Creates the model once."""

    global _model, _transform
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load a pre-trained Mask R-CNN model for object detection
        _model = models.detection.maskrcnn_resnet50_fpn(
            weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT ).to(device)
        _model.eval()
        _transform = transforms.ToTensor()

    return _model, _transform


def get_bounding_boxes(image):
    """Return the bounding boxes of all objects detected by the model

    Args:
        image: The input PIL image.

    Returns:
        list(Tensor): list of bounding boxes
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = get_model_and_preprocess()
    image_tensor = transform(image).unsqueeze(0).to(device)
    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction[0]['boxes'].detach().cpu().numpy().astype(np.int32)


_model_dino = None
_processor_dino = None

def get_model_and_preprocess_dino():
    global _model_dino, _processor_dino
    if _model_dino is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "IDEA-Research/grounding-dino-base"
        _processor_dino = AutoProcessor.from_pretrained(model_id)
        _model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        _model_dino.eval()
    return _model_dino, _processor_dino

def get_bounding_boxes_dino(image, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_model_and_preprocess_dino()
    image = Image.fromarray(image)
    text_labels = [[text]]
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    return boxes, scores
