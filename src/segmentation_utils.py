import torch
from PIL import Image
from transformers import SamModel, SamProcessor

_model_sam = None
_processor_sam = None

def get_model_and_preprocess_sam():
    """Model and transform getter. Creates the model once."""

    global _model_sam, _processor_sam
    if _model_sam is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "facebook/sam-vit-huge"
        _processor_sam = SamProcessor.from_pretrained(model_id)
        _model_sam = SamModel.from_pretrained(model_id).to(device)
        _model_sam.eval()

    return _model_sam, _processor_sam

def get_masks_sam(image, bbox):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_model_and_preprocess_sam()
    image = Image.fromarray(image)
    input_boxes = [[bbox]]

    inputs = processor([image], input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores
    return masks[0], scores