import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoProcessor

from .sam2.sam2_image_predictor import SAM2ImagePredictor

FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "./grounded_sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam2(
        config_file,
        ckpt_path=None,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
        **kwargs,
):
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


def open_vocab_detection(
        image_path,
        labels_list,
        output_dir="./outputs",
) -> Tuple[sv.Detections, dict[str, Path]]:
    """
    Perform open vocabulary detection on an image with a list of labels.

    Args:
        image_path (str or Path): Path to the input image
        labels_list (list): List of labels to detect in the image
        output_dir (str or Path): Directory to save output images
        florence2_model: Pre-loaded Florence-2 model
        florence2_processor: Pre-loaded Florence-2 processor
        sam2_predictor: Pre-loaded SAM2 predictor

    Returns:
        tuple: (detections, output_paths)
            - detections: Supervision Detections object containing bounding boxes, masks, and class IDs
            - output_paths: Dictionary containing paths to output images
    """
    # Convert paths to Path objects
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    # build florence-2
    florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True,
                                                           torch_dtype='auto').eval().to(device)
    florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

    # build sam 2
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare text input by joining labels
    text_input = " <and> ".join(labels_list)

    # Load and process image
    image = Image.open(image_path).convert("RGB")

    # Run Florence-2 detection
    results = run_florence2(
        task_prompt="<OPEN_VOCABULARY_DETECTION>",
        text_input=text_input,
        model=florence2_model,
        processor=florence2_processor,
        image=image
    )

    results = results["<OPEN_VOCABULARY_DETECTION>"]

    # Parse Florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["bboxes_labels"]
    class_ids = np.array(list(range(len(class_names))))

    # No detections case
    if len(input_boxes) == 0:
        return None, {}

    # Predict masks with SAM2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Prepare labels for visualization
    labels = [f"{class_name}" for class_name in class_names]

    # Create Detections object
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    # Visualize results
    img = cv2.imread(str(image_path))  # Convert Path to string for cv2
    box_path = output_dir / "open_vocab_detection_boxes.jpg"
    mask_path = output_dir / "open_vocab_detection_masks.jpg"

    # Draw boxes
    box_annotator = sv.BoxAnnotator()
    box_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    # Add labels
    label_annotator = sv.LabelAnnotator()
    labeled_frame = label_annotator.annotate(scene=box_frame, detections=detections, labels=labels)
    cv2.imwrite(str(box_path), labeled_frame)  # Convert Path to string for cv2

    # Add masks
    mask_annotator = sv.MaskAnnotator()
    mask_frame = mask_annotator.annotate(scene=labeled_frame, detections=detections)
    cv2.imwrite(str(mask_path), mask_frame)  # Convert Path to string for cv2

    output_paths = {
        "boxes": box_path,
        "masks": mask_path
    }

    return detections, output_paths


if __name__ == '__main__':
    result = open_vocab_detection('../captured_images/test1.jpg',
                                  ['blue square', 'black star', 'black triangle', 'cigarette butts'])
