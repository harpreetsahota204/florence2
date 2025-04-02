"""
Florence-2 model wrapper for the FiftyOne Model Zoo.
"""
import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image

import fiftyone.core.models as fom
from fiftyone.core.labels import Detection, Detections, Polyline, Polylines
from transformers import AutoModelForCausalLM, AutoProcessor

# Define operation configurations
FLORENCE2_OPERATIONS = {
    "caption": {
        "params": {"detail_level": ["basic", "detailed", "more_detailed"]},
        "task_mapping": {
            "detailed": "<DETAILED_CAPTION>",
            "more_detailed": "<MORE_DETAILED_CAPTION>",
            "basic": "<CAPTION>",
        }
    },
    "ocr": {
        "params": {"store_region_info": bool},
        "task": "<OCR>",
        "region_task": "<OCR_WITH_REGION>"
    },
    "detection": {
        "params": {"detection_type": ["detection", "dense_region_caption", "region_proposal", "open_vocabulary_detection"]},
        "task_mapping": {
            "detection": "<OD>",
            "dense_region_caption": "<DENSE_REGION_CAPTION>",
            "region_proposal": "<REGION_PROPOSAL>",
            "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
        }
    },
    "phrase_grounding": {
        "params": {},  # Removed caption_field, now using universal prompt
        "task": "<CAPTION_TO_PHRASE_GROUNDING>"
    },
    "segmentation": {
        "params": {},  # Removed expression_field, now using universal prompt
        "task": "<REFERRING_EXPRESSION_SEGMENTATION>"
    }
}

logger = logging.getLogger(__name__)

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _convert_bbox(bbox, width, height):
    """Convert bounding box coordinates to FiftyOne format.
    
    Takes raw bounding box coordinates and converts them to normalized coordinates
    in FiftyOne's [x, y, width, height] format. Handles both standard rectangular
    bounding boxes (4 coordinates) and quadrilateral boxes (8 coordinates).

    Args:
        bbox: List of coordinates. Either [x1,y1,x2,y2] for rectangular boxes
              or [x1,y1,x2,y2,x3,y3,x4,y4] for quadrilateral boxes
        width: Width of the image in pixels
        height: Height of the image in pixels

    Returns:
        list: Normalized coordinates in format [x, y, width, height] where:
            - x,y is the top-left corner (normalized by image dimensions)
            - width,height are the box dimensions (normalized by image dimensions)
    """
    if len(bbox) == 4:
        # Standard rectangular box: convert from [x1,y1,x2,y2] to [x,y,w,h]
        # x1,y1 is top-left corner, x2,y2 is bottom-right corner
        return [
            bbox[0] / width,              # x coordinate (normalized)
            bbox[1] / height,             # y coordinate (normalized) 
            (bbox[2] - bbox[0]) / width,  # width (normalized)
            (bbox[3] - bbox[1]) / height  # height (normalized)
        ]
    else:
        # Quadrilateral box: find bounding rectangle that contains all points
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        x_min = min(x1, x2, x3, x4)  # Leftmost x coordinate
        x_max = max(x1, x2, x3, x4)  # Rightmost x coordinate
        y_min = min(y1, y2, y3, y4)  # Topmost y coordinate
        y_max = max(y1, y2, y3, y4)  # Bottommost y coordinate

        return [
            x_min / width,               # x coordinate (normalized)
            y_min / height,              # y coordinate (normalized)
            (x_max - x_min) / width,     # width (normalized)
            (y_max - y_min) / height     # height (normalized)
        ]


def _convert_polyline(contour, width, height):
    """Convert polyline coordinates to FiftyOne format.
    
    Takes raw polyline coordinates and converts them to normalized coordinates
    in FiftyOne's format.

    Args:
        contour: List of interleaved x,y coordinates [x1,y1,x2,y2,...]
        width: Width of the image in pixels
        height: Height of the image in pixels

    Returns:
        list: List of (x,y) tuples representing normalized coordinates of the contour
    """
    # Initialize empty list to store the normalized coordinate pairs
    xy_points = []
    
    # Process the interleaved coordinates by taking them in pairs (x,y)
    # The interleaved format means all values at even indices are x-coordinates
    # and all values at odd indices are y-coordinates
    for i in range(0, len(contour), 2):
        # Check if we have both x and y coordinates available
        # This guards against malformed input where the list might have an odd length
        if i+1 < len(contour):
            # Extract the x-coordinate and normalize it to [0,1] range
            # Normalization makes coordinates independent of image dimensions
            x = contour[i] / width
            
            # Extract the corresponding y-coordinate and normalize it
            y = contour[i+1] / height
            
            # Add the normalized (x,y) pair to our result list
            # This maintains the original shape without adding artificial points
            xy_points.append((x, y))
    
    # Return the list of normalized coordinate pairs
    # This format is ready to be used with FiftyOne's Polyline class
    # The 'closed' parameter in FiftyOne will determine if the shape is closed, not this function
    return xy_points


class Florence2(fom.SamplesMixin, fom.Model):
    """A FiftyOne model for running the Florence-2 multimodal model.
    
    The Florence-2 model supports multiple vision-language tasks including:
    - Image captioning (with varying levels of detail)
    - OCR with region detection
    - Open vocabulary object detection
    - Phrase grounding (linking caption phrases to regions)
    - Referring expression segmentation
    
    Args:
        model_path (str, optional): Model path or HuggingFace repo name.
                                   Defaults to "microsoft/Florence-2-base-ft".
    """
    
    def __init__(
        self, 
        model_path: str,
        operation: str = None,
        prompt: str = None,
        **kwargs
    ):
        """Initialize the Florence-2 model.
        
        Args:
            model_path (str): Model path or HuggingFace repo name
            **kwargs: Additional configuration parameters including:
                - operation (str): The operation to perform
                - Additional operation-specific parameters
        """
        if not model_path:
            raise ValueError("model_path is required")
        
        self._fields = {}
        self.model_path = model_path
        self._prompt = prompt
        self._operation = None
        self.params = kwargs
        
        # Set initial operation if provided
        if operation:
            self.operation = operation  # Use the property setter
            
        # Store additional parameters
        for key, value in kwargs.items():
            self.params[key] = value
        
        # Set device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else None

        logger.info(f"Loading model from local path: {model_path}")
        # Initialize model
        if self.torch_dtype:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                device_map=self.device
            )

        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

    def _get_field(self):
        """Get the field name to use for prompt extraction."""
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        return prompt_field

    @property
    def operation(self):
        """Get the current operation."""
        return self._operation

    @operation.setter
    def operation(self, value):
        """Set the operation with validation."""
        if value not in FLORENCE2_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(FLORENCE2_OPERATIONS.keys())}")
        self._operation = value

    @property
    def prompt(self):
        """Get the current prompt text."""
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        """Set the prompt text."""
        self._prompt = value

    @property
    def detail_level(self):
        """Get the caption detail level."""
        return self.params.get("detail_level", "basic")

    @detail_level.setter
    def detail_level(self, value):
        """Set the caption detail level."""
        valid_levels = FLORENCE2_OPERATIONS["caption"]["params"]["detail_level"]
        if value not in valid_levels:
            raise ValueError(f"Invalid detail level: {value}. Must be one of {valid_levels}")
        self.params["detail_level"] = value

    # Store region info for OCR operation
    @property
    def store_region_info(self):
        """Get whether to store region info for OCR."""
        return self.params.get("store_region_info", False)

    @store_region_info.setter
    def store_region_info(self, value):
        """Set whether to store region info for OCR."""
        if not isinstance(value, bool):
            raise ValueError("store_region_info must be a boolean")
        self.params["store_region_info"] = value

    # Detection type for detection operation
    @property
    def detection_type(self):
        """Get the detection type."""
        return self.params.get("detection_type", "detection")

    @detection_type.setter
    def detection_type(self, value):
        """Set the detection type."""
        valid_types = FLORENCE2_OPERATIONS["detection"]["params"]["detection_type"]
        if value not in valid_types:
            raise ValueError(f"Invalid detection type: {value}. Must be one of {valid_types}")
        self.params["detection_type"] = value

    @property
    def media_type(self):
        """Get the media type supported by this model."""
        return "image"

    def _generate_and_parse(
        self,
        image: Image.Image,
        task: str,
        text_input: Optional[str] = None,
        max_new_tokens: int = 2048,
        num_beams: int = 3,
    ):
        """Generate and parse a response from the model.
        
        Args:
            image: The input image
            task: The task prompt to use
            text_input: Optional text input that includes the task
            max_new_tokens: Maximum new tokens to generate
            num_beams: Number of beams for beam search
            
        Returns:
            The parsed model output
        """
        text = task
        if text_input is not None:
            text = text_input
            
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        
        # Move inputs to device without dtype casting for MPS/CPU
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                if torch.cuda.is_available() and key == "pixel_values":
                    inputs[key] = inputs[key].to(self.device, self.torch_dtype)
                else:
                    inputs[key] = inputs[key].to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task, 
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def _extract_detections(self, parsed_answer, task, image):
        """Extracts object detections from the model's parsed output and converts them to FiftyOne format.
        
        Args:
            parsed_answer: Dict containing the parsed model output with bounding boxes and labels
            task: String specifying the task type - either "<OPEN_VOCABULARY_DETECTION>" or "<OCR_WITH_REGION>"
            image: PIL Image object used to get dimensions for normalizing coordinates
            
        Returns:
            A FiftyOne Detections object containing the extracted detections, where each detection has:
            - A label (either from model output or "object_N" if no label provided)
            - A normalized bounding box in [0,1] coordinates
        """
        # Choose the appropriate keys based on the task type
        label_key = (
            "bboxes_labels" if task == "<OPEN_VOCABULARY_DETECTION>" else "labels"
        )
        bbox_key = "quad_boxes" if task == "<OCR_WITH_REGION>" else "bboxes"
        
        # Extract bounding boxes and labels from the parsed output
        bboxes = parsed_answer[task][bbox_key]
        labels = parsed_answer[task][label_key]
        
        # Build list of FiftyOne Detection objects
        dets = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Create Detection with either model label or fallback object_N label
            dets.append(
                Detection(
                    label=label if label else f"object_{i+1}",
                    bounding_box=_convert_bbox(bbox, image.width, image.height),
                )
            )
            
        # Return all detections wrapped in a FiftyOne Detections object
        return Detections(detections=dets)

    def _extract_polylines(self, parsed_answer, task, image):
        """Extract polylines from segmentation results and convert them to FiftyOne format.
        
        Args:
            parsed_answer (dict): The parsed model output containing polygon coordinates
            task (str): The segmentation task that was performed
            image (PIL.Image): The input image used to normalize coordinates
            
        Returns:
            fiftyone.core.labels.Polylines: A FiftyOne Polylines object containing all
                the extracted polygons, or None if no polygons were found
        """
        polygons = parsed_answer[task]["polygons"]
        if not polygons:
            return None

        polylines = []

        # Process each polygon
        for k, polygon in enumerate(polygons):
            # Process all contours for this polygon
            all_contours = [
                _convert_polyline(contour, image.width, image.height)
                for contour in polygon
            ]

            # Create FiftyOne Polyline object with all contours
            polylines.append(
                Polyline(
                    points=all_contours,
                    label=f"object_{k+1}",
                    filled=True,
                    closed=True,
                )
            )

        return Polylines(polylines=polylines)

    def _predict_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image."""
        logger.info("Starting caption generation...")
        
        detail_level = self.params.get("detail_level", "basic")
        logger.info(f"Caption detail level: {detail_level}")
        
        task_mapping = FLORENCE2_OPERATIONS["caption"]["task_mapping"]
        task = task_mapping.get(detail_level, task_mapping["basic"])
        logger.info(f"Using task: {task}")
        
        try:
            parsed_answer = self._generate_and_parse(image, task)
            logger.info(f"Parsed answer: {parsed_answer}")
            caption = parsed_answer[task].strip()  # Add strip() here to remove whitespace
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}", exc_info=True)
            raise

    def _predict_ocr(self, image: Image.Image) -> Union[str, Detections]:
        """Perform Optical Character Recognition (OCR) on an input image.
        
        This method uses the Florence-2 model to detect and extract text from images.
        It can operate in two modes:
        1. Text extraction only - returns just the detected text
        2. Region-based OCR - returns text with bounding box coordinates
        
        Args:
            image (Image.Image): PIL Image object containing the image to perform OCR on
            
        Returns:
            Union[str, Detections]: Either:
                - A string containing all detected text (when store_region_info=False)
                - A Detections object containing text regions with bounding boxes
                  (when store_region_info=True)
        """
        # Check if region information should be included in output
        store_region_info = self.params.get("store_region_info", False)
        
        if store_region_info:
            # Use region-based OCR task that includes bounding box coordinates
            task = FLORENCE2_OPERATIONS["ocr"]["region_task"]
            parsed_answer = self._generate_and_parse(image, task)
            # Convert the parsed output into FiftyOne Detections format
            return self._extract_detections(parsed_answer, task, image)
        else:
            # Use basic OCR task that returns only text
            task = FLORENCE2_OPERATIONS["ocr"]["task"]
            parsed_answer = self._generate_and_parse(image, task)
            # Return just the extracted text string
            return parsed_answer[task]

    def _predict_detection(self, image: Image.Image) -> Detections:
        """Detect objects in an image using the Florence2 model."""

        detection_type = self.params.get("detection_type", "detection")
        task = FLORENCE2_OPERATIONS["detection"]["task_mapping"][detection_type]
        
        if detection_type == "open_vocabulary_detection":
            if not self.prompt:
                raise ValueError("prompt is required for open_vocabulary_detection")
            text_input = f"{task}\n{self.prompt}"
            parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        else:
            parsed_answer = self._generate_and_parse(image, task)
        
        return self._extract_detections(parsed_answer, task, image)

    def _predict_phrase_grounding(self, image: Image.Image) -> Detections:
        """Ground caption phrases in an image using the Florence2 model."""
        task = FLORENCE2_OPERATIONS["phrase_grounding"]["task"]
        
        # Use self.prompt instead of extracting from sample
        if not self.prompt:
            raise ValueError("No caption provided for phrase grounding")
        
        text_input = f"{task}\n{self.prompt}"
        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        return self._extract_detections(parsed_answer, task, image)

    def _predict_segmentation(self, image: Image.Image) -> Optional[Polylines]:
        """Segment an object in an image based on a referring expression."""
        task = FLORENCE2_OPERATIONS["segmentation"]["task"]
        
        if not self.prompt:
            raise ValueError("No expression provided for segmentation")
        
        text_input = f"{task}\nExpression: {self.prompt}"
        parsed_answer = self._generate_and_parse(image, task, text_input=text_input)
        return self._extract_polylines(parsed_answer, task, image)

    def _predict(self, image: Image.Image, sample=None) -> Any:
        """Process a single image with Florence2 model."""
        # Centralized field handling
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self._prompt = str(field_value)
        
        # Check if operation is set
        if not self.operation:
            raise ValueError("No operation has been set")
        
        # Route to appropriate method
        prediction_methods = {
            "caption": self._predict_caption,
            "ocr": self._predict_ocr,
            "detection": self._predict_detection,
            "phrase_grounding": self._predict_phrase_grounding,
            "segmentation": self._predict_segmentation,
        }
        
        predict_method = prediction_methods.get(self.operation)
        if predict_method is None:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        return predict_method(image)
    
    def predict(self, image: np.ndarray, sample=None, **kwargs) -> Any:
        """Process an image array with Florence2 model."""
        logger.info("Starting prediction...")
        logger.info(f"Operation: {self.operation}")
        logger.info(f"Parameters: {self.params}")
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            result = self._predict(pil_image, sample)  # Pass sample through to _predict
            logger.info(f"Prediction successful: {result}")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise