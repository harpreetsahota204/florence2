from huggingface_hub import snapshot_download

"""
Florence-2 model from https://huggingface.co/microsoft/Florence-2-base-ft.
"""
import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

# Import constants from zoo.py to ensure consistency
from .zoo import FLORENCE2_OPERATIONS, Florence2

logger = logging.getLogger(__name__)

def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    
    snapshot_download(repo_id=model_name, local_dir=model_path)

def load_model(model_name, model_path, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load
        model_path: the absolute directory containing the downloaded model
        **kwargs: additional parameters (unused in this implementation)

    Returns:
        a fiftyone.core.models.Model
    """
    if model_name != "voxel51/florence2":
        raise ValueError(f"Unsupported model name '{model_name}'")

    # Import Florence2 from zoo.py
    from .zoo import Florence2
    
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. Please ensure the model has been downloaded "
            "using fiftyone.zoo.download_zoo_model('voxel51/florence2')"
        )
    
    logger.info(f"Loading Florence2 model from {model_path}")

    # Create and return the model - operations specified at apply time
    return Florence2(model_path=model_path)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    if model_name != "voxel51/florence2":
        raise ValueError(f"Unsupported model name '{model_name}'")

    inputs = types.Object()
    
    # Operation selection
    inputs.enum(
        "operation",
        options=list(FLORENCE2_OPERATIONS.keys()),
        default="caption",
        required=True,
        label="Operation",
        description="Type of operation to perform with Florence2 model"
    )
    
    # Caption detail level
    inputs.enum(
        "detail_level",
        options=["basic", "detailed", "more_detailed"],
        default="basic",
        required=False,
        label="Caption detail level",
        description="Level of detail in generated captions",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["caption"],
            default_visibility=False,
        ),
    )
    
    # OCR parameters
    inputs.bool(
        "store_region_info",
        default=False,
        required=False,
        label="Store region information",
        description="Whether to include text region bounding boxes",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["ocr"],
            default_visibility=False,
        ),
    )
    
    # Detection parameters
    inputs.enum(
        "detection_type",
        options=["detection", "dense_region_caption", "region_proposal", "open_vocabulary_detection"],
        default="detection",
        required=False,
        label="Detection type",
        description="Type of detection to perform",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["detection"],
            default_visibility=False,
        ),
    )
    
    inputs.str(
        "text_prompt",
        default=None,
        required=False,
        label="Text prompt",
        description="Optional prompt for guiding detection (e.g., 'Find all animals in this image')",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["detection"],
            default_visibility=False,
        ),
    )
    
    # Phrase grounding parameters
    inputs.str(
        "caption",
        default=None,
        required=False,
        label="Caption",
        description="Caption text to ground in the image",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["phrase_grounding"],
            default_visibility=False,
        ),
    )
    
    inputs.str(
        "caption_field",
        default=None,
        required=False,
        label="Caption field",
        description="Name of the sample field containing caption to ground",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["phrase_grounding"],
            default_visibility=False,
        ),
    )
    
    # Segmentation parameters
    inputs.str(
        "expression",
        default=None,
        required=False,
        label="Expression",
        description="Natural language expression describing the object to segment",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["segmentation"],
            default_visibility=False,
        ),
    )
    
    inputs.str(
        "expression_field",
        default=None,
        required=False,
        label="Expression field",
        description="Name of the sample field containing the expression",
        view=types.VisibilityView(
            tied_field="operation",
            tied_values=["segmentation"],
            default_visibility=False,
        ),
    )
    
    return types.Property(inputs)


def parse_parameters(model_name, ctx, params):
    """Processes and validates the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext
        params: a params dict
        
    Raises:
        ValueError: If required parameters are missing
    """
    # Ensure operation is specified
    if "operation" not in params:
        raise ValueError("Operation must be specified")
        
    operation = params["operation"]
    
    # Validate required parameters for specific operations
    if operation == "phrase_grounding":
        if not params.get("caption") and not params.get("caption_field"):
            raise ValueError("Either 'caption' or 'caption_field' must be provided for phrase_grounding")
            
        # If both are provided, prefer caption over caption_field
        if params.get("caption") and params.get("caption_field"):
            logger.warning("Both 'caption' and 'caption_field' provided; using 'caption'")
            params.pop("caption_field")
            
    if operation == "segmentation":
        if not params.get("expression") and not params.get("expression_field"):
            raise ValueError("Either 'expression' or 'expression_field' must be provided for segmentation")
            
        # If both are provided, prefer expression over expression_field  
        if params.get("expression") and params.get("expression_field"):
            logger.warning("Both 'expression' and 'expression_field' provided; using 'expression'")
            params.pop("expression_field")