# Florence2 FiftyOne Remote Model Zoo Implementation

This repository provides a FiftyOne Model Zoo implementation for Florence-2, Microsoft's powerful multimodal model. The implementation allows seamless integration of Florence-2's capabilities with FiftyOne's computer vision tools.

> **NOTE**: Due to recent changes in Transformers 4.50.0 (which are to be patched by Hugging Face) please ensure you have transformers<=4.49.0 installed before running the model

## Features

Florence-2 supports multiple vision-language tasks through this implementation:

1. **Image Captioning**
   - Three detail levels: basic, detailed, and more_detailed
   - Generates natural language descriptions of images

2. **Optical Character Recognition (OCR)**
   - Text extraction from images
   - Optional region-based detection with bounding boxes

3. **Object Detection**
   - Multiple detection modes:
     - Standard object detection
     - Dense region captioning
     - Region proposal generation
     - Open vocabulary detection (with custom prompts)

4. **Phrase Grounding**
   - Links phrases to specific regions in images
   - Requires a caption or text prompt

5. **Referring Expression Segmentation**
   - Segments objects based on natural language descriptions
   - Returns polygon contours for the referenced objects

## Installation

```bash
pip install fiftyone
pip install transformers<=4.49.0
```

## Usage

### Basic Example

# Register and download the model (one-time setup)

```python

import fiftyone.zoo as foz
foz.register_zoo_model_source("https://github.com/harpreetsahota204/florence2", overwrite=True)
foz.download_zoo_model("https://github.com/harpreetsahota204/florence2", model_name="microsoft/Florence-2-base-ft")
```

# Load the model

```python
model = foz.load_zoo_model("microsoft/Florence-2-base-ft")
```

There are four available Florence2 checkpoints:

1. `microsoft/Florence-2-base` - Base model
2. `microsoft/Florence-2-large` - Large model
3. `microsoft/Florence-2-base-ft` - Fine-tuned base model
4. `microsoft/Florence-2-large-ft` - Fine-tuned large model

## Usage

### Initial Setup
```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load the model once
model = foz.load_zoo_model("microsoft/Florence-2-base-ft")

# Create or load your dataset
dataset = fo.Dataset.from_images_dir("path/to/images")
```

### Switching Between Operations
The same model instance can be used for different operations by simply changing its properties:

```python
# Image Captioning
model.operation = "caption"
model.detail_level = "detailed"  # Options: "basic", "detailed", "more_detailed"
dataset.apply_model(model, label_field="captions")

# Switch to OCR
model.operation = "ocr"
model.store_region_info = True
dataset.apply_model(model, label_field="text_detections")

# Switch to Object Detection
model.operation = "detection"
model.detection_type = "open_vocabulary_detection"
model.prompt = "Find all the cats and dogs"
dataset.apply_model(model, label_field="detections")

# Switch to Phrase Grounding
model.operation = "phrase_grounding"
model.prompt = "person wearing a red hat"
dataset.apply_model(model, label_field="grounding")

# Switch to Segmentation
model.operation = "segmentation"
model.prompt = "the cat sleeping on the couch"
dataset.apply_model(model, label_field="segments")
```

You can look at the [example notebook](using_florence2_zoo_model.ipynb) for detailed usage syntax.

## Output Formats

- **Captions**: Returns string
- **OCR**: Returns either string or `fiftyone.core.labels.Detections`
- **Detection**: Returns `fiftyone.core.labels.Detections`
- **Phrase Grounding**: Returns `fiftyone.core.labels.Detections`
- **Segmentation**: Returns `fiftyone.core.labels.Polylines`

## Device Support

The implementation automatically selects the appropriate device:
- CUDA if available
- Apple M1/M2 MPS if available
- CPU as fallback

# Citation

```bibtext
@article{xiao2023florence,
  title={Florence-2: Advancing a unified representation for a variety of vision tasks},
  author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}
```