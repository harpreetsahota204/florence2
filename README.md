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

### Register and download the model (one-time setup)

```python

import fiftyone.zoo as foz

foz.register_zoo_model_source("https://github.com/harpreetsahota204/florence2", overwrite=True)

foz.download_zoo_model("https://github.com/harpreetsahota204/florence2", model_name="microsoft/Florence-2-base-ft")
```

### Load the model

```python
model = foz.load_zoo_model(
   "microsoft/Florence-2-base-ft",
    # install_requirements=True #if you are using for the first time and need to download reuirement,
    # ensure_requirements=True #  ensure any requirements are installed before loading the model
   )
```

There are four available Florence2 checkpoints:

1. `microsoft/Florence-2-base` - Base model
2. `microsoft/Florence-2-large` - Large model
3. `microsoft/Florence-2-base-ft` - Fine-tuned base model
4. `microsoft/Florence-2-large-ft` - Fine-tuned large model

## Usage

### Switching Between Operations
The same model instance can be used for different operations by simply changing its properties:

#### Image Captioning

```python

model.operation = "caption"
model.detail_level = "detailed"  # Options: "basic", "detailed", "more_detailed"
dataset.apply_model(model, label_field="captions")
```

#### OCR

```python
model.operation = "ocr"
model.store_region_info = True # True will return detected bounding boxes, False will return just the text
dataset.apply_model(model, label_field="text_detections")
```

#### Object Detection

Florence-2 supports four different types of detection operations, each serving a different purpose:

##### 1. Standard Detection (`detection_type="detection"`)

```python
model.operation = "detection"
model.detection_type = "detection"
dataset.apply_model(model, label_field="standard_detections")
```
- Basic object detection mode
- Detects common objects in the image
- Returns bounding boxes with object labels


##### 2. Dense Region Captioning (`detection_type="dense_region_caption"`)

```python
model.operation = "detection"
model.detection_type = "dense_region_caption"
dataset.apply_model(model, label_field="region_captions")
```

- Generates detailed captions for different regions in the image
- Each region comes with a descriptive caption
- Useful for understanding scene composition

##### 3. Region Proposal (`detection_type="region_proposal"`)
```python
model.operation = "detection"
model.detection_type = "region_proposal"
dataset.apply_model(model, label_field="region_proposals")
```

- Generates potential regions of interest
- Identifies areas that might contain objects
- Useful as a preprocessing step for other tasks

##### 4. Open Vocabulary Detection (`detection_type="open_vocabulary_detection"`)
```python
model.operation = "detection"
model.detection_type = "open_vocabulary_detection"
model.prompt = "Find all the red cars and blue bicycles"
dataset.apply_model(model, label_field="custom_detections")
```

#### Phrase Grounding

```python
model.operation = "phrase_grounding"
model.prompt = "person wearing a red hat"
dataset.apply_model(model, label_field="grounding")
```

#### Switch to Segmentation

```python
model.operation = "segmentation"
model.prompt = "the cat sleeping on the couch"
dataset.apply_model(model, label_field="segments")
```

You can look at the [example notebook](using_florence2_zoo_model.ipynb) for detailed usage syntax.

## Output Formats

- **Captions**: Returns string: Returns str
   - Natural language text responses in English

- **OCR**: Returns either string or `fiftyone.core.labels.Detections`
   - Bounding box coordinates are normalized to [0,1] x [0,1]

- **Detection**: Returns `fiftyone.core.labels.Detections`
   - Bounding box coordinates are normalized to [0,1] x [0,1]
- **Phrase Grounding**: Returns `fiftyone.core.labels.Detections`
   - Bounding box coordinates are normalized to [0,1] x [0,1]

- **Segmentation**: Returns `fiftyone.core.labels.Polylines`
  - Normalized point coordinates [0,1] x [0,1]

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