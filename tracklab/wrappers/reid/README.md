# TrackLab Re-Identification (ReID) Module

The ReID module provides state-of-the-art person and object re-identification capabilities for the TrackLab multi-object tracking framework. This module enables robust identity association across frames and cameras by extracting discriminative feature embeddings from detected objects.

## Overview

Re-identification (ReID) is crucial for multi-object tracking as it helps maintain consistent identities when objects are temporarily occluded, leave and re-enter the scene, or when tracking across multiple camera views. The TrackLab ReID module offers two specialized implementations:

- **KPReId**: Keypoint Promptable ReID - Uses pose keypoints to enhance feature extraction
- **PRTReId**: PoseTrack ReID - Specialized for sports tracking with role detection

## Algorithm Fundamentals

### Core Principles of Re-Identification

Re-identification algorithms address the fundamental challenge of matching the same object across different images or video frames, despite variations in:

- **Appearance Changes**: Lighting conditions, viewing angles, clothing, pose variations
- **Occlusion**: Partial blocking by other objects or scene elements
- **Temporal Gaps**: Objects leaving and re-entering the field of view
- **Camera Variations**: Different camera positions, resolutions, and settings

The core idea is to learn a **discriminative feature representation** that captures the essential identity characteristics of objects while being robust to irrelevant variations.

### Feature Learning Paradigm

Modern ReID approaches use deep neural networks to learn feature embeddings in a metric learning framework:

1. **Feature Extraction**: Convolutional neural networks extract hierarchical features from input images
2. **Embedding Learning**: Features are transformed into compact, discriminative embeddings
3. **Similarity Matching**: Cosine similarity or Euclidean distance measures identity relationships
4. **Ranking Optimization**: Triplet loss or contrastive loss ensures similar identities are close, dissimilar ones are far apart

### Key Challenges and Solutions

#### 1. Pose Variations

**Challenge**: Different body poses dramatically change appearance

**Solution**: Incorporate pose information to normalize pose variations

#### 2. Occlusion Handling

**Challenge**: Partial visibility due to occlusion or cropping

**Solution**: Part-based representations with visibility scores

#### 3. Domain Adaptation

**Challenge**: Models trained on one dataset may not generalize to others

**Solution**: Domain adaptation techniques and multi-dataset training

#### 4. Scale Variations

**Challenge**: Objects appear at different sizes across cameras

**Solution**: Multi-scale feature extraction and normalization

### KPReId: Keypoint Promptable ReID

#### Fundamental Concept

KPReId leverages human pose keypoints as "prompts" to guide the feature extraction process. The key insight is that pose information provides strong semantic cues about body structure and orientation, enabling more robust identity matching.

#### Technical Approach

**Keypoint-Guided Feature Extraction**:

- Pose keypoints serve as anatomical landmarks
- Features are extracted relative to keypoint positions
- Pose normalization reduces appearance variations

**Prompt Mask Generation**:

- Keypoints are converted to probability maps (heatmaps)
- These heatmaps act as attention mechanisms
- Model focuses on relevant body regions

**Multi-Part Representation**:

- Body divided into semantic parts (head, torso, limbs)
- Each part modeled separately
- Part-level features combined for final embedding

**Mathematical Foundation**:

```math
Feature Extraction: f(x, k) → e
Where:
- x: input image
- k: keypoint coordinates
- e: discriminative embedding
- f: deep neural network with keypoint conditioning
```

#### Advantages

- **Pose Invariance**: Robust to pose changes
- **Occlusion Robustness**: Part-level features handle partial visibility
- **Semantic Awareness**: Leverages human body structure knowledge

### PRTReId: PoseTrack ReID

#### Fundamental Concept
PRTReId is designed specifically for sports tracking scenarios, particularly soccer, where role information (player, referee, goalkeeper) provides crucial context for identity disambiguation.

#### Technical Approach

**Role-Aware Feature Learning**:
- Joint learning of appearance and role features
- Role classification integrated with ReID
- Multi-task learning framework

**Body Mask Integration**:
- Full body segmentation masks
- Part-level visibility modeling
- Occlusion-aware feature extraction

**Sports-Specific Optimizations**:
- Trained on sports datasets (SoccerNet, PoseTrack)
- Handles team uniforms and equipment
- Optimized for fast-moving objects

**Multi-Modal Fusion**:

```math
Combined Representation: e = α·e_appearance + β·e_pose + γ·e_role
Where:
- e_appearance: visual appearance features
- e_pose: pose and body structure features
- e_role: role classification features
- α, β, γ: learned fusion weights
```

#### Advantages

- **Context Awareness**: Role information aids disambiguation
- **Sports Optimization**: Tailored for athletic tracking scenarios
- **Multi-Modal Integration**: Combines appearance, pose, and semantic cues

### Comparative Analysis

| Aspect | KPReId | PRTReId |
|--------|--------|----------|
| **Primary Focus** | General pose-aware ReID | Sports-specific tracking |
| **Key Innovation** | Keypoint prompting | Role detection integration |
| **Best Use Case** | General multi-person tracking | Soccer/player tracking |
| **Pose Utilization** | Keypoint-guided extraction | Full pose + role modeling |
| **Training Data** | General ReID datasets | Sports-specific datasets |
| **Output Features** | Embeddings + visibility | Embeddings + roles + masks |

### Training and Inference Pipeline

#### Training Phase

1. **Data Preparation**: Image crops with pose annotations
2. **Feature Extraction**: Forward pass through backbone network
3. **Loss Computation**: Identity classification + triplet loss
4. **Optimization**: Gradient descent with learning rate scheduling

#### Inference Phase

1. **Detection Processing**: Extract crops from bounding boxes
2. **Feature Extraction**: Generate embeddings for each detection
3. **Similarity Computation**: Compare embeddings across frames
4. **Identity Association**: Link detections to existing tracks

### Performance Considerations

#### Computational Complexity

- **KPReId**: Moderate complexity due to keypoint processing
- **PRTReId**: Higher complexity with role classification
- **Batch Processing**: Essential for efficient GPU utilization

#### Memory Requirements

- Feature embeddings: Compact 512-2048 dimensional vectors
- Model weights: ~100-500MB depending on backbone
- Training memory: Scales with batch size and model complexity

#### Real-time Performance

- Inference speed: 10-50ms per detection on modern GPUs
- Batch processing: Significant speedup for multiple detections
- Trade-offs: Accuracy vs. speed optimization available

## Features

- 🔍 **Advanced Feature Extraction**: Deep learning-based embedding generation
- 🎯 **Pose-Aware ReID**: Integration with pose estimation for improved accuracy
- 🏃 **Sports-Optimized**: Specialized models for soccer and sports tracking
- 🎭 **Role Detection**: Automatic classification of player roles (player, referee, goalkeeper, etc.)
- 📊 **Visibility Scoring**: Confidence scores for different body parts
- 🔧 **Modular Design**: Easy integration with TrackLab pipeline
- 🎓 **Training Support**: Built-in training capabilities for custom datasets

## Architecture

### Core Classes

#### KPReId (Keypoint Promptable ReID)

```python
from tracklab.wrappers.reid.kpreid_api import KPReId

# Key features:
# - Uses keypoints for enhanced feature extraction
# - Supports prompt masks and target masks
# - Integrates with torchreid framework
# - Automatic model downloading from Hugging Face
```

**Inputs:**

- `bbox_ltwh`: Bounding box coordinates
- `keypoints` (optional): Pose keypoints
- `negative_kps` (optional): Negative keypoints for occlusion handling

**Outputs:**

- `embeddings`: Feature vectors for similarity matching
- `visibility_scores`: Confidence scores for body parts
- `parts_masks` (optional): Part-level segmentation masks

#### PRTReId (PoseTrack ReID)

```python
from tracklab.wrappers.reid.prtreid_api import PRTReId

# Key features:
# - Role detection and classification
# - Body mask generation
# - Soccer-specific optimizations
# - Integration with SoccerNet dataset
```

**Inputs:**

- `bbox_ltwh`: Bounding box coordinates
- `keypoints` (optional): Pose keypoints

**Outputs:**

- `embeddings`: Feature vectors for similarity matching
- `visibility_scores`: Confidence scores for body parts
- `body_masks`: Full body segmentation masks
- `role_detection`: Detected roles (player, referee, goalkeeper, ball, other)
- `role_confidence`: Confidence scores for role classification

### Dataset Classes

#### KPReId Dataset

- Handles keypoint-based data preparation
- Generates Gaussian heatmaps from keypoints
- Supports multiple mask preprocessing strategies
- Compatible with torchreid data pipeline

#### PRTReId Dataset

- Specialized for PoseTrack and SoccerNet datasets
- Includes role mapping and classification
- Supports body mask transformations
- Integrated with prtreid framework

## Installation

### Dependencies

The ReID module requires the following key dependencies:

```bash
# Core dependencies
pip install torch torchvision torchreid
pip install huggingface-hub omegaconf yacs
pip install pandas scikit-image tqdm

# For KPReId
pip install torchreid

# For PRTReId
pip install prtreid
```

### Model Weights

Pre-trained models are automatically downloaded:

- **KPReId**: Downloads from Hugging Face (`trackinglaboratory/keypoint_promptable_reid`)
- **PRTReId**: Downloads from Zenodo (SoccerNet baseline model)

## Configuration

### KPReId Configuration

```yaml
# tracklab/configs/modules/reid/kpr.yaml
_target_: tracklab.wrappers.reid.kpreid_api.KPReId
training_enabled: false
batch_size: 2
save_path: reid
use_keypoints_visibility_scores_for_reid: false

cfg:
  model:
    load_weights: "${model_dir}/reid/kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"
    kpr:
      keypoints:
        enabled: false
        prompt_masks: keypoints_gaussian
        prompt_preprocess: cck6
      masks:
        preprocess: "five_v"
```

### PRTReId Configuration

```yaml
# tracklab/configs/modules/reid/prtreid.yaml
_target_: sn_gamestate.reid.prtreid_api.PRTReId
batch_size: 32
save_path: reid
use_keypoints_visibility_scores_for_reid: false
training_enabled: false

cfg:
  model:
    name: "bpbreid"
    load_weights: "${model_dir}/reid/prtreid-soccernet-baseline.pth.tar"
    bpbreid:
      pooling: "gwap"
      last_stride: 1
      test_embeddings: ["global", "parts", "body_masks", "role_cls"]
```

## Usage

### Basic Usage in TrackLab Pipeline

```python
from tracklab.wrappers.reid.kpreid_api import KPReId
import torch

# Initialize ReID module
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reid_module = KPReId(
    cfg=config,
    device=device,
    save_path="outputs/reid",
    training_enabled=False,
    batch_size=32
)

# Process detections
detections_df = pd.DataFrame({
    'bbox_ltwh': [[100, 200, 50, 100]],  # [left, top, width, height]
    'keypoints': [keypoints_array]  # Optional pose keypoints
})

# Get embeddings
embeddings_df = reid_module.process_batch(detections_df, metadata_df)
print(embeddings_df['embeddings'])  # Feature vectors
print(embeddings_df['visibility_scores'])  # Visibility scores
```

### Training Custom Models

```python
# Enable training mode
reid_module = KPReId(
    cfg=config,
    device=device,
    save_path="outputs/reid",
    training_enabled=True,  # Enable training
    batch_size=32
)

# Start training
reid_module.train()
```

### Integration with TrackLab

The ReID modules integrate seamlessly with the TrackLab tracking pipeline:

```yaml
# In your TrackLab config
modules:
  - detector: yolov8
  - pose_estimator: rtmpose
  - reid: kpr  # or prtreid
  - tracker: strong_sort

pipeline:
  - detector
  - pose_estimator
  - reid
  - tracker
```

## API Reference

### KPReId Class

#### KPReId Constructor

```python
KPReId(
    cfg: DictConfig,
    device: torch.device,
    save_path: str,
    training_enabled: bool,
    batch_size: int,
    job_id: int = 0
)
```

#### KPReId Methods

##### preprocess(image, detection, metadata)

- Crops detection region from image
- Applies keypoint-based preprocessing
- Returns batch dictionary for model input

##### process(batch, detections, metadatas)

- Extracts features using the ReID model
- Returns DataFrame with embeddings and visibility scores

##### train()

- Trains the ReID model using configured datasets

### PRTReId Class

#### PRTReId Constructor

```python
PRTReId(
    cfg: DictConfig,
    tracking_dataset: TrackingDataset,
    dataset: DatasetConfig,
    device: torch.device,
    save_path: str,
    job_id: int,
    use_keypoints_visibility_scores_for_reid: bool,
    training_enabled: bool,
    batch_size: int
)
```

#### PRTReId Methods

##### preprocess(image, detection, metadata)

- Crops and preprocesses detection for feature extraction
- Handles mask preprocessing if available

##### process(batch, detections, metadatas)

- Performs feature extraction with role classification
- Returns comprehensive ReID results including roles and masks

##### train()

- Trains the model with role detection capabilities

## Performance Tips

1. **Batch Processing**: Use appropriate batch sizes based on your GPU memory
2. **Keypoint Integration**: Enable keypoints for better performance when available
3. **Model Selection**: Choose KPReId for general tracking, PRTReId for sports
4. **Preprocessing**: Ensure consistent image preprocessing across training/inference
5. **GPU Utilization**: Models run faster on GPU with CUDA support

## Troubleshooting

### Common Issues

#### Model Download Failures

```python
# Manual download for KPReId
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="trackinglaboratory/keypoint_promptable_reid",
    filename="model_name.pth",
    local_dir="path/to/models"
)
```

#### Memory Issues

- Reduce batch size in configuration
- Use smaller input image sizes
- Enable gradient checkpointing for training

#### Keypoint Format Errors

- Ensure keypoints are in the expected format (x,y,confidence)
- Check coordinate system (image vs normalized)

## Contributing

To contribute to the ReID module:

1. **Add New ReID Methods**: Extend the `DetectionLevelModule` base class
2. **Improve Performance**: Optimize preprocessing and inference pipelines
3. **Add Datasets**: Create new dataset classes for different domains
4. **Enhance Training**: Improve training procedures and loss functions

## References

- [TorchReID](https://github.com/KaiyangZhou/deep-person-reid) - Base framework for ReID
- [PoseTrack](https://posetrack.net/) - Pose tracking dataset
- [SoccerNet](https://www.soccer-net.org/) - Soccer video understanding
- [TrackLab](https://github.com/TrackingLaboratory/tracklab) - Main tracking framework

## License

This module is part of TrackLab and follows the same license terms.
