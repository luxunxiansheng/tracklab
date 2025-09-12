# Pose Estimator Module

This module provides comprehensive pose estimation capabilities for the TrackLab framework, supporting multiple state-of-the-art pose estimation models and frameworks. The module is designed to extract 2D human pose keypoints from images and integrate seamlessly with the tracking pipeline.

## Overview

The pose estimator module supports several leading pose estimation frameworks, each with different strengths and use cases:

- **MMPose**: Comprehensive pose estimation with bottom-up and top-down approaches
- **YOLO Ultralytics Pose**: Real-time pose estimation integrated with YOLO object detection
- **RTMLib**: High-performance RTMPose and RTMO models optimized for speed
- **OpenPifPaf**: Part affinity field-based pose estimation
- **Transformers**: Vision transformer-based pose estimation (ViTPose)

## Supported Frameworks

### MMPose Integration

**Bottom-up Pose Estimation**: Detects all persons and their poses simultaneously in an image.

**Key Features:**

- **Multi-person Detection**: Processes entire images without requiring person detections
- **High Accuracy**: State-of-the-art performance on COCO and other benchmarks
- **Flexible Models**: Support for various MMPose model architectures

**Configuration:**

```yaml
```yaml
_target_: tracklab.wrappers.pose_estimator.mmpose_api.BottomUpMMPose
batch_size: 1
cfg:
  path_to_config: "path/to/mmpose/config.py"
  path_to_checkpoint: "path/to/mmpose/checkpoint.pth"
  download_url: "https://download.openmmlab.com/mmpose/..."
```

### YOLO Ultralytics Pose

**Integrated Detection and Pose**: Combines person detection with pose estimation in a single model.

**Key Features:**

- **Real-time Performance**: Optimized for speed and efficiency
- **End-to-end Pipeline**: Single model for detection and pose estimation
- **Multiple Model Sizes**: From nano to extra-large variants

**Available Models:**

- `yolo11n-pose.pt`: Nano (fastest, lowest accuracy)
- `yolo11s-pose.pt`: Small
- `yolo11m-pose.pt`: Medium (recommended balance)
- `yolo11l-pose.pt`: Large
- `yolo11x-pose.pt`: Extra Large (highest accuracy)

**Configuration:**

```yaml
```yaml
modules:
  pose_estimator:
    _target_: tracklab.wrappers.pose_estimator.yolo_ultralytics_pose_api.YOLOUltralyticsPose
    batch_size: 8
    cfg:
      path_to_checkpoint: "${model_dir}/yolo_ultralytics/yolo11m-pose.pt"
      min_confidence: 0.4
```

### RTMLib (RTMPose & RTMO)

**High-Performance Pose Estimation**: Optimized models for real-time applications.

#### RTMPose

**Top-down Pose Estimation**: Requires person bounding boxes as input.

**Model Variants:**

| Model | Input Size | AP (COCO) | Description |
|-------|------------|-----------|-------------|
| RTMPose-t | 256x192 | 65.9 | Fast, lightweight |
| RTMPose-s | 256x192 | 69.7 | Balanced performance |
| RTMPose-m | 256x192 | 74.9 | Good accuracy |
| RTMPose-l | 256x192 | 76.7 | High accuracy |
| RTMPose-l | 384x288 | 78.3 | Highest accuracy |
| RTMPose-x | 384x288 | 78.8 | Maximum accuracy |

**Configuration:**

```yaml
_target_: tracklab.wrappers.pose_estimator.rtmlib_api.RTMPose
model:
  _target_: rtmlib.RTMPose
  onnx_model: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
  model_input_size: [192, 256]
```

#### RTMO

**One-stage Pose Estimation**: Detects persons and estimates poses simultaneously.

**Key Features:**

- **Integrated Detection**: No separate person detection required
- **High Efficiency**: Optimized for real-time performance
- **Confidence Filtering**: Built-in confidence thresholding

**Configuration:**

```yaml
_target_: tracklab.wrappers.pose_estimator.rtmlib_api.RTMO
model:
  _target_: rtmlib.RTMO
  onnx_model: "path/to/rtmo/model.onnx"
  model_input_size: [192, 256]
min_confidence: 0.3
```

### OpenPifPaf

**Part Affinity Fields**: Advanced pose estimation using part affinity fields and confidence maps.

**Key Features:**

- **Probabilistic Approach**: Uses confidence maps for pose estimation
- **Flexible Architecture**: Supports various backbone networks
- **Research-Oriented**: Popular in academic pose estimation research

### Vision Transformers (ViTPose)

**Transformer-based Pose Estimation**: Uses vision transformer architecture for pose estimation.

**Model Variants:**

- `vitpose-small`: Lightweight transformer model
- `vitpose-base`: Standard transformer model
- `vitpose-large`: Large transformer model
- `vitpose-huge`: Maximum accuracy transformer model

**Configuration:**

```yaml
_target_: tracklab.wrappers.pose_estimator.transformers_api.ViTPose
model:
  _target_: transformers.ViTPose
  model_name: "vitpose-base"
  pretrained: true
```

## Algorithm Details

### Bottom-up vs Top-down Approaches

#### Bottom-up Pose Estimation (MMPose, RTMO)

**Algorithm Overview:**

1. **Feature Extraction**: CNN backbone processes entire image
2. **Keypoint Detection**: Detects all keypoints simultaneously across all persons
3. **Keypoint Grouping**: Associates keypoints to form complete poses
4. **Pose Assembly**: Groups keypoints belonging to same person

**Mathematical Foundation:**

- Uses **Part Affinity Fields (PAF)** or **Part Confidence Maps**
- **Grouping Algorithm**: Hungarian matching or greedy assignment
- **Complexity**: O(N) where N is number of persons

**Advantages:**

- **Scalability**: Handles variable number of persons naturally
- **Efficiency**: Single forward pass for entire image
- **Robustness**: Less sensitive to detection failures

#### Top-down Pose Estimation (RTMPose, ViTPose)

**Algorithm Overview:**

1. **Person Detection**: First detect all persons in image
2. **Region of Interest (RoI) Extraction**: Crop person regions
3. **Single-person Pose Estimation**: Estimate pose for each person independently
4. **Pose Integration**: Combine all person poses

**Mathematical Foundation:**

- **Heatmap Regression**: Predict keypoint probability maps
- **Coordinate Decoding**: Convert heatmaps to (x,y) coordinates
- **Confidence Estimation**: Peak value indicates keypoint confidence

**Advantages:**

- **Accuracy**: Higher precision for individual poses
- **Flexibility**: Easy to incorporate person-specific context
- **Interpretability**: Clear per-person pose estimation

### Framework-Specific Algorithms

#### MMPose Algorithm Details

**Multi-stage Pipeline:**

1. **Backbone Network**: ResNet, HRNet, or ViT-based feature extraction
2. **Feature Pyramid Network (FPN)**: Multi-scale feature fusion
3. **Pose Decoder**: Heatmap or regression-based keypoint prediction
4. **Post-processing**: Keypoint refinement and scoring

**Key Innovations:**

- **MSPN (Multi-Stage Pose Network)**: Progressive refinement
- **DarkPose**: Improved heatmap decoding with distribution awareness
- **SimCC**: Coordinate classification for higher precision

#### YOLO Pose Algorithm Details

**Unified Detection and Pose:**

1. **Feature Extraction**: YOLO backbone (CSPDarknet, EfficientNet)
2. **Detection Head**: Predicts bounding boxes and pose keypoints simultaneously
3. **Keypoint Association**: Direct keypoint-to-person mapping
4. **Pose Refinement**: Optional keypoint refinement module

**Key Features:**

- **Single-shot Prediction**: No separate detection and pose estimation stages
- **Anchor-free Design**: Learns keypoint locations directly
- **Multi-task Learning**: Joint optimization of detection and pose objectives

#### RTMPose Algorithm Details

**Efficient Pose Estimation:**

1. **Lightweight Backbone**: MobileNetV3, EfficientNet, or RepVGG
2. **SimCC (Simultaneous Coordinate Classification)**: Predicts x,y coordinates independently
3. **Codec Design**: Efficient coordinate decoding
4. **Knowledge Distillation**: Teacher-student training for better performance

**Key Innovations:**

- **Coordinate Classification**: More accurate than heatmap regression
- **Efficient Architecture**: Optimized for mobile and edge devices
- **Multi-dataset Training**: Robust across different domains

#### ViTPose Algorithm Details

**Transformer-based Pose Estimation:**

1. **Vision Transformer Backbone**: Self-attention based feature extraction
2. **Pose-specific Pretraining**: Trained on large-scale pose datasets
3. **Hierarchical Feature Fusion**: Multi-scale pose feature integration
4. **Keypoint Regression**: Direct coordinate prediction

**Key Innovations:**

- **Global Context**: Self-attention captures long-range dependencies
- **Scale Invariance**: Better handling of persons at different scales
- **Transfer Learning**: Strong performance with limited pose-specific data

## Comprehensive Framework Comparison

### Performance Comparison

| Framework | Method | AP (COCO) | Speed (FPS) | Memory (GB) | Parameters (M) | Best Use Case |
|-----------|--------|-----------|-------------|-------------|----------------|---------------|
| MMPose | Bottom-up | 75.8 | 15-30 | 2-8 | 50-200 | Research, flexibility |
| YOLO11n-pose | Unified | 65.2 | 120+ | 0.5 | 3.1 | Real-time, edge |
| YOLO11x-pose | Unified | 78.1 | 25 | 4.2 | 56.9 | High accuracy, real-time |
| RTMPose-t | Top-down | 65.9 | 200+ | 0.3 | 2.0 | Ultra-fast inference |
| RTMPose-x | Top-down | 78.8 | 60 | 1.2 | 8.5 | Balanced performance |
| RTMO | Bottom-up | 72.3 | 150+ | 0.8 | 4.2 | One-stage, efficiency |
| ViTPose-B | Top-down | 76.5 | 40 | 3.5 | 88.6 | High accuracy, context |

### Algorithm Comparison

| Aspect | MMPose | YOLO Pose | RTMPose | RTMO | ViTPose |
|--------|--------|-----------|---------|------|---------|
| **Approach** | Bottom-up | Unified | Top-down | Bottom-up | Top-down |
| **Person Detection** | Integrated | Integrated | External | Integrated | External |
| **Keypoint Method** | Heatmaps | Direct | SimCC | Heatmaps | Regression |
| **Architecture** | CNN-based | CNN-based | CNN-based | CNN-based | Transformer |
| **Training Data** | Multi-dataset | COCO | Multi-dataset | COCO | Multi-dataset |
| **Inference Speed** | Medium | Fast | Fastest | Fast | Medium |
| **Accuracy** | High | Medium-High | Medium-High | Medium | Highest |
| **Memory Usage** | High | Low-Medium | Low | Low | High |
| **Scalability** | Excellent | Good | Good | Excellent | Good |
| **Ease of Use** | Complex | Simple | Simple | Simple | Medium |

### Detailed Algorithm Comparison

#### Accuracy vs Speed Trade-offs

```text
High Accuracy    RTMPose-x (78.8 AP) → ViTPose-B (76.5 AP) → MMPose (75.8 AP)
                   ↓
Balanced         YOLO11x-pose (78.1 AP) → RTMO (72.3 AP)
                   ↓
High Speed       RTMPose-t (65.9 AP) → YOLO11n-pose (65.2 AP)
```

#### Memory Efficiency Ranking

```text
Most Efficient: RTMPose-t (0.3 GB) → RTMO (0.8 GB) → YOLO11n-pose (0.5 GB)
                   ↓
Balanced:        RTMPose-x (1.2 GB) → YOLO11x-pose (4.2 GB)
                   ↓
Memory Intensive: ViTPose-B (3.5 GB) → MMPose (2-8 GB)
```

#### Use Case Recommendations

**Choose MMPose when:**

- Maximum research flexibility is needed
- Custom model architectures are required
- Integration with OpenMMLab ecosystem is preferred
- Multi-stage pose refinement is beneficial

**Choose YOLO Pose when:**

- Real-time performance is critical (120+ FPS possible)
- Edge device deployment is planned
- Simple, unified detection and pose pipeline is desired
- Resource constraints are significant

**Choose RTMPose when:**

- Top-down approach with pre-computed detections
- Mobile or edge device optimization is needed
- ONNX deployment for cross-platform compatibility
- Speed-accuracy balance is important

**Choose RTMO when:**

- One-stage detection and pose estimation is required
- Maximum inference speed is needed
- Simplified pipeline complexity is desired
- RTMLib ecosystem integration is preferred

**Choose ViTPose when:**

- Highest accuracy is the primary requirement
- Global context understanding is important
- Scale invariance is critical
- Research applications with ample compute resources

### Technical Implementation Details

#### Coordinate Systems

All frameworks output keypoints in **pixel coordinates** relative to the input image:

- **Origin**: Top-left corner (0,0)
- **X-axis**: Increases left to right
- **Y-axis**: Increases top to bottom
- **Units**: Pixels with sub-pixel precision

#### Confidence Scoring

- **MMPose**: Heatmap peak values (0-1)
- **YOLO Pose**: Object keypoint confidence scores
- **RTMPose**: SimCC confidence from classification
- **RTMO**: Detection and pose confidence combination
- **ViTPose**: Regression confidence from model uncertainty

#### Keypoint Format Standardization

All frameworks follow **COCO format** with 17 keypoints:

```text
0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
```

#### Batch Processing Capabilities

- **MMPose**: Limited batch support, optimized for single images
- **YOLO Pose**: Excellent batch processing with configurable batch sizes
- **RTMPose**: Efficient batch processing for multiple persons
- **RTMO**: Optimized for batch processing with detection integration
- **ViTPose**: Good batch support with transformer efficiency

### Performance Optimization Strategies

#### For Maximum Speed

1. Use RTMPose-t with ONNX runtime
2. Implement batch processing where possible
3. Use lower resolution inputs (256x192 vs 384x288)
4. Leverage GPU acceleration consistently

#### For Maximum Accuracy

1. Use ViTPose-B or RTMPose-x
2. Higher resolution inputs (384x288 preferred)
3. Multi-scale testing augmentation
4. Model ensemble techniques

#### For Edge Deployment

1. Quantize models to INT8 precision
2. Use RTMPose-t or YOLO11n-pose
3. Optimize for specific hardware (TensorRT, CoreML)
4. Implement efficient pre/post-processing

This comprehensive comparison enables informed decision-making for pose estimation framework selection based on specific project requirements, computational resources, and performance targets.

- `keypoints_xyc`: Keypoint coordinates with confidence scores (x, y, confidence)
- `keypoints_conf`: Overall keypoint confidence scores
- `bbox_ltwh`: Bounding box in LTWH format (when applicable)
- `bbox_conf`: Bounding box confidence (when applicable)
- `image_id`: Image identifier
- `video_id`: Video identifier
- `category_id`: Category identifier (usually 1 for person)

## Keypoint Format

**COCO Keypoint Format**: All models output keypoints in COCO format with 17 keypoints:

1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

**Coordinate System**: Keypoints are in image pixel coordinates (x, y) with confidence scores.

## Usage

### Pipeline Integration

Pose estimators are typically used as `ImageLevelModule` components in the TrackLab pipeline:

```python
# In your TrackLab pipeline configuration
pipeline:
  - pose_estimator

modules:
  pose_estimator:
    _target_: tracklab.wrappers.pose_estimator.yolo_ultralytics_pose_api.YOLOUltralyticsPose
    batch_size: 8
    cfg:
      path_to_checkpoint: "${model_dir}/yolo_ultralytics/yolo11m-pose.pt"
      min_confidence: 0.4
```

### Model Selection Guidelines

**Choose YOLO Ultralytics Pose when:**

- Real-time performance is critical
- You want integrated detection and pose estimation
- Working with edge devices or resource constraints

**Choose RTMPose when:**

- High accuracy is required
- You have pre-computed person detections
- Need fine-grained control over model size

**Choose RTMO when:**

- You want one-stage detection and pose estimation
- Maximum speed is required
- Working with RTMLib ecosystem

**Choose MMPose when:**

- Maximum flexibility and model variety is needed
- Research or custom model development
- Integration with OpenMMLab ecosystem

## Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **MMPose**: For MMPose models (optional)
- **Ultralytics**: For YOLO models (optional)
- **RTMLib**: For RTM models (optional)
- **OpenPifPaf**: For PAF models (optional)
- **Transformers**: For ViT models (optional)

## Performance Considerations

### Model Size vs Accuracy Trade-offs

| Framework | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| YOLO Nano | Fastest | Lowest | Real-time, edge devices |
| RTMPose-t | Fast | Good | Real-time applications |
| RTMPose-m | Medium | High | Balanced performance |
| RTMPose-x | Slow | Highest | Maximum accuracy |

### Memory Requirements

- **YOLO Models**: Moderate GPU memory usage
- **RTM Models**: Low memory footprint, ONNX optimized
- **MMPose Models**: Variable based on architecture
- **Transformer Models**: High memory usage for large models

### Batch Processing

- **Batch Size**: Adjust based on GPU memory and desired throughput
- **Preprocessing**: Models handle different input resolutions
- **Postprocessing**: Standardized keypoint format across all models

## Integration Notes

- **Coordinate Systems**: All models output in image pixel coordinates
- **Confidence Scores**: Keypoint confidences are model-specific
- **Missing Keypoints**: Low-confidence keypoints may be filtered or marked invalid
- **Multi-person Handling**: Bottom-up models handle multiple persons automatically
- **Temporal Consistency**: Consider pose smoothing for video sequences

This comprehensive pose estimation module enables TrackLab to extract rich pose information for advanced sports analytics, motion analysis, and behavioral understanding.
