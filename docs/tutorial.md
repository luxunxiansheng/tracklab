# TrackLab Tutorial

## How to Train a Module in TrackLab

### Overview
TrackLab supports training certain modules (primarily ReID models) on tracking datasets. The training process is integrated into the main pipeline and can be enabled through configuration.

### Supported Modules for Training

Currently, the following modules support training:
- **PRTReID** (`prtreid`) - Pose-based Re-identification
- **KPReID** (`kpreid`) - Keypoint-based Re-identification

### Step-by-Step Training Guide

#### 1. Enable Training in Configuration

First, modify the module configuration to enable training:

```yaml
# In tracklab/configs/modules/reid/prtreid.yaml
training_enabled: true  # Change from false to true

# Configure training parameters
cfg:
  train:
    batch_size: 32
    max_epoch: 20
  data:
    root: "${data_dir}/reid"
    sources: ["SoccerNet"]
    targets: ["SoccerNet"]
```

#### 2. Prepare Training Data

Ensure your dataset has the required training data:

```bash
# For SoccerNet, the data should be in:
${data_dir}/SoccerNetGS/
├── train/
├── valid/
└── test/
```

#### 3. Configure Dataset for Training

Set up the dataset configuration for training:

```yaml
# In tracklab/configs/modules/reid/dataset/prtreid_dataset.yaml
name: "SoccerNet"
train:
  set_name: "train"
  min_samples_per_id: 4
  max_samples_per_id: 15
test:
  set_name: "valid"
  min_samples_per_id: 4
  max_samples_per_id: 10
```

#### 4. Run Training

Execute TrackLab with training enabled:

```bash
# Basic training command
tracklab dataset.nvid=10 modules/reid=prtreid

# With custom training parameters
tracklab \
  dataset.nvid=10 \
  modules/reid=prtreid \
  modules.reid.cfg.train.max_epoch=50 \
  modules.reid.cfg.train.batch_size=64
```

#### 5. Monitor Training Progress

Training progress is logged and can be monitored through:

- **Console Output**: Real-time training metrics
- **TensorBoard**: If enabled in configuration
- **WandB**: If enabled in configuration

```yaml
# Enable logging in prtreid.yaml
cfg:
  project:
    logger:
      use_tensorboard: true
      use_wandb: true
```

### Training Configuration Options

#### Dataset Configuration

```yaml
# prtreid_dataset.yaml
train:
  set_name: "train"           # Which dataset split to use for training
  min_vis: 0.3               # Minimum visibility threshold
  min_h: 30                  # Minimum height in pixels
  min_w: 30                  # Minimum width in pixels
  min_samples_per_id: 4      # Minimum samples per identity
  max_samples_per_id: 15     # Maximum samples per identity
  max_total_ids: -1          # Maximum total identities (-1 for unlimited)
```

#### Model Configuration

```yaml
# prtreid.yaml
cfg:
  model:
    name: "bpbreid"
    backbone: "hrnet32"       # Model backbone
    pooling: "gwap"          # Pooling method
    dim_reduce_output: 256   # Output dimension
    
  loss:
    name: 'part_based'
    part_based:
      weights:
        globl:
          id: 1.0           # Identity loss weight
          tr: 1.0           # Triplet loss weight
          
  train:
    batch_size: 32
    max_epoch: 20
    optimizer:
      name: 'adam'
      lr: 0.0003
```

### Advanced Training Features

#### 1. Multi-GPU Training

```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 tracklab modules.reid.cfg.train.devices="0,1"
```

#### 2. Resume Training

```yaml
# Resume from checkpoint
cfg:
  model:
    resume: "/path/to/checkpoint.pth"
```

#### 3. Custom Data Augmentation

```yaml
# Configure data transforms
cfg:
  data:
    transforms: ["rc", "re", "color_jitter"]
```

#### 4. Learning Rate Scheduling

```yaml
cfg:
  train:
    lr_scheduler:
      name: 'cosine'
      warmup_epochs: 5
```

### Training Output

After training completes, the model is automatically saved:

```
outputs/
└── experiment_name/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── checkpoints/
            │   └── model_best.pth
            ├── logs/
            │   ├── tensorboard/
            │   └── wandb/
            └── config.yaml
```

### Using Trained Models

After training, update your configuration to use the trained model:

```yaml
# In prtreid.yaml
cfg:
  model:
    load_weights: "outputs/experiment_name/YYYY-MM-DD/HH-MM-SS/checkpoints/model_best.pth"
    training_enabled: false  # Disable training for inference
```

### Troubleshooting Training

#### Common Issues

1. **Out of Memory**
   ```yaml
   # Reduce batch size
   cfg:
     train:
       batch_size: 16
   ```

2. **No Training Data**
   ```bash
   # Check data path
   ls ${data_dir}/SoccerNetGS/train/
   ```

3. **Low Training Accuracy**
   ```yaml
   # Increase training epochs
   cfg:
     train:
       max_epoch: 50
   ```

#### Debug Training

```bash
# Run with debug logging
tracklab --debug modules.reid=prtreid

# Check data loading
tracklab dataset.nvid=1 modules/reid=prtreid cfg.data.workers=0
```

### Example Training Configurations

#### Quick Training (Development)
```bash
tracklab \
  dataset.nvid=5 \
  modules.reid=prtreid \
  modules.reid.cfg.train.max_epoch=5 \
  modules.reid.cfg.train.batch_size=8
```

#### Full Training (Production)
```bash
tracklab \
  dataset.nvid=-1 \
  modules.reid=prtreid \
  modules.reid.cfg.train.max_epoch=50 \
  modules.reid.cfg.train.batch_size=32 \
  modules.reid.cfg.project.logger.use_wandb=true
```

### Best Practices

1. **Start Small**: Begin with small dataset and batch size
2. **Monitor Metrics**: Track loss and accuracy during training
3. **Use Validation**: Always validate on held-out data
4. **Save Checkpoints**: Enable regular checkpoint saving
5. **Version Control**: Track configuration changes
6. **GPU Utilization**: Optimize batch size for your GPU memory

### Integration with Tracking Pipeline

After training, the model is automatically integrated into the tracking pipeline:

```yaml
pipeline:
  - bbox_detector
  - reid          # Uses your trained model
  - track
  - calibration
```

The trained ReID model will provide better person re-identification, leading to more accurate tracking results.

## Training YOLO Detectors

TrackLab supports training custom YOLO detectors for domain-specific object detection tasks. This is particularly useful when the pre-trained YOLO models don't perform well on your specific dataset or tracking scenario.

### YOLO Training Architecture

TrackLab's YOLO training follows the same modular pattern as ReID training:

```python
class YOLOUltralytics(ImageLevelModule):
    training_enabled = False  # Set to True to enable training
    
    def train(self, tracking_dataset, pipeline, evaluator, dataset_config):
        """Train YOLO model following TrackLab's training pattern"""
        # Implementation details below
```

### Preparing Training Data

#### 1. Dataset Format
YOLO training requires data in Ultralytics format:

```
dataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

#### 2. Label Format
Each image needs a corresponding `.txt` file with annotations:

```
# Format: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.2 0.3
1 0.7 0.8 0.15 0.25
```

#### 3. Data YAML Configuration
Create a `data.yaml` file:

```yaml
path: /path/to/dataset
train: images/train
val: images/valid
test: images/test

names:
  0: person
  1: ball
  2: referee
```

### Training Configuration

#### Basic Training Config
```yaml
# training_config.yaml
model:
  type: yolov8n  # or yolov8s, yolov8m, yolov8l, yolov8x
  pretrained: true

training:
  epochs: 100
  batch_size: 16
  img_size: 640
  optimizer: Adam
  lr0: 0.001
  lrf: 0.01

data:
  path: /path/to/your/data.yaml
```

#### Advanced Training Config
```yaml
# advanced_training_config.yaml
model:
  type: yolov11m
  pretrained: yolo11m.pt

training:
  epochs: 200
  batch_size: 32
  img_size: 1280
  optimizer: SGD
  lr0: 0.01
  lrf: 0.1
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1

data:
  path: /path/to/data.yaml

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0
```

### Training Implementation

#### 1. Extend YOLO Module for Training
```python
# tracklab/pipeline/bbox_detector/yolo_ultralytics_api.py
import logging
from typing import Any
import torch
import pandas as pd
from ultralytics import YOLO
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
import yaml
from pathlib import Path

log = logging.getLogger(__name__)

def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)

class YOLOUltralytics(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]
    training_enabled = False  # Set to True in config to enable training

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.training_config = cfg.get('training', {})
        
        # Load or initialize model
        if hasattr(cfg, 'path_to_checkpoint') and cfg.path_to_checkpoint:
            self.model = YOLO(cfg.path_to_checkpoint)
        else:
            # Initialize with default model if no checkpoint provided
            self.model = YOLO('yolov8n.pt')  # or any default model
        
        self.model.to(device)
        self.id = 0

    def train(self, tracking_dataset, pipeline, evaluator, dataset_config):
        """Train YOLO model following TrackLab's training pattern"""
        log.info("Starting YOLO training...")
        
        # Get training configuration
        train_cfg = self.training_config
        if not train_cfg:
            log.warning("No training configuration found, using defaults")
            train_cfg = {
                'epochs': 100,
                'batch_size': 16,
                'img_size': 640,
                'data_path': None
            }
        
        # Prepare training data
        data_path = train_cfg.get('data_path')
        if not data_path:
            # Try to infer from dataset config or create default
            data_path = self._prepare_training_data(tracking_dataset, dataset_config)
        
        # Set up training arguments
        training_args = {
            'data': data_path,
            'epochs': train_cfg.get('epochs', 100),
            'batch': train_cfg.get('batch_size', 16),
            'imgsz': train_cfg.get('img_size', 640),
            'device': self.device,
            'workers': 8,
            'project': 'tracklab_training',
            'name': f'yolo_{self.cfg.get("model_type", "custom")}',
            'exist_ok': True,
            'verbose': True
        }
        
        # Add optional training parameters
        if 'optimizer' in train_cfg:
            training_args['optimizer'] = train_cfg['optimizer']
        if 'lr0' in train_cfg:
            training_args['lr0'] = train_cfg['lr0']
        if 'lrf' in train_cfg:
            training_args['lrf'] = train_cfg['lrf']
        
        # Add augmentation if specified
        if 'augmentation' in train_cfg:
            aug = train_cfg['augmentation']
            training_args.update({
                'hsv_h': aug.get('hsv_h', 0.015),
                'hsv_s': aug.get('hsv_s', 0.7),
                'hsv_v': aug.get('hsv_v', 0.4),
                'degrees': aug.get('degrees', 0.0),
                'translate': aug.get('translate', 0.1),
                'scale': aug.get('scale', 0.5),
                'shear': aug.get('shear', 0.0),
                'perspective': aug.get('perspective', 0.0),
                'flipud': aug.get('flipud', 0.0),
                'fliplr': aug.get('fliplr', 0.5),
                'mosaic': aug.get('mosaic', 1.0),
                'mixup': aug.get('mixup', 0.0)
            })
        
        log.info(f"Training YOLO with config: {training_args}")
        
        # Start training
        results = self.model.train(**training_args)
        
        # Update model path to trained weights
        trained_model_path = Path('tracklab_training') / f'yolo_{self.cfg.get("model_type", "custom")}' / 'weights' / 'best.pt'
        if trained_model_path.exists():
            log.info(f"Trained model saved to: {trained_model_path}")
            # Update config to use trained model for future inference
            self.cfg.path_to_checkpoint = str(trained_model_path)
            self.model = YOLO(str(trained_model_path))
        
        return results
    
    def _prepare_training_data(self, tracking_dataset, dataset_config):
        """Prepare training data from TrackLab dataset"""
        # This is a simplified implementation
        # In practice, you'd convert TrackLab's detection data to YOLO format
        
        log.info("Preparing training data from TrackLab dataset...")
        
        # Create a basic data.yaml
        data_yaml = {
            'path': './training_data',
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'names': {0: 'person', 1: 'ball'}  # Default classes
        }
        
        # Save data configuration
        data_path = Path('./training_data/data.yaml')
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(data_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        log.info(f"Created data configuration at {data_path}")
        return str(data_path)

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
```python
# tracklab/pipeline/bbox_detector/yolo_trainable.py
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
from .yolo_ultralytics_api import YOLOUltralyticsApi

class YOLOUltralyticsTrainable(YOLOUltralyticsApi):
    def __init__(self, config):
        super().__init__(config)
        self.training_config = config.get('training', {})

    def train(self, dataset_path, config_path=None):
        """Train YOLO model on custom dataset"""
        # Load training configuration
        if config_path:
            with open(config_path, 'r') as f:
                train_config = yaml.safe_load(f)
        else:
            train_config = self.training_config

        # Initialize model
        model_type = train_config.get('model', {}).get('type', 'yolov8n')
        pretrained = train_config.get('model', {}).get('pretrained', True)

        if isinstance(pretrained, str):
            model = YOLO(pretrained)
        else:
            model = YOLO(f'{model_type}.pt' if pretrained else model_type)

        # Training parameters
        training_args = {
            'data': train_config['data']['path'],
            'epochs': train_config['training']['epochs'],
            'batch': train_config['training']['batch_size'],
            'imgsz': train_config['training']['img_size'],
            'optimizer': train_config['training']['optimizer'],
            'lr0': train_config['training']['lr0'],
            'lrf': train_config['training']['lrf'],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'project': 'tracklab_training',
            'name': f'yolo_{model_type}_custom',
            'exist_ok': True,
            'pretrained': pretrained,
            'verbose': True
        }

        # Add advanced parameters if specified
        if 'momentum' in train_config['training']:
            training_args['momentum'] = train_config['training']['momentum']
        if 'weight_decay' in train_config['training']:
            training_args['weight_decay'] = train_config['training']['weight_decay']
        if 'warmup_epochs' in train_config['training']:
            training_args['warmup_epochs'] = train_config['training']['warmup_epochs']

        # Add augmentation parameters
        if 'augmentation' in train_config:
            aug = train_config['augmentation']
            training_args.update({
                'hsv_h': aug.get('hsv_h', 0.015),
                'hsv_s': aug.get('hsv_s', 0.7),
                'hsv_v': aug.get('hsv_v', 0.4),
                'degrees': aug.get('degrees', 0.0),
                'translate': aug.get('translate', 0.1),
                'scale': aug.get('scale', 0.5),
                'shear': aug.get('shear', 0.0),
                'perspective': aug.get('perspective', 0.0),
                'flipud': aug.get('flipud', 0.0),
                'fliplr': aug.get('fliplr', 0.5),
                'mosaic': aug.get('mosaic', 1.0),
                'mixup': aug.get('mixup', 0.0)
            })

        # Start training
        print(f"Starting YOLO training with config: {training_args}")
        results = model.train(**training_args)

        # Save trained model
        trained_model_path = Path('tracklab_training') / f'yolo_{model_type}_custom' / 'weights' / 'best.pt'
        if trained_model_path.exists():
            print(f"Trained model saved to: {trained_model_path}")
            # Update config to use trained model
            self.config['model_path'] = str(trained_model_path)

        return results

    def load_trained_model(self, model_path):
        """Load a trained YOLO model"""
        self.model = YOLO(model_path)
        print(f"Loaded trained model from: {model_path}")
```

### Training Examples

#### 1. Enable Training in TrackLab Config
```yaml
# tracklab/configs/yolo_training_config.yaml
bbox_detector:
  name: yoloultralytics
  path_to_checkpoint: yolov8n.pt  # Initial model
  training_enabled: true  # Enable training mode
  training:
    epochs: 100
    batch_size: 16
    img_size: 640
    data_path: /path/to/your/data.yaml
    optimizer: SGD
    lr0: 0.01
    lrf: 0.1
    augmentation:
      hsv_h: 0.015
      hsv_s: 0.7
      hsv_v: 0.4
      degrees: 0.0
      translate: 0.1
      scale: 0.5
      shear: 0.0
      perspective: 0.0
      flipud: 0.0
      fliplr: 0.5
      mosaic: 1.0
      mixup: 0.0
```

#### 2. Run Training with TrackLab
```bash
# Training will automatically run when training_enabled: true
tracklab -cn yolo_training_config
```

#### 3. Training from TrackLab Config
```python
# The training is automatically called by TrackLab's main loop
# when training_enabled is set to true in the config
```

### Best Practices

#### 1. Data Quality
- Ensure diverse training data covering different scenarios
- Balance class distribution
- Include various lighting conditions, angles, and occlusions
- Validate annotations for accuracy

#### 2. Training Optimization
- Start with smaller models (YOLOv8n) for experimentation
- Use appropriate batch sizes based on GPU memory
- Monitor validation metrics closely
- Implement early stopping to prevent overfitting

#### 3. Performance Tuning
- Adjust image size based on your target objects
- Fine-tune augmentation parameters for domain-specific data
- Use transfer learning from pre-trained weights
- Experiment with different optimizers and learning rates

#### 4. Evaluation and Iteration
- Evaluate on separate validation set
- Monitor both precision and recall
- Analyze failure cases and augment data accordingly
- Consider ensemble methods for improved performance

### Integration with TrackLab Pipeline

After training, your custom detector is automatically integrated into the TrackLab pipeline. The trained model will be used for inference in subsequent runs.

```yaml
# tracklab/configs/trained_detector_config.yaml
bbox_detector:
  name: yoloultralytics
  path_to_checkpoint: tracklab_training/yolo_custom/weights/best.pt  # Auto-updated after training
  conf_threshold: 0.3
  iou_threshold: 0.5
  classes: [0, 1]  # person, ball
```

This trained detector will now be used in your TrackLab tracking pipeline, providing better detection performance for your specific domain.

### Troubleshooting

#### Common Issues
1. **CUDA out of memory**: Reduce batch size or image size
2. **Poor convergence**: Adjust learning rate or check data quality
3. **Overfitting**: Increase augmentation or add regularization
4. **Low mAP**: Review annotations and class balance

#### Performance Monitoring
- Track training metrics (loss, mAP, precision, recall)
- Monitor validation performance to detect overfitting
- Use TensorBoard or Weights & Biases for visualization
- Save model checkpoints regularly

This comprehensive YOLO training guide enables you to create custom detectors tailored to your specific tracking scenarios, significantly improving TrackLab's performance on domain-specific datasets.