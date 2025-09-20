# KPReID: Keypoint-based Person Re-Identification Package

## Overview

The KPReID package implements a sophisticated keypoint-based person re-identification system for the TrackLab multi-object tracking framework. This module enhances traditional ReID approaches by incorporating pose keypoint information to generate more discriminative person representations.

### Key Innovation

Unlike standard appearance-based ReID methods that rely solely on visual features, KPReID leverages human pose keypoints to create pose-aware masks. These masks help the model focus on relevant body parts while suppressing background noise, leading to improved re-identification accuracy, especially in challenging scenarios like occlusion, pose variations, and crowded environments.

### Integration with TrackLab

KPReID is designed as a detection-level module within TrackLab's modular pipeline architecture. It seamlessly integrates with:

- **Detection modules**: Receives bounding box detections from detectors like YOLO, RTMDet
- **Pose estimation modules**: Utilizes keypoint predictions from pose estimators like RTMPose, ViTPose
- **Tracking modules**: Provides enhanced embeddings for trackers like DeepSORT, ByteTrack

## Architecture

### Core Components

1. **Keypoint Processing Pipeline**
   - Converts raw keypoint coordinates into gaussian heatmaps
   - Applies visibility thresholds to filter unreliable keypoints
   - Generates pose-aware masks for feature extraction

2. **Dual Mask System**
   - **Prompt Masks**: Guide the attention mechanism during feature extraction
   - **Target Masks**: Define regions of interest for embedding generation

3. **Feature Extraction Backbone**
   - Utilizes ResNet-IBN-A architecture for robust feature learning
   - Incorporates mask-guided attention for pose-aware representations
   - Produces high-dimensional embeddings with visibility scores

### Technical Details

- **Input Format**: Expects bounding boxes in LTWH format and associated keypoint data
- **Output Format**: Generates 2048-dimensional embeddings with per-keypoint visibility scores
- **Supported Datasets**: Optimized for PoseTrack21 and other pose-annotated datasets
- **Framework Integration**: Built on top of Torchreid for maximum compatibility

## Algorithm Details

### Mathematical Formulation

The KPReID algorithm can be formally described as follows:

Given a person detection with bounding box $B = [x, y, w, h]$ and pose keypoints $K = \{k_i\}_{i=1}^{17}$ where each keypoint $k_i = (x_i, y_i, v_i)$ consists of coordinates and visibility score, the algorithm computes:

1. **Gaussian Heatmap Generation**:
   $$
   H_i(x,y) = \exp\left(-\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma^2}\right) \cdot v_i
   $$
   where $\sigma$ is the gaussian kernel size and $v_i$ is the visibility score.

2. **Pose Mask Generation**:
   $$
   M_{prompt} = \bigcup_{i=1}^{17} H_i(x,y) \odot W_{prompt}
   $$
   $$
   M_{target} = \bigcup_{i=1}^{17} H_i(x,y) \odot W_{target}
   $$
   where $W_{prompt}$ and $W_{target}$ are learned weight matrices.

3. **Feature Extraction with Mask Attention**:
   $$
   f = \text{ResNet-IBN-A}(I) \odot M_{prompt}
   $$
   $$
   e = \text{FC}(f) \odot M_{target}
   $$
   where $I$ is the input image crop, $f$ are intermediate features, and $e$ is the final embedding.

### Processing Pipeline

### Processing Pipeline

#### Step 1: Keypoint Preprocessing

```python
def preprocess_keypoints(keypoints, visibility_scores, image_size):
    """
    Convert raw keypoints to gaussian heatmaps with visibility weighting
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        visibility_scores: (17,) array of confidence scores
        image_size: (H, W) target heatmap size
    
    Returns:
        heatmaps: (17, H, W) gaussian heatmaps
    """
    heatmaps = []
    for i, (kp, vis) in enumerate(zip(keypoints, visibility_scores)):
        if vis > vis_thresh:
            heatmap = gaussian_kernel(kp, sigma=2.0, size=image_size)
            heatmaps.append(heatmap * vis)
        else:
            heatmaps.append(np.zeros(image_size))
    return np.stack(heatmaps)
```

#### Step 2: Mask Generation

The algorithm employs a dual-mask strategy:

- **Prompt Masks**: Used during feature extraction to guide attention
- **Target Masks**: Applied to the final embeddings for pose-aware refinement

```python
def generate_pose_masks(heatmaps, mode='gaussian'):
    """
    Generate pose-aware masks from keypoint heatmaps
    
    Args:
        heatmaps: (17, H, W) keypoint heatmaps
        mode: Mask generation strategy ('gaussian', 'body_parts', 'skeleton')
    
    Returns:
        prompt_mask: (H, W) attention mask for feature extraction
        target_mask: (H, W) refinement mask for embeddings
    """
    if mode == 'gaussian':
        # Combine all keypoints with equal weighting
        combined = np.sum(heatmaps, axis=0)
        prompt_mask = normalize(combined)
        target_mask = prompt_mask
    elif mode == 'body_parts':
        # Group keypoints by body parts (head, torso, limbs)
        head_mask = combine_keypoints(heatmaps, [0, 1, 2, 3, 4])  # face keypoints
        torso_mask = combine_keypoints(heatmaps, [5, 6, 11, 12])  # shoulder/hip
        limbs_mask = combine_keypoints(heatmaps, [7, 8, 9, 10, 13, 14, 15, 16])
        
        prompt_mask = 0.4 * head_mask + 0.4 * torso_mask + 0.2 * limbs_mask
        target_mask = torso_mask  # Focus on torso for identity
    elif mode == 'skeleton':
        # Generate skeleton-based mask using keypoint connections
        skeleton_mask = generate_skeleton_mask(heatmaps, connections)
        prompt_mask = skeleton_mask
        target_mask = skeleton_mask
    
    return prompt_mask, target_mask
```

#### Step 3: Mask-Guided Feature Extraction

```python
def extract_features_with_masks(image, prompt_mask, backbone):
    """
    Extract features using mask-guided attention
    
    Args:
        image: (3, H, W) input image tensor
        prompt_mask: (H, W) attention mask
        backbone: ResNet-IBN-A model
    
    Returns:
        features: (C, H', W') feature maps
    """
    # Extract base features
    base_features = backbone(image)  # (C, H, W)
    
    # Apply mask attention
    mask_tensor = torch.tensor(prompt_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_tensor = F.interpolate(mask_tensor, size=base_features.shape[-2:], mode='bilinear')
    
    attended_features = base_features * mask_tensor.expand_as(base_features)
    
    return attended_features
```

#### Step 4: Embedding Generation and Refinement

```python
def generate_embedding(features, target_mask, embedding_head):
    """
    Generate final embedding with target mask refinement
    
    Args:
        features: (C, H, W) attended feature maps
        target_mask: (H, W) refinement mask
        embedding_head: FC layer for embedding generation
    
    Returns:
        embedding: (D,) feature vector
        visibility_scores: (17,) keypoint visibility scores
    """
    # Global average pooling with mask weighting
    mask_tensor = torch.tensor(target_mask).unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(mask_tensor, size=features.shape[-2:], mode='bilinear')
    
    pooled_features = torch.sum(features * mask_tensor, dim=(-2, -1))
    pooled_features = pooled_features / (torch.sum(mask_tensor) + 1e-8)
    
    # Generate embedding
    embedding = embedding_head(pooled_features)
    embedding = F.normalize(embedding, p=2, dim=-1)
    
    # Compute visibility scores from original keypoints
    visibility_scores = compute_visibility_scores(original_keypoints)
    
    return embedding, visibility_scores
```

### Key Algorithm Components

#### Gaussian Kernel Generation

```python
def gaussian_kernel(center, sigma=2.0, size=(64, 32), aspect_ratio=2.0):
    """
    Generate 2D gaussian kernel for keypoint heatmap
    
    Args:
        center: (x, y) keypoint coordinates
        sigma: standard deviation
        size: (H, W) output size
        aspect_ratio: elongation factor for pose-aware kernels
    """
    x, y = center
    H, W = size
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    
    # Anisotropic gaussian for pose keypoints
    sigma_x = sigma
    sigma_y = sigma * aspect_ratio
    
    # Compute gaussian values
    exponent = -0.5 * (
        ((x_grid - x) / sigma_x) ** 2 + 
        ((y_grid - y) / sigma_y) ** 2
    )
    
    heatmap = np.exp(exponent)
    return heatmap
```

#### Visibility Score Computation

```python
def compute_visibility_scores(keypoints, bbox, image_size):
    """
    Compute refined visibility scores considering bbox and image boundaries
    
    Args:
        keypoints: (17, 3) array with (x, y, confidence)
        bbox: (x, y, w, h) bounding box
        image_size: (H, W) image dimensions
    """
    scores = []
    for kp in keypoints:
        x, y, conf = kp
        
        # Boundary check
        if not (0 <= x < image_size[1] and 0 <= y < image_size[0]):
            scores.append(0.0)
            continue
            
        # BBox containment check
        bbox_left, bbox_top = bbox[0], bbox[1]
        bbox_right = bbox_left + bbox[2]
        bbox_bottom = bbox_top + bbox[3]
        
        if not (bbox_left <= x <= bbox_right and bbox_top <= y <= bbox_bottom):
            scores.append(conf * 0.5)  # Reduce score for keypoints outside bbox
        else:
            scores.append(conf)
    
    return np.array(scores)
```

#### Multi-Scale Processing

```python
def multi_scale_processing(image, keypoints, scales=[0.5, 1.0, 1.5]):
    """
    Process image at multiple scales for robust feature extraction
    
    Args:
        image: Input image tensor
        keypoints: Keypoint coordinates
        scales: List of scale factors
    """
    embeddings = []
    
    for scale in scales:
        # Scale image
        scaled_image = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode='bilinear')
        scaled_image = scaled_image.squeeze(0)
        
        # Scale keypoints accordingly
        scaled_keypoints = keypoints * scale
        
        # Generate masks and extract features
        heatmaps = preprocess_keypoints(scaled_keypoints, vis_scores, scaled_image.shape[-2:])
        prompt_mask, target_mask = generate_pose_masks(heatmaps)
        
        features = extract_features_with_masks(scaled_image, prompt_mask, backbone)
        embedding, _ = generate_embedding(features, target_mask, embedding_head)
        
        embeddings.append(embedding)
    
    # Fuse multi-scale embeddings
    fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return F.normalize(fused_embedding, p=2, dim=-1)
```

## Training and Fine-tuning Guide

### Getting Started with Training

#### Prerequisites
Before training KPReID, ensure you have:
- **Dataset**: Pose-annotated person ReID dataset (PoseTrack21, custom dataset with keypoints)
- **Hardware**: GPU with at least 8GB VRAM (16GB+ recommended)
- **Dependencies**: PyTorch, Torchreid, OpenCV, and other dependencies installed
- **Pretrained Models**: ResNet-IBN-A backbone weights (automatically downloaded)

#### Quick Start Training Script

```python
import torch
from torchreid.scripts.main import build_config, build_torchreid_model_engine
from tracklab.pipeline.reid.kpreid.kpreid_dataset import ReidDataset

# 1. Prepare your configuration
config = {
    'model': {
        'name': 'resnet50_ibn_a',
        'kpr': {
            'enabled': True,
            'keypoints': {
                'enabled': True,
                'prompt_masks': 'keypoints_gaussian',
                'vis_thresh': 0.5
            }
        }
    },
    'data': {
        'sources': ['posetrack21'],  # Your dataset name
        'targets': ['posetrack21'],
        'height': 256,
        'width': 128,
        'transforms': ['random_flip', 'random_crop', 'random_erase']
    },
    'loss': {
        'name': 'triplet',
        'margin': 0.3
    },
    'train': {
        'optim': 'adam',
        'lr': 0.0003,
        'weight_decay': 0.0005,
        'max_epoch': 60,
        'batch_size': 32
    }
}

# 2. Initialize training components
cfg = build_config(config)
model = build_torchreid_model_engine(cfg)
dataset = ReidDataset(cfg)

# 3. Start training
model.train(dataset)
```

### Data Preparation

#### Required Data Components

For training KPReID, you need the following data for each person instance:

1. **RGB Images**: High-quality person images or video frames
2. **Bounding Boxes**: Person localization coordinates `[x, y, width, height]`
3. **Pose Keypoints**: 17-keypoint pose estimation (COCO format)
4. **Identity Labels**: Person identity annotations for supervised training
5. **Camera IDs**: Camera/viewpoint information (for cross-camera evaluation)

#### Minimum Data Requirements

| Component | Format | Shape | Description |
|-----------|--------|-------|-------------|
| **Images** | RGB JPG/PNG | Variable | Person images or crops |
| **Bounding Boxes** | Float array | [4] | [x, y, w, h] in pixel coordinates |
| **Keypoints** | Float array | [17, 3] | [x, y, confidence] for each keypoint |
| **Person IDs** | Integer | Scalar | Unique identity label |
| **Camera IDs** | Integer | Scalar | Camera/viewpoint identifier |

#### Supported Dataset Formats

##### 1. PoseTrack21 Format (Recommended)

```
dataset/
├── images/
│   ├── train/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── val/
│       ├── 000101.jpg
│       └── ...
├── annotations/
│   ├── person_train.json
│   ├── person_val.json
│   ├── keypoints_train.json
│   └── keypoints_val.json
└── keypoints/
    ├── train/
    │   ├── 000001.npy  # Shape: (17, 3) - [x, y, confidence]
    │   └── 000002.npy
    └── val/
        └── ...
```

**JSON Annotation Format:**

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000001.jpg",
      "height": 1080,
      "width": 1920
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 150, 300],  // [x, y, w, h]
      "person_id": 42,                // Unique person identity
      "camera_id": 1,                 // Camera identifier
      "keypoints": [                  // 17 keypoints in COCO format
        150, 220, 0.95,    // nose
        140, 210, 0.90,    // left_eye
        160, 210, 0.90,    // right_eye
        // ... 14 more keypoints
      ]
    }
  ]
}
```

##### 2. Market-1501 Style Format

```
dataset/
├── bounding_box_train/     # Training person crops
│   ├── 0001_c1s1_000001_00.jpg
│   ├── 0001_c1s1_000002_00.jpg
│   └── ...
├── bounding_box_test/      # Test person crops
├── gt_query/              # Query images
└── keypoints/             # Keypoint annotations
    ├── 0001_c1s1_000001_00.npy
    └── ...
```

##### 3. Custom Dataset Format

```python
class CustomKPReIDDataset(ImageDataset):
    """
    Custom dataset class for KPReID training
    """
    dataset_dir = "path/to/your/dataset"

    def __init__(self, dataset_dir, mode='train', **kwargs):
        super().__init__(dataset_dir, mode, **kwargs)
        self.dataset_dir = dataset_dir
        self.mode = mode

        # Load your annotations
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """Load your custom annotations"""
        # Implement based on your data format
        annotations = []
        # Parse your annotation files
        return annotations

    def __getitem__(self, index):
        """Return training sample"""
        ann = self.annotations[index]

        # Load image
        img_path = os.path.join(self.dataset_dir, ann['image_path'])
        img = self._load_image(img_path)

        # Load keypoints
        keypoints = self._load_keypoints(ann['keypoints_path'])

        # Extract person crop if needed
        if 'bbox' in ann:
            img = self._extract_person_crop(img, ann['bbox'])
            keypoints = self._adjust_keypoints_to_crop(keypoints, ann['bbox'])

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return {
            'img': img,
            'pid': ann['person_id'],
            'camid': ann['camera_id'],
            'keypoints': keypoints,
            'bbox': ann.get('bbox', None)
        }
```

#### Data Quality Requirements

##### Image Quality

- **Resolution**: Minimum 128x256 pixels, recommended 256x512+
- **Aspect Ratio**: Maintain person aspect ratio (typically 0.5-0.8)
- **Quality**: Clear, well-lit images without heavy blur or compression artifacts
- **Background**: Clean background preferred, but algorithm handles cluttered scenes

##### Keypoint Quality

- **Visibility**: At least 8-10 keypoints should be visible (confidence > 0.5)
- **Accuracy**: Keypoints should align well with anatomical landmarks
- **Consistency**: Same keypoint definitions across all images
- **Coverage**: Full body coverage when possible

##### Annotation Quality

- **Bounding Boxes**: Tight bounding boxes around persons
- **Identity Labels**: Consistent person IDs across cameras/views
- **Camera IDs**: Proper camera/viewpoint labeling for evaluation

#### Data Preprocessing Pipeline

##### Step 1: Raw Data Collection

```python
def collect_raw_data(video_paths, pose_estimator, output_dir):
    """
    Collect raw images and pose annotations from videos
    """
    for video_path in video_paths:
        # Extract frames from video
        frames = extract_frames_from_video(video_path)

        for frame_idx, frame in enumerate(frames):
            # Run pose estimation
            poses = pose_estimator.detect_poses(frame)

            for pose_idx, pose in enumerate(poses):
                # Extract person crop
                bbox = pose['bbox']
                person_crop = extract_person_crop(frame, bbox)

                # Save crop and keypoints
                save_sample(person_crop, pose['keypoints'],
                          f"video_{video_idx}_frame_{frame_idx}_person_{pose_idx}",
                          output_dir)
```

##### Step 2: Data Cleaning and Filtering

```python
def clean_dataset(raw_data_dir, output_dir, quality_thresholds):
    """
    Clean and filter dataset based on quality criteria
    """
    cleaned_samples = []

    for sample_path in glob.glob(f"{raw_data_dir}/*.jpg"):
        # Load sample data
        img = cv2.imread(sample_path)
        keypoints_path = sample_path.replace('.jpg', '_keypoints.npy')
        keypoints = np.load(keypoints_path)

        # Quality checks
        if not check_image_quality(img, quality_thresholds):
            continue

        if not check_keypoint_quality(keypoints, quality_thresholds):
            continue

        if not check_pose_completeness(keypoints):
            continue

        # Copy to cleaned dataset
        shutil.copy2(sample_path, output_dir)
        shutil.copy2(keypoints_path, output_dir)

        cleaned_samples.append({
            'image_path': os.path.basename(sample_path),
            'keypoints_path': os.path.basename(keypoints_path),
            'quality_score': compute_quality_score(img, keypoints)
        })

    return cleaned_samples

def check_image_quality(img, thresholds):
    """Check image quality metrics"""
    # Blur detection
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    if blur_score < thresholds['min_blur']:
        return False

    # Brightness check
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if not (thresholds['min_brightness'] <= brightness <= thresholds['max_brightness']):
        return False

    # Contrast check
    contrast = img.std()
    if contrast < thresholds['min_contrast']:
        return False

    return True

def check_keypoint_quality(keypoints, thresholds):
    """Check keypoint quality"""
    # Visibility check
    visible_kpts = np.sum(keypoints[:, 2] > thresholds['visibility_thresh'])
    if visible_kpts < thresholds['min_visible_kpts']:
        return False

    # Confidence distribution check
    mean_conf = np.mean(keypoints[:, 2])
    if mean_conf < thresholds['min_mean_confidence']:
        return False

    return True

def check_pose_completeness(keypoints):
    """Check if pose covers required body parts"""
    # Check for essential keypoints
    essential_kpts = [0, 1, 2, 5, 6, 11, 12]  # nose, eyes, shoulders, hips
    essential_visible = np.sum(keypoints[essential_kpts, 2] > 0.5)

    return essential_visible >= len(essential_kpts) * 0.6  # 60% of essential keypoints
```

##### Step 3: Data Augmentation

```python
def augment_training_data(dataset, augmentation_config):
    """
    Apply data augmentation to increase dataset diversity
    """
    augmented_samples = []

    for sample in dataset:
        img = sample['img']
        keypoints = sample['keypoints']

        # Original sample
        augmented_samples.append(sample)

        # Geometric augmentations
        for aug_type in augmentation_config['geometric']:
            if aug_type == 'flip':
                aug_img, aug_kpts = horizontal_flip(img, keypoints)
                augmented_samples.append({
                    **sample,
                    'img': aug_img,
                    'keypoints': aug_kpts
                })

            elif aug_type == 'rotation':
                for angle in [-15, -10, -5, 5, 10, 15]:
                    aug_img, aug_kpts = rotate_image(img, keypoints, angle)
                    augmented_samples.append({
                        **sample,
                        'img': aug_img,
                        'keypoints': aug_kpts
                    })

        # Appearance augmentations
        for aug_type in augmentation_config['appearance']:
            if aug_type == 'color_jitter':
                aug_img = apply_color_jitter(img)
                augmented_samples.append({
                    **sample,
                    'img': aug_img
                })

            elif aug_type == 'motion_blur':
                aug_img = apply_motion_blur(img)
                augmented_samples.append({
                    **sample,
                    'img': aug_img
                })

    return augmented_samples
```

##### Step 4: Train/Validation/Test Split

```python
def create_data_split(dataset, split_config):
    """
    Create train/val/test splits with proper person distribution
    """
    # Group by person ID
    person_groups = {}
    for sample in dataset:
        pid = sample['pid']
        if pid not in person_groups:
            person_groups[pid] = []
        person_groups[pid].append(sample)

    # Split persons into train/val/test
    person_ids = list(person_groups.keys())
    np.random.shuffle(person_ids)

    n_train = int(len(person_ids) * split_config['train_ratio'])
    n_val = int(len(person_ids) * split_config['val_ratio'])

    train_persons = person_ids[:n_train]
    val_persons = person_ids[n_train:n_train + n_val]
    test_persons = person_ids[n_train + n_val:]

    # Create splits
    splits = {
        'train': [sample for pid in train_persons for sample in person_groups[pid]],
        'val': [sample for pid in val_persons for sample in person_groups[pid]],
        'test': [sample for pid in test_persons for sample in person_groups[pid]]
    }

    return splits
```

#### Data Statistics and Analysis

##### Dataset Analysis Script

```python
def analyze_dataset(dataset_dir):
    """
    Analyze dataset statistics and quality metrics
    """
    stats = {
        'total_samples': 0,
        'unique_persons': set(),
        'camera_distribution': {},
        'keypoint_visibility': [],
        'image_quality': [],
        'bbox_sizes': []
    }

    for sample_path in glob.glob(f"{dataset_dir}/*.jpg"):
        stats['total_samples'] += 1

        # Load data
        img = cv2.imread(sample_path)
        keypoints_path = sample_path.replace('.jpg', '_keypoints.npy')
        keypoints = np.load(keypoints_path)

        # Extract metadata from filename (assuming standard naming)
        filename_parts = os.path.basename(sample_path).split('_')
        person_id = int(filename_parts[0])
        camera_id = int(filename_parts[1])

        stats['unique_persons'].add(person_id)

        if camera_id not in stats['camera_distribution']:
            stats['camera_distribution'][camera_id] = 0
        stats['camera_distribution'][camera_id] += 1

        # Keypoint statistics
        visibility = np.mean(keypoints[:, 2])
        stats['keypoint_visibility'].append(visibility)

        # Image quality
        quality = compute_image_quality(img)
        stats['image_quality'].append(quality)

        # Bounding box size (if available)
        # bbox_size = compute_bbox_size(bbox)
        # stats['bbox_sizes'].append(bbox_size)

    # Print statistics
    print(f"Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Unique persons: {len(stats['unique_persons'])}")
    print(f"  Cameras: {len(stats['camera_distribution'])}")
    print(f"  Avg keypoint visibility: {np.mean(stats['keypoint_visibility']):.3f}")
    print(f"  Avg image quality: {np.mean(stats['image_quality']):.3f}")

    return stats
```

#### Data Loading and PyTorch Integration

##### Custom DataLoader

```python
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

class KPReIDDataLoader:
    """
    Custom data loader for KPReID training
    """

    def __init__(self, dataset_dir, batch_size=32, num_workers=4):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transforms
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_train_loader(self):
        """Get training data loader"""
        train_dataset = KPReIDDataset(
            self.dataset_dir,
            mode='train',
            transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        return train_loader

    def get_val_loader(self):
        """Get validation data loader"""
        val_transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = KPReIDDataset(
            self.dataset_dir,
            mode='val',
            transform=val_transform
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return val_loader
```

#### Memory-Efficient Data Loading

```python
class MemoryEfficientKPReIDDataset(Dataset):
    """
    Memory-efficient dataset for large-scale training
    """

    def __init__(self, data_list, transform=None):
        self.data_list = data_list  # List of file paths/metadata
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Load data on-demand to save memory
        sample_info = self.data_list[idx]

        # Load image
        img = cv2.imread(sample_info['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load keypoints
        keypoints = np.load(sample_info['keypoints_path'])

        # Apply transformations
        if self.transform:
            # Custom transform that handles keypoints
            img, keypoints = self.transform(img, keypoints)

        return {
            'img': img,
            'keypoints': keypoints,
            'pid': sample_info['person_id'],
            'camid': sample_info['camera_id']
        }
```

#### Data Quality Validation

##### Pre-training Validation

```python
def validate_training_data(data_loader, quality_checks):
    """
    Validate data quality before training
    """
    issues = []

    for batch in tqdm(data_loader):
        for i in range(len(batch['img'])):
            img = batch['img'][i]
            keypoints = batch['keypoints'][i]

            # Check keypoint visibility
            visible_count = torch.sum(keypoints[:, 2] > 0.5)
            if visible_count < quality_checks['min_visible_kpts']:
                issues.append(f"Sample {i}: Low keypoint visibility ({visible_count})")

            # Check image quality
            if is_blurry(img):
                issues.append(f"Sample {i}: Blurry image")

            # Check pose validity
            if not is_valid_pose(keypoints):
                issues.append(f"Sample {i}: Invalid pose configuration")

    return issues

def is_blurry(image, threshold=100):
    """Check if image is blurry"""
    gray = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score < threshold

def is_valid_pose(keypoints, min_distance=10):
    """Check if pose keypoints form a valid human pose"""
    # Check shoulder width
    left_shoulder = keypoints[5][:2]   # COCO keypoint indices
    right_shoulder = keypoints[6][:2]
    shoulder_distance = torch.norm(left_shoulder - right_shoulder)

    if shoulder_distance < min_distance:
        return False

    # Check hip width
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]
    hip_distance = torch.norm(left_hip - right_hip)

    if hip_distance < min_distance:
        return False

    return True
```

This comprehensive data preparation guide ensures you have high-quality, well-formatted data for training KPReID models. The pipeline includes collection, cleaning, augmentation, and validation steps to maximize training effectiveness.

### Training Configurations

#### Base Training Configuration

```yaml
# kpreid_training_config.yaml
model:
  name: resnet50_ibn_a
  pretrained: true
  kpr:
    enabled: true
    keypoints:
      enabled: true
      prompt_masks: keypoints_gaussian
      target_masks: keypoints_gaussian
      vis_thresh: 0.5
      vis_continous: true
    masks:
      preprocess: coco_to_six_body_masks
      softmax_weight: 1.0

data:
  sources: [posetrack21]
  targets: [posetrack21]
  height: 256
  width: 128
  transforms: [random_flip, random_crop, random_erase]
  norm_mean: [0.485, 0.456, 0.406]
  norm_std: [0.229, 0.224, 0.225]

loss:
  name: triplet
  margin: 0.3
  weight: 1.0

train:
  optim: adam
  lr: 0.0003
  weight_decay: 0.0005
  lr_scheduler: warmup_multi_step
  warmup_epochs: 10
  milestones: [20, 40]
  gamma: 0.1
  max_epoch: 60
  batch_size: 32
  workers: 4

test:
  batch_size: 32
  dist_metric: cosine
  normalize_feature: true
  evaluate: true
  eval_freq: 10
  rerank: true
```

#### Advanced Training Strategies

##### Multi-Stage Training

```python
def multi_stage_training(cfg):
    """
    Multi-stage training for better convergence
    """
    # Stage 1: Train backbone with frozen pose processing
    cfg.model.kpr.keypoints.enabled = False
    train_stage(cfg, epochs=20, lr=0.001)
    
    # Stage 2: Fine-tune with pose processing enabled
    cfg.model.kpr.keypoints.enabled = True
    cfg.train.lr = 0.0003
    train_stage(cfg, epochs=40, lr=0.0003)
    
    # Stage 3: Final fine-tuning with lower learning rate
    cfg.train.lr = 0.0001
    train_stage(cfg, epochs=20, lr=0.0001)
```

##### Curriculum Learning

```python
def curriculum_training(cfg, difficulty_levels=3):
    """
    Gradually increase training difficulty
    """
    for level in range(difficulty_levels):
        # Adjust visibility threshold (start easy, get harder)
        cfg.model.kpr.keypoints.vis_thresh = 0.3 + level * 0.2
        
        # Adjust data augmentation intensity
        cfg.data.transforms = get_transforms_for_level(level)
        
        # Train for several epochs at this difficulty
        train_stage(cfg, epochs=15)
```

### Fine-tuning Pretrained Models

#### Domain Adaptation Fine-tuning

```python
def fine_tune_for_domain(cfg, target_dataset, pretrained_path):
    """
    Fine-tune pretrained KPReID model on target domain
    """
    # 1. Load pretrained model
    model = load_pretrained_model(pretrained_path)
    
    # 2. Modify configuration for target domain
    cfg.data.sources = [target_dataset]
    cfg.data.targets = [target_dataset]
    
    # 3. Use lower learning rate for fine-tuning
    cfg.train.lr = 0.0001
    cfg.train.max_epoch = 30
    
    # 4. Enable gradual unfreezing
    cfg.train.freeze_backbone = True  # Initially freeze backbone
    
    # 5. Start fine-tuning
    model.train(target_dataset)
    
    # 6. Unfreeze backbone for final epochs
    cfg.train.freeze_backbone = False
    cfg.train.lr = 0.00005
    model.train(target_dataset, additional_epochs=10)
```

#### Sports-Specific Fine-tuning

```python
def fine_tune_for_sports(cfg, sports_dataset):
    """
    Fine-tune for sports-specific scenarios (soccer, basketball, etc.)
    """
    # Adjust keypoint processing for sports poses
    cfg.model.kpr.keypoints.prompt_masks = 'body_parts'
    cfg.model.kpr.masks.preprocess = 'sports_pose_masks'
    
    # Use sports-specific data augmentation
    cfg.data.transforms = [
        'random_flip',
        'random_rotation_sports',  # Sports-specific rotations
        'motion_blur',             # Simulate motion
        'occlusion_sports'         # Sports-specific occlusions
    ]
    
    # Fine-tune with domain-specific settings
    fine_tune_for_domain(cfg, sports_dataset, 'pretrained_kpreid.pth')
```

### Hyperparameter Optimization

#### Key Hyperparameters to Tune

| Parameter | Recommended Range | Description |
|-----------|------------------|-------------|
| `lr` | 0.0001 - 0.001 | Learning rate (lower for fine-tuning) |
| `batch_size` | 16 - 64 | Batch size (depends on GPU memory) |
| `margin` | 0.2 - 0.5 | Triplet loss margin |
| `vis_thresh` | 0.3 - 0.7 | Keypoint visibility threshold |
| `weight_decay` | 0.0001 - 0.001 | L2 regularization strength |

#### Automated Hyperparameter Search

```python
def hyperparameter_search(cfg, param_grid, dataset):
    """
    Perform grid search for optimal hyperparameters
    """
    best_score = 0
    best_params = {}
    
    for lr in param_grid['lr']:
        for margin in param_grid['margin']:
            for vis_thresh in param_grid['vis_thresh']:
                # Update configuration
                cfg.train.lr = lr
                cfg.loss.margin = margin
                cfg.model.kpr.keypoints.vis_thresh = vis_thresh
                
                # Train and evaluate
                model = train_model(cfg, dataset)
                score = evaluate_model(model, dataset)
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'lr': lr,
                        'margin': margin,
                        'vis_thresh': vis_thresh
                    }
    
    return best_params, best_score
```

### Training Best Practices

#### Data Quality Checks

```python
def validate_training_data(dataset):
    """
    Validate training data quality before training
    """
    issues = []
    
    for sample in dataset:
        img, pid, keypoints = sample
        
        # Check keypoint visibility
        visible_kpts = np.sum(keypoints[:, 2] > 0.5)
        if visible_kpts < 8:  # Less than half keypoints visible
            issues.append(f"Low visibility for person {pid}")
        
        # Check keypoint distribution
        if not keypoints_in_bbox(keypoints, sample['bbox']):
            issues.append(f"Keypoints outside bbox for person {pid}")
        
        # Check image quality
        if is_blurry(img):
            issues.append(f"Blurry image for person {pid}")
    
    return issues
```

#### Monitoring Training Progress

```python
def setup_training_monitoring(cfg):
    """
    Setup comprehensive training monitoring
    """
    # TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=cfg.train.log_dir)
    
    # Custom metrics to track
    metrics = {
        'train_loss': [],
        'val_mAP': [],
        'val_rank1': [],
        'keypoint_visibility': [],
        'mask_quality': []
    }
    
    return writer, metrics

def log_training_metrics(writer, metrics, epoch):
    """
    Log training metrics for monitoring
    """
    writer.add_scalar('Loss/train', metrics['train_loss'][-1], epoch)
    writer.add_scalar('Accuracy/val_mAP', metrics['val_mAP'][-1], epoch)
    writer.add_scalar('Accuracy/val_rank1', metrics['val_rank1'][-1], epoch)
    writer.add_scalar('Data/keypoint_visibility', metrics['keypoint_visibility'][-1], epoch)
```

#### Handling Training Issues

**Common Issues and Solutions:**

1. **Overfitting**:
   - Increase data augmentation
   - Add dropout or regularization
   - Use early stopping

2. **Poor Convergence**:
   - Reduce learning rate
   - Use learning rate scheduling
   - Check data preprocessing

3. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable gradient checkpointing

4. **Low ReID Accuracy**:
   - Verify keypoint quality
   - Adjust visibility thresholds
   - Check mask generation parameters

### Evaluation and Validation

#### Training Evaluation Script

```python
def evaluate_training_progress(model, val_dataset, cfg):
    """
    Comprehensive evaluation during training
    """
    # Standard ReID metrics
    rank1, rank5, mAP = evaluate_reid_metrics(model, val_dataset)
    
    # Pose-specific metrics
    pose_accuracy = evaluate_pose_accuracy(model, val_dataset)
    mask_quality = evaluate_mask_quality(model, val_dataset)
    
    # Keypoint visibility analysis
    visibility_stats = analyze_keypoint_visibility(val_dataset)
    
    results = {
        'rank1': rank1,
        'rank5': rank5,
        'mAP': mAP,
        'pose_accuracy': pose_accuracy,
        'mask_quality': mask_quality,
        'visibility_stats': visibility_stats
    }
    
    return results
```

#### Model Checkpointing

```python
def save_best_model(model, metrics, epoch, save_path):
    """
    Save model checkpoints based on validation performance
    """
    # Save based on mAP
    if metrics['mAP'] > best_mAP:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': cfg
        }, f"{save_path}/best_model_mAP_{metrics['mAP']:.3f}.pth")
    
    # Save based on rank-1 accuracy
    if metrics['rank1'] > best_rank1:
        torch.save(model.state_dict(), f"{save_path}/best_model_rank1.pth")
    
    # Regular checkpointing
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{save_path}/checkpoint_epoch_{epoch}.pth")
```

### Deployment After Training

#### Exporting Trained Models

```python
def export_trained_model(model_path, output_path, input_size=(256, 128)):
    """
    Export trained model for inference
    """
    # Load trained model
    model = load_model(model_path)
    model.eval()
    
    # Create example input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    dummy_keypoints = torch.randn(1, 17, 3)
    
    # Export to TorchScript
    scripted_model = torch.jit.trace(model, (dummy_input, dummy_keypoints))
    scripted_model.save(f"{output_path}/kpreid_model.pt")
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        (dummy_input, dummy_keypoints),
        f"{output_path}/kpreid_model.onnx",
        opset_version=11,
        input_names=['input_image', 'keypoints'],
        output_names=['embeddings', 'visibility_scores']
    )
```

This comprehensive training guide provides everything you need to train and fine-tune KPReID models effectively. Start with the basic training script and gradually incorporate advanced techniques as needed.

### Performance Optimizations

#### Batch Processing

```python
def batch_process_detections(detections_batch, model):
    """
    Optimized batch processing for multiple detections
    
    Args:
        detections_batch: List of detection dictionaries
        model: KPReID model
    """
    # Pre-compute all heatmaps
    batch_heatmaps = []
    batch_masks = []
    
    for detection in detections_batch:
        keypoints = detection['keypoints']
        vis_scores = detection['visibility_scores']
        
        heatmaps = preprocess_keypoints(keypoints, vis_scores, heatmap_size)
        prompt_mask, target_mask = generate_pose_masks(heatmaps)
        
        batch_heatmaps.append(heatmaps)
        batch_masks.append((prompt_mask, target_mask))
    
    # Batch tensor operations
    batch_images = torch.stack([d['image'] for d in detections_batch])
    batch_prompt_masks = torch.stack([m[0] for m in batch_masks])
    batch_target_masks = torch.stack([m[1] for m in batch_masks])
    
    # Forward pass
    features = model.backbone(batch_images)
    attended_features = features * batch_prompt_masks.unsqueeze(1)
    
    # Generate embeddings
    embeddings = model.embedding_head(attended_features.mean(dim=(-2, -1)))
    embeddings = embeddings * batch_target_masks.mean(dim=(-2, -1), keepdim=True)
    
    return F.normalize(embeddings, p=2, dim=-1)
```

## Features

The algorithm employs a dual-mask strategy:

- **Prompt Masks**: Used during feature extraction to guide attention
- **Target Masks**: Applied to the final embeddings for pose-aware refinement

```python
def generate_pose_masks(heatmaps, mode='gaussian'):
    """
    Generate pose-aware masks from keypoint heatmaps

    Args:
        heatmaps: (17, H, W) keypoint heatmaps
        mode: Mask generation strategy ('gaussian', 'body_parts', 'skeleton')

    Returns:
        prompt_mask: (H, W) attention mask for feature extraction
        target_mask: (H, W) refinement mask for embeddings
    """
    if mode == 'gaussian':
        # Combine all keypoints with equal weighting
        combined = np.sum(heatmaps, axis=0)
        prompt_mask = normalize(combined)
        target_mask = prompt_mask
    elif mode == 'body_parts':
        # Group keypoints by body parts (head, torso, limbs)
        head_mask = combine_keypoints(heatmaps, [0, 1, 2, 3, 4])  # face keypoints
        torso_mask = combine_keypoints(heatmaps, [5, 6, 11, 12])  # shoulder/hip
        limbs_mask = combine_keypoints(heatmaps, [7, 8, 9, 10, 13, 14, 15, 16])

        prompt_mask = 0.4 * head_mask + 0.4 * torso_mask + 0.2 * limbs_mask
        target_mask = torso_mask  # Focus on torso for identity
    elif mode == 'skeleton':
        # Generate skeleton-based mask using keypoint connections
        skeleton_mask = generate_skeleton_mask(heatmaps, connections)
        prompt_mask = skeleton_mask
        target_mask = skeleton_mask

    return prompt_mask, target_mask
```

### Mask-Guided Feature Extraction

```python
def extract_features_with_masks(image, prompt_mask, backbone):
    """
    Extract features using mask-guided attention
    
    Args:
        image: (3, H, W) input image tensor
        prompt_mask: (H, W) attention mask
        backbone: ResNet-IBN-A model
    
    Returns:
        features: (C, H', W') feature maps
    """
    # Extract base features
    base_features = backbone(image)  # (C, H, W)
    
    # Apply mask attention
    mask_tensor = torch.tensor(prompt_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_tensor = F.interpolate(mask_tensor, size=base_features.shape[-2:], mode='bilinear')
    
    attended_features = base_features * mask_tensor.expand_as(base_features)
    
    return attended_features
```

#### Step 4: Embedding Generation and Refinement
```python
def generate_embedding(features, target_mask, embedding_head):
    """
    Generate final embedding with target mask refinement
    
    Args:
        features: (C, H, W) attended feature maps
        target_mask: (H, W) refinement mask
        embedding_head: FC layer for embedding generation
    
    Returns:
        embedding: (D,) feature vector
        visibility_scores: (17,) keypoint visibility scores
    """
    # Global average pooling with mask weighting
    mask_tensor = torch.tensor(target_mask).unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(mask_tensor, size=features.shape[-2:], mode='bilinear')
    
    pooled_features = torch.sum(features * mask_tensor, dim=(-2, -1))
    pooled_features = pooled_features / (torch.sum(mask_tensor) + 1e-8)
    
    # Generate embedding
    embedding = embedding_head(pooled_features)
    embedding = F.normalize(embedding, p=2, dim=-1)
    
    # Compute visibility scores from original keypoints
    visibility_scores = compute_visibility_scores(original_keypoints)
    
    return embedding, visibility_scores
```

### Key Algorithm Components

#### Gaussian Kernel Generation
The gaussian heatmap generation uses an anisotropic gaussian kernel:
```python
def gaussian_kernel(center, sigma=2.0, size=(64, 32), aspect_ratio=2.0):
    """
    Generate 2D gaussian kernel for keypoint heatmap
    
    Args:
        center: (x, y) keypoint coordinates
        sigma: standard deviation
        size: (H, W) output size
        aspect_ratio: elongation factor for pose-aware kernels
    """
    x, y = center
    H, W = size
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    
    # Anisotropic gaussian for pose keypoints
    sigma_x = sigma
    sigma_y = sigma * aspect_ratio
    
    # Compute gaussian values
    exponent = -0.5 * (
        ((x_grid - x) / sigma_x) ** 2 + 
        ((y_grid - y) / sigma_y) ** 2
    )
    
    heatmap = np.exp(exponent)
    return heatmap
```

#### Visibility Score Computation
```python
def compute_visibility_scores(keypoints, bbox, image_size):
    """
    Compute refined visibility scores considering bbox and image boundaries
    
    Args:
        keypoints: (17, 3) array with (x, y, confidence)
        bbox: (x, y, w, h) bounding box
        image_size: (H, W) image dimensions
    """
    scores = []
    for kp in keypoints:
        x, y, conf = kp
        
        # Boundary check
        if not (0 <= x < image_size[1] and 0 <= y < image_size[0]):
            scores.append(0.0)
            continue
            
        # BBox containment check
        bbox_left, bbox_top = bbox[0], bbox[1]
        bbox_right = bbox_left + bbox[2]
        bbox_bottom = bbox_top + bbox[3]
        
        if not (bbox_left <= x <= bbox_right and bbox_top <= y <= bbox_bottom):
            scores.append(conf * 0.5)  # Reduce score for keypoints outside bbox
        else:
            scores.append(conf)
    
    return np.array(scores)
```

#### Multi-Scale Processing
```python
def multi_scale_processing(image, keypoints, scales=[0.5, 1.0, 1.5]):
    """
    Process image at multiple scales for robust feature extraction
    
    Args:
        image: Input image tensor
        keypoints: Keypoint coordinates
        scales: List of scale factors
    """
    embeddings = []
    
    for scale in scales:
        # Scale image
        scaled_image = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode='bilinear')
        scaled_image = scaled_image.squeeze(0)
        
        # Scale keypoints accordingly
        scaled_keypoints = keypoints * scale
        
        # Generate masks and extract features
        heatmaps = preprocess_keypoints(scaled_keypoints, vis_scores, scaled_image.shape[-2:])
        prompt_mask, target_mask = generate_pose_masks(heatmaps)
        
        features = extract_features_with_masks(scaled_image, prompt_mask, backbone)
        embedding, _ = generate_embedding(features, target_mask, embedding_head)
        
        embeddings.append(embedding)
    
    # Fuse multi-scale embeddings
    fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return F.normalize(fused_embedding, p=2, dim=-1)
```

### Loss Functions and Training

#### Triplet Loss with Pose Awareness
```python
def pose_aware_triplet_loss(anchor_emb, positive_emb, negative_emb, 
                           anchor_mask, positive_mask, negative_mask):
    """
    Triplet loss weighted by pose mask similarity
    
    Args:
        anchor_emb, positive_emb, negative_emb: Embeddings
        anchor_mask, positive_mask, negative_mask: Pose masks
    """
    # Compute embedding distances
    pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
    neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
    
    # Compute mask similarities
    mask_sim_pos = F.cosine_similarity(anchor_mask.flatten(), positive_mask.flatten(), dim=0)
    mask_sim_neg = F.cosine_similarity(anchor_mask.flatten(), negative_mask.flatten(), dim=0)
    
    # Weighted triplet loss
    margin = 0.3
    loss = torch.relu(pos_dist - neg_dist + margin)
    
    # Boost loss for dissimilar poses
    pose_weight = 1.0 + 0.5 * (1.0 - mask_sim_pos)
    loss = loss * pose_weight
    
    return loss.mean()
```

#### Visibility-Aware Loss
```python
def visibility_aware_loss(embeddings, visibility_scores, targets):
    """
    Loss function that considers keypoint visibility
    
    Args:
        embeddings: Batch of embeddings
        visibility_scores: (B, 17) visibility scores
        targets: Ground truth identity labels
    """
    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(embeddings, targets)
    
    # Visibility regularization
    mean_visibility = torch.mean(visibility_scores, dim=-1)
    visibility_reg = torch.mean((mean_visibility - 0.8) ** 2)  # Encourage high visibility
    
    total_loss = ce_loss + 0.1 * visibility_reg
    return total_loss
```

### Performance Optimizations

#### Batch Processing
```python
def batch_process_detections(detections_batch, model):
    """
    Optimized batch processing for multiple detections
    
    Args:
        detections_batch: List of detection dictionaries
        model: KPReID model
    """
    # Pre-compute all heatmaps
    batch_heatmaps = []
    batch_masks = []
    
    for detection in detections_batch:
        keypoints = detection['keypoints']
        vis_scores = detection['visibility_scores']
        
        heatmaps = preprocess_keypoints(keypoints, vis_scores, heatmap_size)
        prompt_mask, target_mask = generate_pose_masks(heatmaps)
        
        batch_heatmaps.append(heatmaps)
        batch_masks.append((prompt_mask, target_mask))
    
    # Batch tensor operations
    batch_images = torch.stack([d['image'] for d in detections_batch])
    batch_prompt_masks = torch.stack([m[0] for m in batch_masks])
    batch_target_masks = torch.stack([m[1] for m in batch_masks])
    
    # Forward pass
    features = model.backbone(batch_images)
    attended_features = features * batch_prompt_masks.unsqueeze(1)
    
    # Generate embeddings
    embeddings = model.embedding_head(attended_features.mean(dim=(-2, -1)))
    embeddings = embeddings * batch_target_masks.mean(dim=(-2, -1), keepdim=True)
    
    return F.normalize(embeddings, p=2, dim=-1)
```

