# TrackLab: High-Level Design Document

## 1. Overview

TrackLab is a modular, research-oriented framework for multi-object tracking and pose estimation. It provides a unified pipeline for processing video data through detection, tracking, re-identification, and evaluation stages, with particular emphasis on sports analytics and game state reconstruction.

## 2. Architecture Overview

### 2.1 Core Design Principles

- **Modularity**: Pluggable components for easy experimentation
- **Data-Centric**: Structured data flow using pandas DataFrames
- **Hierarchical Processing**: Video → Frame → Detection processing levels
- **Configuration-Driven**: Hydra-based declarative configuration
- **Research-Friendly**: Easy debugging, state persistence, and evaluation

### 2.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrackLab Framework                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Datasets   │ │  Modules   │ │   Configuration     │   │
│  │             │ │            │ │                     │   │
│  │ • SoccerNet │ │ • Detectors│ │ • Hydra Config      │   │
│  │ • MOT17     │ │ • Trackers │ │ • YAML Files        │   │
│  │ • DanceTrack│ │ • ReID     │ │ • Command Line      │   │
│  │ • ...       │ │ • ...      │ │ • Overrides         │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │Data Pipeline│ │ Execution  │ │   State Management  │   │
│  │             │ │ Engine     │ │                     │   │
│  │ • TrackingSet│ │ • Offline │ │ • TrackerState      │   │
│  │ • DataFrames │ │ • Online  │ │ • Persistence       │   │
│  │ • Hierarchy  │ │ • Batching│ │ • Checkpointing     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Evaluation  │ │ Callbacks  │ │   Visualization     │   │
│  │             │ │            │ │                     │   │
│  │ • TrackEval │ │ • Logging  │ │ • Video Output      │   │
│  │ • GS-HOTA   │ │ • Progress │ │ • Debug Info        │   │
│  │ • Custom    │ │ • Metrics  │ │ • Real-time         │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 3. Data Architecture

### 3.1 Hierarchical Data Model

TrackLab implements a three-level hierarchical data structure:

#### Level 1: Video Metadata
```python
video_metadatas = pd.DataFrame({
    'id': int,                    # Unique video identifier
    'name': str,                  # Video name/filename
    'nframes': int,               # Number of frames
    'frame_rate': int,            # FPS
    'im_width': int,              # Video width
    'im_height': int,             # Video height
    # ... dataset-specific metadata
})
```

#### Level 2: Image/Frame Metadata
```python
image_metadatas = pd.DataFrame({
    'id': int,                    # Unique frame identifier
    'video_id': int,              # Foreign key to video
    'frame': int,                 # Frame number (0-based)
    'file_path': str,             # Path to image file
    # ... additional frame metadata
})
```

#### Level 3: Detection Data
```python
detections = pd.DataFrame({
    'id': int,                    # Unique detection identifier
    'image_id': int,              # Foreign key to frame
    'video_id': int,              # Foreign key to video
    'track_id': int,              # Tracking identifier (assigned by tracker)
    'bbox_ltwh': np.array,        # Bounding box [left, top, width, height]
    'bbox_conf': float,           # Detection confidence
    'category_id': int,           # Object category
    # ... additional detection features
})
```

### 3.2 Data Flow Relationships

```
Video (id=1)
├── Frame 1 (id=1, video_id=1)
│   ├── Detection A (id=1, image_id=1, video_id=1, track_id=1)
│   └── Detection B (id=2, image_id=1, video_id=1, track_id=2)
├── Frame 2 (id=2, video_id=1)
│   ├── Detection C (id=3, image_id=2, video_id=1, track_id=1)  # Same object as A
│   └── Detection D (id=4, image_id=2, video_id=1, track_id=2)  # Same object as B
└── ...
```

## 4. Module System

### 4.1 Module Types

TrackLab defines three types of processing modules based on data granularity:

#### 4.1.1 ImageLevelModule
- **Scope**: Processes entire images/frames
- **Examples**: Object detectors, bottom-up pose estimators
- **Input**: Full image + existing detections
- **Output**: New detections/features for the image

#### 4.1.2 DetectionLevelModule
- **Scope**: Processes individual detections
- **Examples**: Re-ID models, top-down pose estimators
- **Input**: Single detection + cropped image region
- **Output**: Enhanced detection features

#### 4.1.3 VideoLevelModule
- **Scope**: Processes complete video sequences
- **Examples**: Offline trackers, tracklet aggregators
- **Input**: All detections across video
- **Output**: Track assignments, aggregated attributes

### 4.2 Module Interface

All modules implement a consistent interface:

```python
class BaseModule:
    input_columns: List[str]      # Required input data columns
    output_columns: List[str]     # Produced output data columns
    training_enabled: bool        # Whether module supports training

    def preprocess(self, image, detections, metadata):
        # Prepare data for processing
        pass

    def process(self, batch, detections, metadatas):
        # Main processing logic
        pass
```

### 4.3 Pipeline Composition

Modules are composed into processing pipelines:

```python
pipeline = Pipeline([
    bbox_detector,     # ImageLevelModule
    reid_model,        # DetectionLevelModule
    tracker,          # VideoLevelModule
    pose_estimator,   # DetectionLevelModule
    team_classifier   # VideoLevelModule
])
```

## 5. Execution Engine

### 5.1 Engine Types

#### 5.1.1 OfflineTrackingEngine
- **Strategy**: Process modules sequentially on entire videos
- **Advantages**: Maximum GPU utilization, batch processing
- **Use Case**: Research, evaluation, offline processing

#### 5.1.2 OnlineTrackingEngine
- **Strategy**: Process frame-by-frame in real-time
- **Advantages**: Real-time processing, streaming
- **Use Case**: Live video processing, deployment

### 5.2 Execution Flow

```
1. Dataset Loading
   ├── Load video metadata
   ├── Load image metadata
   └── Load ground truth detections

2. Video Processing Loop
   ├── For each video:
   │   ├── Load video data into TrackerState
   │   ├── Execute pipeline modules
   │   ├── Update TrackerState
   │   └── Save results

3. Evaluation
   ├── Compute metrics
   ├── Generate reports
   └── Save visualizations
```

### 5.3 Data Batching Strategy

```python
# Image-level processing (e.g., detection)
for batch in dataloader:  # Batch of images
    detections = detector.process(batch)

# Detection-level processing (e.g., ReID)
for batch in dataloader:  # Batch of detections
    embeddings = reid_model.process(batch)

# Video-level processing (e.g., tracking)
tracks = tracker.process(all_detections, all_images)
```

## 6. State Management

### 6.1 TrackerState Class

The TrackerState manages the current processing state:

```python
class TrackerState:
    def __init__(self, tracking_set, pipeline, save_file=None, load_file=None):
        self.detections_pred = pd.DataFrame()  # Predicted detections
        self.image_pred = pd.DataFrame()       # Predicted image features
        self.pipeline = pipeline               # Current processing pipeline

    def __enter__(self):
        # Load state from file or initialize
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save state to file
        pass
```

### 6.2 State Persistence

```python
# Save intermediate results
tracker_state.save_file = "states/experiment.pklz"

# Load from checkpoint
tracker_state.load_file = "states/experiment.pklz"

# Resume processing from any pipeline stage
pipeline = Pipeline([tracker])  # Skip detector, reid
```

## 7. Configuration System

### 7.1 Hydra Configuration

TrackLab uses Hydra for declarative configuration:

```yaml
# config.yaml
defaults:
  - dataset: soccernet
  - modules/bbox_detector: yolo_ultralytics
  - modules/tracker: strongsort

pipeline:
  - bbox_detector
  - reid
  - track

dataset:
  nvid: 10  # For development
  eval_set: "valid"

modules:
  bbox_detector:
    batch_size: 4
  reid:
    batch_size: 32
```

### 7.2 Configuration Overrides

```bash
# Command line overrides
tracklab dataset.nvid=5 modules.bbox_detector.batch_size=8

# Configuration groups
tracklab +modules/bbox_detector=rtmdet +modules/tracker=ocsort
```

## 8. Evaluation Framework

### 8.1 Evaluator Interface

```python
class Evaluator(ABC):
    @abstractmethod
    def run(self, tracker_state: TrackerState):
        # Compute evaluation metrics
        pass
```

### 8.2 Supported Metrics

- **TrackEval**: MOTA, MOTP, IDF1, HOTA, etc.
- **GS-HOTA**: Game State HOTA for soccer
- **PoseTrack**: Pose tracking metrics
- **Custom**: User-defined metrics

## 9. Callback System

### 9.1 Event-Driven Architecture

```python
class Callback:
    def on_video_loop_start(self, video_metadata, video_idx):
        # Called before processing each video
        pass

    def on_module_step_end(self, task, batch, detections):
        # Called after each module processes a batch
        pass

    def on_video_loop_end(self, detections, image_pred):
        # Called after processing each video
        pass
```

### 9.2 Built-in Callbacks

- **ProgressCallback**: Progress bars and logging
- **VisualizationCallback**: Generate video outputs
- **WandbCallback**: Experiment tracking
- **DebugCallback**: Detailed debugging information

## 10. Key Design Decisions

### 10.1 Data-Centric Approach

**Decision**: Use pandas DataFrames as core data structure
**Rationale**:
- Efficient vectorized operations
- SQL-like querying capabilities
- Easy serialization and persistence
- Familiar interface for data scientists

### 10.2 Modular Pipeline Design

**Decision**: Decompose tracking into independent modules
**Rationale**:
- Easy experimentation with different combinations
- Reusable components across projects
- Clear separation of concerns
- Simplified debugging and testing

### 10.3 Hierarchical Processing

**Decision**: Process data at appropriate granularity levels
**Rationale**:
- Image-level: Whole scene understanding
- Detection-level: Individual object processing
- Video-level: Temporal reasoning and aggregation

### 10.4 State Persistence

**Decision**: Enable saving/loading of intermediate results
**Rationale**:
- Fast iteration during development
- Resume interrupted experiments
- Share intermediate results between team members
- Memory-efficient processing of large datasets

### 10.5 Configuration-Driven Architecture

**Decision**: Use Hydra for configuration management
**Rationale**:
- Declarative experiment specification
- Easy hyperparameter sweeps
- Version control of experimental setups
- Command-line flexibility

## 11. Performance Considerations

### 11.1 Memory Management

- **Streaming Processing**: Process videos one at a time
- **Batch Processing**: Maximize GPU utilization
- **DataFrame Optimization**: Efficient pandas operations
- **Garbage Collection**: Explicit cleanup of large objects

### 11.2 Scalability

- **Horizontal Scaling**: Multiple GPUs/workers
- **Vertical Scaling**: Large batch sizes for GPU efficiency
- **Data Subsampling**: Configurable dataset size reduction
- **Checkpointing**: Resume from any processing stage

### 11.3 Optimization Strategies

- **GPU Batching**: Maximize GPU utilization
- **Memory Pooling**: Reuse allocated memory
- **Lazy Loading**: Load data only when needed
- **Caching**: Cache expensive computations

## 12. Extensibility

### 12.1 Adding New Datasets

```python
class MyDataset(TrackingDataset):
    def __init__(self, dataset_path, **kwargs):
        # Load dataset-specific data
        sets = {
            "train": self._load_split("train"),
            "val": self._load_split("val"),
            "test": self._load_split("test")
        }
        super().__init__(dataset_path, sets, **kwargs)
```

### 12.2 Adding New Modules

```python
class MyDetector(ImageLevelModule):
    input_columns = []
    output_columns = ["bbox_ltwh", "bbox_conf", "category_id"]

    def process(self, batch, detections, metadatas):
        # Detection logic here
        return new_detections
```

### 12.3 Adding New Evaluators

```python
class MyEvaluator(Evaluator):
    def run(self, tracker_state):
        # Custom evaluation logic
        return metrics
```

## 13. Development Workflow

### 13.1 Research Workflow

1. **Setup**: Configure dataset and modules
2. **Development**: Use small dataset subset (nvid=5)
3. **Debugging**: Enable state saving/loading
4. **Evaluation**: Run on validation set
5. **Optimization**: Tune hyperparameters
6. **Final Evaluation**: Run on test set

### 13.2 Production Workflow

1. **Configuration**: Set production parameters
2. **Batch Processing**: Process full dataset
3. **Monitoring**: Track progress and metrics
4. **Result Analysis**: Generate reports and visualizations
5. **Deployment**: Export models and configurations

## 14. Conclusion

TrackLab's architecture provides a robust, flexible framework for multi-object tracking research. Its modular design, hierarchical data model, and configuration-driven approach enable efficient experimentation while maintaining production-ready performance. The framework's emphasis on data-centric processing and state management makes it particularly well-suited for complex tracking tasks like sports analytics and game state reconstruction.