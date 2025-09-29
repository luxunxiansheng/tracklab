# Visualization Configuration Files

This directory contains YAML configuration files for customizing the visualization output of TrackLab. Each file defines a different visualization preset that can be used to display various aspects of tracking results on video frames.

## Available Configurations

### bbox.yaml

- **Purpose**: Standard bounding box visualization
- **Features**: Shows bounding boxes around detected objects with tracking IDs
- **Visualizers**: Frame counter, default detections
- **Use case**: Basic object detection and tracking visualization

### keypoints.yaml

- **Purpose**: Pose/keypoint visualization
- **Features**: Displays keypoints (e.g., body joints) along with bounding boxes
- **Visualizers**: Frame counter, default detections, default keypoints
- **Use case**: Human pose estimation and tracking

### tracking.yaml

- **Purpose**: Tracking trajectory visualization
- **Features**: Shows tracking lines/history for object movement
- **Visualizers**: Frame counter, tracking lines
- **Use case**: Analyzing object movement patterns over time

### ellipse.yaml

- **Purpose**: Elliptical detection visualization
- **Features**: Displays detections as ellipses instead of rectangles
- **Use case**: Alternative detection representation

### gamestate.yaml

- **Purpose**: Game state visualization
- **Features**: Specialized for sports analytics (e.g., player positions, game statistics)
- **Use case**: Sports tracking with game context

### stats.yaml

- **Purpose**: Statistical visualization
- **Features**: Shows detection statistics and metrics
- **Use case**: Performance analysis and debugging

### debug.yaml

- **Purpose**: Debug visualization
- **Features**: Detailed debugging information and intermediate results
- **Use case**: Development and troubleshooting

### none.yaml

- **Purpose**: No visualization
- **Features**: Disables video saving and visualization
- **Use case**: Headless processing without visual output

## Color Configuration

### colors.yaml

Base color configuration file that defines color schemes for different visualization elements:

- `default`: General color settings
- `bbox`: Bounding box colors
- `text`: Text label colors
- `keypoint`: Keypoint colors

### colors_gs.yaml

Game state specific color configuration (likely for sports visualization).

## Configuration Structure

Each visualization config follows this structure:

```yaml
defaults:
  - colors  # Inherits color settings

_target_: tracklab.callbacks.visualization.VisualizationEngine
save_videos: True  # Whether to save output videos

visualizers:
  # List of visualizer components to use
  frame_counter:
    _target_: tracklab.callbacks.visualization.FrameCount
  detections:
    _target_: tracklab.callbacks.visualization.DefaultDetection
  # ... other visualizers

colors:
  # Color overrides (optional)
  default:
    no_id: [255, 0, 0]  # Red for untracked objects
    prediction: "track_id"  # Color by tracking ID
    ground_truth: [0, 255, 0]  # Green for ground truth
  cmap: 10  # Number of distinct colors for track_id mode
```

## Color Options

- `null`: No display
- `"track_id"`: Automatic color based on tracking ID
- `[R, G, B]`: Specific RGB color values

## Available Visualizers

- **Detections**: `DefaultDetection`, `FullDetection`, `DebugDetection`, `EllipseDetection`, `SimpleDetectionStats`, `DetectionStats`
- **Keypoints**: `DefaultKeypoints`, `FullKeypoints`
- **Tracking**: `TrackingLine` (shows tracking history)
- **Images**: `FrameCount`, `IgnoreRegions`

## Usage

To use a visualization config in your TrackLab pipeline:

```yaml
# In your main config file
visualization:
  _target_: tracklab.configs.visualization.bbox  # or keypoints, tracking, etc.
```

Or specify the config file directly when running TrackLab:

```bash
tracklab --config your_config.yaml --visualization bbox
```

## Customization

You can create custom visualization configs by:

1. Copying an existing config file
2. Modifying the `visualizers` section to add/remove components
3. Adjusting color settings in the `colors` section
4. Setting `save_videos` to control output generation

For advanced customization, refer to the visualization engine and individual visualizer implementations in `tracklab/visualization/`.
