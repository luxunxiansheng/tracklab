# Team Assignment Module

This module provides team assignment functionality for sports tracking, specifically designed for team sports like soccer where players need to be assigned to teams and team sides (left/right) based on their positions and embeddings.

## Overview

The team assignment module consists of two main components that work together to automatically assign players to teams and determine team sides:

1. **TrackletTeamClustering**: Uses K-means clustering on player embeddings to group players into two teams
2. **TrackletTeamSideLabeling**: Determines team sides (left/right) based on player positions on the field

This module is particularly useful for sports analytics where team identification and side determination are crucial for tactical analysis and event detection.

## Main Components

### TrackletTeamClustering

Performs unsupervised clustering of player tracklets using their visual embeddings to automatically separate players into two teams.

**Algorithm:**
- Extracts embeddings from player tracklets (role = "player")
- Computes mean embedding for each tracklet
- Applies K-means clustering (k=2) to group players into teams
- Assigns cluster labels (0, 1) to each tracklet

**Key Features:**

- **Unsupervised Learning**: No manual team assignment required
- **Embedding-Based**: Uses visual features for robust team separation
- **Robust to Missing Data**: Handles cases with insufficient tracklets
- **Deterministic**: Uses fixed random state for reproducible results

**Input Requirements:**

- `track_id`: Unique identifier for each tracklet
- `embeddings`: Visual embeddings for each detection
- `role`: Must be "player" for clustering

**Output:**

- `team_cluster`: Cluster assignment (0 or 1)

### TrackletTeamSideLabeling

Determines the spatial orientation of teams on the field by analyzing player positions and assigning "left" or "right" team sides.

**Algorithm:**
1. **Team Position Analysis**: Calculates average x-coordinates of players in each cluster
2. **Side Determination**: Team with smaller average x-coordinate gets "left" side
3. **Goalkeeper Assignment**: Special handling for goalkeepers based on their position

**Key Features:**

- **Spatial Reasoning**: Uses field coordinates for side determination
- **Goalkeeper Handling**: Separate logic for goalkeeper team assignment
- **Robust Positioning**: Handles missing or invalid position data

**Input Requirements:**

- `track_id`: Tracklet identifier
- `team_cluster`: Cluster assignment from clustering module
- `bbox_pitch`: Pitch coordinates with `x_bottom_middle` field
- `role`: Player role ("player" or "goalkeeper")

**Output:**

- `team`: Team side assignment ("left" or "right")

## Configuration

### Clustering Configuration

```yaml
_target_: tracklab.pipeline.team.tracklet_team_clustering_api.TrackletTeamClustering
```

### Side Labeling Configuration

```yaml
_target_: tracklab.pipeline.team.tracklet_team_side_labeling_api.TrackletTeamSideLabeling
```

## Usage

The team assignment modules are typically used in sequence within the TrackLab pipeline:

1. **First**: Run `TrackletTeamClustering` to assign players to clusters
2. **Then**: Run `TrackletTeamSideLabeling` to determine team sides

### Pipeline Integration

```python
# In your TrackLab pipeline configuration
pipeline:
  - team_clustering
  - team_side_labeling

modules:
  team_clustering:
    _target_: tracklab.pipeline.team.tracklet_team_clustering_api.TrackletTeamClustering

  team_side_labeling:
    _target_: tracklab.pipeline.team.tracklet_team_side_labeling_api.TrackletTeamSideLabeling
```

## Algorithm Details

### K-means Clustering Algorithm

The team clustering uses the K-means algorithm to partition player embeddings into two clusters representing the two teams.

#### Mathematical Foundation

**K-means Objective:**

```math
minimize ∑_{i=1}^n ||x_i - μ_{c_i}||²
```

Where:

- `x_i`: Player embedding vector
- `μ_{c_i}`: Centroid of cluster `c_i`
- `n`: Number of player tracklets

#### Implementation Steps

1. **Data Filtering & Preparation**

   ```python
   player_detections = detections[detections.role == "player"]
   ```

   - Filters detections to include only players (excludes referees, goalkeepers for clustering)

2. **Embedding Aggregation**

   ```python
   embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
   ```

   - Computes mean embedding across all detections in each tracklet
   - Reduces noise from individual frame variations
   - Creates single representative vector per player

3. **K-means Clustering**

   ```python
   kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
   ```

   - **k=2**: Fixed for two-team sports
   - **random_state=0**: Ensures reproducible results
   - Uses Euclidean distance for similarity measurement

4. **Edge Case Handling**
   - **Single Player**: Assigns cluster 0 when only one tracklet exists
   - **No Players**: Returns NaN values for all detections
   - **Missing Track IDs**: Skips NaN track identifiers

#### Algorithm Characteristics

- **Convergence**: K-means iteratively updates centroids until convergence
- **Initialization**: Uses k-means++ initialization for better starting centroids
- **Distance Metric**: Euclidean distance in embedding space
- **Time Complexity**: O(n × k × d × i) where n=tracklets, k=2, d=embedding_dim, i=iterations

### Side Labeling Algorithm

Determines team spatial orientation using pitch coordinates and geometric reasoning.

#### Coordinate System

The algorithm uses pitch-normalized coordinates where:

- **Origin (0,0)**: Typically center or corner of the pitch
- **X-axis**: Horizontal position (left to right from camera perspective)
- **Y-axis**: Vertical position (goal to goal)
- **x_bottom_middle**: X-coordinate of player's feet center

#### Side Determination Logic

1. **Position Aggregation**

   ```python
   avg_a = np.nanmean(xa_coordinates)  # Team cluster 0
   avg_b = np.nanmean(xb_coordinates)  # Team cluster 1
   ```

2. **Side Assignment**

   ```python
   if avg_a > avg_b:
       team_a → "right", team_b → "left"
   else:
       team_a → "left", team_b → "right"
   ```

   - **Lower x-coordinate** = "left" team (closer to left goal)
   - **Higher x-coordinate** = "right" team (closer to right goal)

3. **Goalkeeper Special Handling**

   ```python
   gk_team = "right" if bbox["x_bottom_middle"] > 0 else "left"
   ```

   - Goalkeepers assigned based on their absolute position
   - Overrides cluster-based assignment for goalkeepers
   - Uses coordinate origin as the dividing line

#### Robustness Features

- **NaN Handling**: Uses `np.nanmean()` to ignore missing coordinates
- **Empty Clusters**: Gracefully handles teams with no valid positions
- **Coordinate Validation**: Checks for dictionary format and required fields

### Algorithm Performance Considerations

#### Strengths

- **Unsupervised**: No manual labeling required
- **Scalable**: Linear time complexity with number of tracklets
- **Robust**: Handles missing data and edge cases
- **Deterministic**: Reproducible results with fixed random seed

#### Limitations

- **Assumption**: Exactly two teams in the scene
- **Embedding Quality**: Performance depends on embedding discriminative power
- **Position Accuracy**: Requires accurate pitch coordinate estimation
- **Static Assignment**: Teams assigned for entire video (no team changes)

#### Optimization Opportunities

- **Dimensionality Reduction**: PCA on embeddings before clustering
- **Distance Metrics**: Cosine similarity for high-dimensional embeddings
- **Temporal Smoothing**: Consistency across video frames
- **Confidence Scoring**: Uncertainty estimation for cluster assignments

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `torch`: Deep learning framework (for VideoLevelModule)
- `scikit-learn`: K-means clustering implementation
- TrackLab utilities (pipeline, coordinate systems)

## Integration Notes

- **Pipeline Order**: Clustering must run before side labeling
- **Data Requirements**: Requires embeddings and pitch coordinates
- **Role Filtering**: Only processes detections with appropriate roles
- **Error Handling**: Gracefully handles missing data and edge cases

## Output Format

The modules add the following columns to the detections DataFrame:

- `team_cluster`: Integer (0 or 1) from clustering
- `team`: String ("left" or "right") from side labeling

These columns can be used by downstream modules for team-specific analysis, event detection, and tactical insights.
