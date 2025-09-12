# Tracklet Aggregation Module

This module provides tracklet aggregation functionality for the TrackLab framework, specifically implementing majority voting for attribute consensus across tracklets.

## Overview

The tracklet aggregation module processes detection data to resolve attribute conflicts within tracklets using majority voting. This is particularly useful for multi-frame attribute detection where individual frame detections may be noisy or inconsistent.

## Algorithm Details

### Confidence-Weighted Majority Voting

The `MajorityVoteTracklet` implements a **confidence-weighted majority voting algorithm** that resolves attribute inconsistencies across tracklets. Unlike simple majority voting, this algorithm uses detection confidence scores to weight votes.

#### Algorithm Steps

1. **Tracklet Grouping**: Group detections by `track_id` to form tracklets
2. **Attribute Processing**: For each configured attribute (e.g., jersey_number, role):
   - Extract attribute detections and their confidence scores
   - Apply confidence-weighted voting to determine the most likely value
3. **Value Assignment**: Assign the winning attribute value to all detections in the tracklet

#### Voting Mechanism (`select_highest_voted_att`)

```python
def select_highest_voted_att(atts, atts_confidences=None):
    confidence_sum = {}
    atts_confidences = [1] * len(atts) if atts_confidences is None else atts_confidences

    # Sum confidence scores for each unique attribute value
    for attribute_value, confidence in zip(atts, atts_confidences):
        if attribute_value not in confidence_sum:
            confidence_sum[attribute_value] = 0
        confidence_sum[attribute_value] += confidence

    # Return attribute with highest total confidence
    if len(confidence_sum) == 0:
        return None
    return max(confidence_sum, key=confidence_sum.get)
```

**Key Characteristics:**

- **Confidence Weighting**: Higher confidence detections have more influence on the final decision
- **Handles Missing Data**: Gracefully handles cases with no confidence data (defaults to equal weighting)
- **Robust to Noise**: Reduces impact of low-confidence detections
- **Deterministic**: Always produces the same result for the same input data

#### Example

For a tracklet with jersey number detections: `[10, 10, 12, 10]` with confidences `[0.8, 0.9, 0.6, 0.7]`:

- Jersey 10 total confidence: `0.8 + 0.9 + 0.7 = 2.4`
- Jersey 12 total confidence: `0.6`
- **Result**: Jersey number `10` (highest confidence sum)

### MajorityVoteTracklet Class

The core class that implements the confidence-weighted majority voting algorithm.

**Key Features:**

- Processes tracklets to resolve attribute inconsistencies
- Uses confidence-weighted majority voting
- Supports multiple attributes simultaneously
- Integrates with the TrackLab pipeline as a VideoLevelModule

**Input Requirements:**

- Detection DataFrame with track IDs
- Attribute detection columns (e.g., `jersey_number_detection`, `role_detection`)
- Attribute confidence columns (e.g., `jersey_number_confidence`, `role_confidence`)

**Output:**

- Consolidated attribute values for each tracklet
- NaN values for tracks without sufficient data

## Configuration

The module is configured via YAML files in the TrackLab config system:

```yaml
_target_: tracklab.wrappers.tracklet_agg.majority_vote_api.MajorityVoteTracklet
cfg:
  attributes: ["jersey_number", "role"]
```

**Parameters:**

- `attributes`: List of attribute names to process (e.g., "jersey_number", "role")

## Usage

The module is automatically instantiated and used within the TrackLab pipeline. It processes detection data after individual frame processing but before final evaluation.

### Example Configuration Files

- `majority_vote.yaml`: Basic majority voting configuration
- `voting_role_jn.yaml`: Configuration for jersey number and role voting

## Dependencies

- pandas
- numpy
- torch
- TrackLab utilities (cv2, attribute_voting, pipeline modules)

## Integration

This module integrates with the TrackLab pipeline as a VideoLevelModule, processing data at the video level rather than individual frames. It expects pre-processed detection data with track IDs and attribute detections from earlier pipeline stages.
