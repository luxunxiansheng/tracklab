# BoT-SORT

BoT-SORT (Beyond Tracking - Simple Online and Realtime Tracking) is an advanced multi-object tracking algorithm that enhances the original SORT framework by incorporating both motion and appearance information for more robust association.

## High-Level Ideas
- **Motion Modeling**: Uses a Kalman filter to predict object trajectories and handle occlusions.
- **Appearance Features**: Integrates ReID (Re-identification) features to distinguish objects with similar motion patterns.
- **Association Strategy**: Combines IoU (Intersection over Union) matching with appearance similarity for better track continuity.
- **Robustness**: Improves handling of crowded scenes, fast-moving objects, and temporary disappearances by leveraging both cues.