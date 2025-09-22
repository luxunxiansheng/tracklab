# Deep OC-SORT

Deep OC-SORT extends OC-SORT by integrating deep learning-based appearance features for more accurate association in multi-object tracking, particularly in complex scenes with occlusions.

## High-Level Ideas

- **Observation-Centric Approach**: Focuses on reliable detections and uses motion information to guide associations.
- **Deep Features**: Incorporates neural network-based ReID features for appearance matching.
- **Motion Modeling**: Employs Kalman filtering for trajectory prediction and velocity estimation.
- **Association Strategy**: Combines IoU matching with appearance similarity, prioritizing high-confidence observations.
- **Robustness**: Enhances performance in crowded scenes and when dealing with similar-looking objects.
