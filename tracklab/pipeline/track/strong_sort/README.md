# StrongSORT

StrongSORT is an enhanced version of SORT that incorporates appearance information through ReID (Re-identification) features for more accurate and robust multi-object tracking.

## High-Level Ideas
- **Appearance Integration**: Uses deep learning-based ReID models to extract appearance features for identity preservation.
- **Motion Modeling**: Applies Kalman filtering for trajectory prediction and state management.
- **Association Mechanism**: Combines IoU matching with appearance similarity scores for track-to-detection association.
- **Long-Term Tracking**: Improves performance in scenarios requiring identity maintenance over extended periods, such as in surveillance or sports tracking.