# ByteTrack

ByteTrack is a simple yet effective multi-object tracking algorithm developed by ByteDance, which improves upon SORT by utilizing both high and low-confidence detections for better track initialization and maintenance.

## High-Level Ideas

- **Detection Utilization**: Incorporates low-confidence detections as track candidates, allowing for earlier track starts and recovery from misses.
- **Motion Prediction**: Uses Kalman filter for trajectory prediction and state estimation.
- **Association**: Employs IoU-based matching with a threshold strategy to associate detections to existing tracks.
- **Efficiency**: Maintains real-time performance while improving tracking accuracy in challenging scenarios like crowded environments.
