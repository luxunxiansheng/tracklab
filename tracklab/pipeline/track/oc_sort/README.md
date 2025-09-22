# OC-SORT

OC-SORT (Observation-Centric SORT) is an improvement over the original SORT algorithm that prioritizes high-confidence detections and uses motion information for better track association and handling of occlusions.

## High-Level Ideas

- **Observation Focus**: Emphasizes reliable detections over predicted states for association decisions.
- **Motion Utilization**: Leverages velocity information from Kalman filter predictions to guide matching.
- **Association**: Uses IoU-based matching with motion-aware scoring to reduce identity switches.
- **Simplicity**: Maintains the efficiency of SORT while improving robustness in challenging tracking scenarios.
