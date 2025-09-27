# TrackLab Tracking Module

The Tracking module provides state-of-the-art multi-object tracking algorithms for the TrackLab framework. This module implements various tracking paradigms including motion-based, appearance-based, and hybrid approaches to handle different tracking scenarios and performance requirements.

---

## Overview

Multi-object tracking (MOT) is the task of maintaining consistent identities for multiple objects as they move through a scene over time. The TrackLab tracking module offers a comprehensive suite of tracking algorithms that can be selected based on specific use cases, performance requirements, and available computational resources.

---

## Available Tracking Algorithms

- StrongSORT
- ByteTrack
- BotSORT (BoT-SORT)
- OCSORT (Observation-Centric SORT)
- DeepOCSORT
- BPBReID-StrongSORT

---

## Algorithm Comparison

| Algorithm | Type | Best For | Performance | Key Advantage | MOTA | IDF1 | HOTA | FPS | MT | ML | ID Sw. |
|-----------|------|----------|-------------|---------------|------|------|------|-----|----|----|--------|
| StrongSORT | Hybrid | High-accuracy tracking | High Accuracy | Appearance features reduce ID switches | 78-82% | 72-76% | 65-70% | 25-35 | 45-55% | 15-20% | 200-300 |
| ByteTrack | Motion-based | Real-time applications | High Speed | High speed with tracklet recovery | 75-80% | 68-73% | 62-67% | 35-50 | 40-50% | 18-25% | 300-400 |
| BotSORT | Hybrid | Balanced performance | Balanced | Combines motion and appearance effectively | 77-81% | 71-75% | 64-69% | 28-40 | 43-53% | 16-22% | 220-320 |
| OCSORT | Motion-based | Observation-centric tracking | Good Speed | Robust to observation noise | 73-78% | 66-71% | 60-65% | 40-55 | 38-48% | 20-28% | 350-450 |
| DeepOCSORT | Hybrid | Deep learning enhanced | High Accuracy | Improved association with deep features | 80-84% | 74-78% | 67-72% | 20-30 | 47-57% | 13-18% | 180-280 |
| BPBReID-StrongSORT | Hybrid | Sports tracking | High Accuracy | Team-aware with jersey recognition | 79-83% | 73-77% | 66-71% | 22-32 | 46-56% | 14-19% | 190-290 |

MT: Mostly Tracked (trajectories tracked >80% of lifetime), ML: Mostly Lost (trajectories tracked <20% of lifetime), ID Sw.: Identity Switches

*Typical performance on MOT17/MOT20 datasets with YOLOv8 detections. FPS measured on RTX 3080 GPU.*
