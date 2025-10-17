"""BPBReID StrongSORT multi-object tracking module for TrackLab."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from tracklab.pipeline import ImageLevelModule
from . import strong_sort as strong_sort

import logging

log = logging.getLogger(__name__)


class BPBReIDStrongSORT(ImageLevelModule):
    """BPBReID StrongSORT multi-object tracking module for TrackLab.

    This module performs multi-object tracking using StrongSORT with BPBReID
    features, combining motion, appearance, and pose information for robust tracking.
    """

    input_columns: List[str] = [
        "bbox_ltwh",
        "embeddings",
        "visibility_scores",
    ]
    output_columns: List[str] = [
        "track_id",
        "track_bbox_kf_ltwh",
        "track_bbox_pred_kf_ltwh",
        "matched_with",
        "costs",
        "hits",
        "age",
        "time_since_update",
        "state",
    ]

    def __init__(
        self, cfg: Any, device: str, batch_size: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Initialize the BPBReID StrongSORT module.

        Args:
            cfg: Configuration object containing tracking parameters.
            device: Device to run inference on.
            batch_size: Batch size for processing (unused, fixed to 1).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device

        self.reset()

    def reset(self) -> None:
        """Reset the tracker state to start tracking in a new video."""
        self.model = strong_sort.StrongSORT(
            ema_alpha=self.cfg.ema_alpha,
            mc_lambda=self.cfg.mc_lambda,
            max_dist=self.cfg.max_dist,
            motion_criterium=self.cfg.motion_criterium,
            max_iou_distance=self.cfg.max_iou_distance,
            max_oks_distance=self.cfg.max_oks_distance,
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            nn_budget=self.cfg.nn_budget,
            min_bbox_confidence=self.cfg.min_bbox_confidence,
            only_position_for_kf_gating=self.cfg.only_position_for_kf_gating,
            max_kalman_prediction_without_update=self.cfg.max_kalman_prediction_without_update,
            matching_strategy=self.cfg.matching_strategy,
            gating_thres_factor=self.cfg.gating_thres_factor,
            w_kfgd=self.cfg.w_kfgd,
            w_reid=self.cfg.w_reid,
            w_st=self.cfg.w_st,
        )
        # For camera compensation
        self.prev_frame: Optional[np.ndarray] = None

    def prepare_next_frame(self, next_frame: np.ndarray) -> None:
        """Prepare tracker for next frame with Kalman filter prediction and camera compensation.

        Args:
            next_frame: Next frame image for camera motion compensation.
        """
        # Propagate the state distribution to the current time step using a Kalman filter prediction step.
        self.model.tracker.predict()

        # Camera motion compensation
        if self.cfg.ecc:
            if self.prev_frame is not None:
                self.model.tracker.camera_update(self.prev_frame, next_frame)
            self.prev_frame = next_frame

    @torch.no_grad()
    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Dict[str, Union[List[Any], np.ndarray]]:
        """Preprocess detections with embeddings and visibility scores for tracking.

        Args:
            image: Input image array (unused for preprocessing).
            detections: Detection DataFrame with embeddings and visibility scores.
            metadata: Image metadata series.

        Returns:
            Dictionary containing processed detection inputs.
        """
        # Filter out balls (category_id=0) from person tracking
        # Balls should not be tracked with persons due to different motion/appearance characteristics
        if "category_id" in detections.columns:
            detections = detections[detections["category_id"] != 0]

        if len(detections) == 0:
            return {
                "id": [],
                "bbox_ltwh": [],
                "reid_features": [],
                "visibility_scores": [],
                "scores": [],
                "classes": [],
                "frame": [],
            }
        if hasattr(detections, "bbox_conf"):
            score = detections.bbox.conf()
        else:
            score = detections.keypoints_conf
        input_tuple = {
            "id": detections.index.to_numpy(),
            "bbox_ltwh": np.stack(detections.bbox_ltwh),  # type: ignore
            "reid_features": np.stack(detections.embeddings),  # type: ignore
            "visibility_scores": np.stack(detections.visibility_scores),  # type: ignore
            "scores": np.stack(score),  # type: ignore
            "classes": np.zeros(len(detections.index)),
            "frame": np.ones(len(detections.index)) * metadata.frame,
        }
        if "keypoints_xyc" in detections:
            input_tuple["keypoints"] = np.stack(detections.keypoints_xyc)  # type: ignore
        return input_tuple

    @torch.no_grad()
    def process(
        self, batch: Dict[str, Any], detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> Union[pd.DataFrame, List[Any]]:
        """Process batch and perform advanced tracking with track merging.

        Args:
            batch: Preprocessed batch data with embeddings and visibility scores.
            detections: Detection DataFrame.
            metadatas: Image metadata DataFrame.

        Returns:
            DataFrame with tracking results after track merging or empty list.
        """
        # Check if batch is empty (can happen when all detections are filtered out, e.g., only balls)
        if len(batch["id"]) == 0:
            return []
        results = self.model.update(
            batch["id"][0],
            batch["bbox_ltwh"][0],
            batch["reid_features"][0],
            batch["visibility_scores"][0],
            batch["scores"][0],
            batch["classes"][0],
            batch["frame"][0],
            batch["keypoints"][0] if "keypoints" in batch else None,
        )
        # Filter out short tracks (less than min_hits hits)
        if not results.empty:
            results = results[results["hits"] >= getattr(self.cfg, "min_hits", 5)]

        # Post-processing: merge tracks with high overlap and similar appearance
        def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
            """Calculate Intersection over Union between two bounding boxes.

            Args:
                bbox1: First bounding box [x, y, w, h].
                bbox2: Second bounding box [x, y, w, h].

            Returns:
                IoU score between 0 and 1.
            """
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            union_area = bbox1_area + bbox2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0

        merged = set()
        track_ids = results["track_id"].values
        if len(results) == 0:
            assert set(results.index).issubset(
                detections.index
            ), "Mismatch of indexes during the tracking. The results should match the detections."
            return results
        bboxes = np.stack(list(results["track_bbox_kf_ltwh"].values))
        # If embeddings available, use them
        if "reid_features" in detections:
            features = np.stack(list(detections.embeddings.values))
        else:
            features = None
        for i in range(len(track_ids)):
            if track_ids[i] in merged:
                continue
            for j in range(i + 1, len(track_ids)):
                if track_ids[j] in merged:
                    continue
                if iou(bboxes[i], bboxes[j]) > 0.7:
                    if features is not None:
                        sim = np.dot(features[i], features[j]) / (
                            np.linalg.norm(features[i]) * np.linalg.norm(features[j])
                            + 1e-6
                        )
                        if sim > 0.85:
                            merged.add(track_ids[j])
                    else:
                        merged.add(track_ids[j])
        if merged:
            results = results[~results["track_id"].isin(merged)]

        assert set(results.index).issubset(
            detections.index
        ), "Mismatch of indexes during the tracking. The results should match the detections."
        return results
