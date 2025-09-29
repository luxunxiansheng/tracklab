"""Detection-only evaluator for TrackLab bounding box evaluation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tracklab.callbacks.evaluation_callback.evaluator import Evaluator as EvaluatorBase

log = logging.getLogger(__name__)


class DetectionOnlyEvaluator(EvaluatorBase):
    """Simple detection evaluator for bounding box evaluation without tracking.

    This evaluator computes Average Precision (AP) and Average Recall (AR)
    metrics for bounding box detection tasks. It performs evaluation at
    multiple IoU thresholds and provides comprehensive detection performance
    assessment without requiring tracking or calibration data.
    """

    def __init__(
        self,
        cfg: Any,
        tracking_dataset: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the detection-only evaluator.

        Args:
            cfg: Configuration object containing evaluation parameters.
            tracking_dataset: The tracking dataset instance.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg
        self.tracking_dataset = tracking_dataset
        self.eval_set: str = cfg.get("eval_set", "val")
        self.show_progressbar: bool = cfg.get("show_progressbar", True)
        self.dataset_path: Optional[str] = cfg.get("dataset_path", None)
        self.confidence_thresholds: List[float] = cfg.confidence_thresholds
        self.iou_thresholds: List[float] = cfg.iou_thresholds

    def run(self, tracker_state: Any) -> Dict[str, Union[float, str]]:
        """Run detection evaluation on the tracker state.

        Args:
            tracker_state: The tracker state containing predictions and ground truth.

        Returns:
            Dictionary containing evaluation results or error messages.
        """
        log.info("Starting detection-only evaluation")

        # Get predictions and ground truth
        detections_pred = tracker_state.detections_pred
        detections_gt = tracker_state.detections_gt

        # Check if we have valid DataFrames
        if detections_pred is None or (
            hasattr(detections_pred, "empty") and detections_pred.empty
        ):
            log.warning("No predictions found. Cannot evaluate.")
            return {"error": "No predictions found"}

        if detections_gt is None or (
            hasattr(detections_gt, "empty") and detections_gt.empty
        ):
            log.warning("No ground truth found. Cannot evaluate.")
            return {"error": "No ground truth found"}

        log.info(
            f"Evaluating {len(detections_pred)} predictions against {len(detections_gt)} ground truth detections"
        )

        results = {}

        # Compute metrics for each IoU threshold
        for iou_thresh in self.iou_thresholds:
            log.info(f"Computing metrics for IoU threshold: {iou_thresh}")

            # Compute AP and AR for this IoU threshold
            ap_scores = []
            ar_scores = []

            # Group by image for evaluation
            for image_id in detections_gt["image_id"].unique():
                gt_image = detections_gt[detections_gt["image_id"] == image_id]
                pred_image = detections_pred[detections_pred["image_id"] == image_id]

                if pred_image.empty:
                    # No predictions for this image
                    ap_scores.append(0.0)
                    ar_scores.append(0.0)
                    continue

                # Compute IoU matrix between predictions and ground truth
                iou_matrix = self._compute_iou_matrix(pred_image, gt_image)

                # Match predictions to ground truth
                matches, tp, fp, fn = self._match_detections(iou_matrix, iou_thresh)

                # Compute AP and AR for this image
                if len(gt_image) > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    ap_scores.append(precision)
                    ar_scores.append(recall)

            # Average across all images
            mean_ap = np.mean(ap_scores) if ap_scores else 0.0
            mean_ar = np.mean(ar_scores) if ar_scores else 0.0

            results[f"AP@{iou_thresh}"] = mean_ap
            results[f"AR@{iou_thresh}"] = mean_ar

        # Compute mAP (mean across IoU thresholds)
        ap_values = [results[f"AP@{thresh}"] for thresh in self.iou_thresholds]
        ar_values = [results[f"AR@{thresh}"] for thresh in self.iou_thresholds]

        results["mAP"] = np.mean(ap_values)
        results["mAR"] = np.mean(ar_values)

        # Log results
        self._log_results(results)

        return results

    def _compute_iou_matrix(
        self, pred_detections: pd.DataFrame, gt_detections: pd.DataFrame
    ) -> np.ndarray:
        """Compute IoU matrix between predictions and ground truth detections.

        Args:
            pred_detections: DataFrame containing predicted detections.
            gt_detections: DataFrame containing ground truth detections.

        Returns:
            2D numpy array with IoU values between all prediction-GT pairs.
        """
        n_pred = len(pred_detections)
        n_gt = len(gt_detections)

        iou_matrix = np.zeros((n_pred, n_gt))

        for i, pred_row in enumerate(pred_detections.itertuples()):
            pred_bbox = pred_row.bbox_ltwh
            for j, gt_row in enumerate(gt_detections.itertuples()):
                gt_bbox = gt_row.bbox_ltwh
                iou_matrix[i, j] = self._compute_bbox_iou(pred_bbox, gt_bbox)

        return iou_matrix

    def _compute_bbox_iou(self, bbox1: Any, bbox2: Any) -> float:
        """Compute IoU between two bounding boxes in LTWH format.

        Args:
            bbox1: First bounding box as (left, top, width, height).
            bbox2: Second bounding box as (left, top, width, height).

        Returns:
            IoU value between 0.0 and 1.0.
        """
        # Convert ltwh to xyxy
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _match_detections(
        self, iou_matrix: np.ndarray, iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], int, int, int]:
        """Match predictions to ground truth using greedy matching.

        Args:
            iou_matrix: 2D array of IoU values between predictions and ground truth.
            iou_threshold: Minimum IoU threshold for considering a match.

        Returns:
            Tuple of (matches, true_positives, false_positives, false_negatives).
        """
        n_pred, n_gt = iou_matrix.shape

        if n_pred == 0 or n_gt == 0:
            return [], 0, n_pred, n_gt

        # Simple greedy matching (can be replaced with Hungarian algorithm for better matching)
        matches = []
        used_gt = set()
        used_pred = set()

        # Sort predictions by confidence if available
        pred_indices = list(range(n_pred))

        tp = 0
        for pred_idx in pred_indices:
            best_gt_idx = -1
            best_iou = 0.0

            for gt_idx in range(n_gt):
                if gt_idx in used_gt:
                    continue

                iou = iou_matrix[pred_idx, gt_idx]
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matches.append((pred_idx, best_gt_idx))
                used_gt.add(best_gt_idx)
                used_pred.add(pred_idx)
                tp += 1

        fp = n_pred - tp  # False positives (unmatched predictions)
        fn = n_gt - tp  # False negatives (unmatched ground truth)

        return matches, tp, fp, fn

    def _log_results(self, results: Dict[str, Union[float, str]]) -> None:
        """Log evaluation results in a formatted manner.

        Args:
            results: Dictionary containing evaluation metrics.
        """
        log.info("Detection Evaluation Results:")
        log.info("=" * 40)

        # Log individual IoU threshold results
        for iou_thresh in self.iou_thresholds:
            ap = results[f"AP@{iou_thresh}"]
            ar = results[f"AR@{iou_thresh}"]
            log.info(f"IoU@{iou_thresh}: AP={ap:.3f}, AR={ar:.3f}")

        log.info("-" * 40)
        log.info(f"mAP@[.5:.95]: {results['mAP']:.3f}")
        log.info(f"mAR@[.5:.95]: {results['mAR']:.3f}")
        log.info("=" * 40)
