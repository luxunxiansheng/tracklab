"""Self-evaluation callback for TrackLab tracking without ground truth data."""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from collections import defaultdict

from tracklab.callbacks.callback import Callback

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class SelfEvaluationCallback(Callback):
    """Self-evaluation callback that computes proxy metrics without ground truth.

    This callback analyzes tracking quality using internal consistency metrics,
    track statistics, and data quality checks that don't require ground truth annotations.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the self-evaluation callback."""
        pass

    def on_dataset_track_end(self, engine: "TrackingEngine") -> None:
        """Run self-evaluation on the completed tracking results."""
        log.info("Starting self-evaluation (no ground truth required)...")

        # Get tracking results
        tracker_state = engine.tracker_state
        detections = tracker_state.detections_pred

        if detections is None or detections.empty:
            log.warning("No tracking results found for evaluation")
            return

        # Compute comprehensive self-evaluation metrics
        metrics = self._compute_self_evaluation_metrics(detections)

        # Log results
        self._log_evaluation_results(metrics)

        log.info("Self-evaluation completed.")

    def _compute_self_evaluation_metrics(
        self, detections: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute various self-evaluation metrics from tracking data."""
        metrics = {}

        # Basic statistics
        metrics.update(self._compute_basic_statistics(detections))

        # Track consistency metrics
        metrics.update(self._compute_track_consistency_metrics(detections))

        # Detection quality metrics
        metrics.update(self._compute_detection_quality_metrics(detections))

        # Team classification metrics
        metrics.update(self._compute_team_classification_metrics(detections))

        # Pitch coordinate quality metrics
        metrics.update(self._compute_pitch_coordinate_metrics(detections))

        # Temporal consistency metrics
        metrics.update(self._compute_temporal_consistency_metrics(detections))

        return metrics

    def _compute_basic_statistics(self, detections: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic tracking statistics."""
        stats = {}

        # Total detections and frames
        total_detections = len(detections)
        unique_frames = detections["image_id"].nunique()
        unique_tracks = detections["track_id"].nunique()

        stats["total_detections"] = total_detections
        stats["unique_frames"] = unique_frames
        stats["unique_tracks"] = unique_tracks
        stats["detections_per_frame"] = (
            total_detections / unique_frames if unique_frames > 0 else 0
        )
        stats["frames_with_detections"] = len(detections["image_id"].unique())
        stats["detection_coverage"] = (
            stats["frames_with_detections"] / unique_frames if unique_frames > 0 else 0
        )

        return stats

    def _compute_track_consistency_metrics(
        self, detections: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute track consistency and quality metrics."""
        track_metrics = {}

        # Group by track_id
        track_groups = detections.groupby("track_id")

        # Track lengths
        track_lengths = track_groups.size()
        track_metrics["mean_track_length"] = track_lengths.mean()
        track_metrics["median_track_length"] = track_lengths.median()
        track_metrics["min_track_length"] = track_lengths.min()
        track_metrics["max_track_length"] = track_lengths.max()
        track_metrics["std_track_length"] = track_lengths.std()

        # Track length distribution
        track_metrics["tracks_gt_10_frames"] = (track_lengths > 10).sum()
        track_metrics["tracks_gt_50_frames"] = (track_lengths > 50).sum()
        track_metrics["tracks_gt_100_frames"] = (track_lengths > 100).sum()
        track_metrics["tracks_gt_500_frames"] = (track_lengths > 500).sum()

        # Track fragmentation (gaps in tracks)
        track_fragmentation = []
        for track_id, group in track_groups:
            frames = sorted(group["image_id"].unique())
            if len(frames) > 1:
                gaps = np.diff(frames)
                fragmentation = np.sum(gaps > 1)  # Number of gaps > 1 frame
                track_fragmentation.append(fragmentation)

        if track_fragmentation:
            track_metrics["mean_track_fragmentation"] = np.mean(track_fragmentation)
            track_metrics["max_track_fragmentation"] = np.max(track_fragmentation)
        else:
            track_metrics["mean_track_fragmentation"] = 0
            track_metrics["max_track_fragmentation"] = 0

        return track_metrics

    def _compute_detection_quality_metrics(
        self, detections: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute detection quality metrics."""
        quality_metrics = {}

        # Bounding box sizes
        if "bbox_ltwh" in detections.columns:
            bboxes = np.stack(detections["bbox_ltwh"].values)
            widths = bboxes[:, 2]
            heights = bboxes[:, 3]
            areas = widths * heights

            quality_metrics["mean_bbox_width"] = np.mean(widths)
            quality_metrics["mean_bbox_height"] = np.mean(heights)
            quality_metrics["mean_bbox_area"] = np.mean(areas)
            quality_metrics["bbox_size_variability"] = np.std(areas)

        # Detection confidence (if available)
        if "bbox_conf" in detections.columns:
            confs = detections["bbox_conf"].values
            quality_metrics["mean_detection_confidence"] = np.mean(confs)
            quality_metrics["min_detection_confidence"] = np.min(confs)
            quality_metrics["max_detection_confidence"] = np.max(confs)
            quality_metrics["confidence_std"] = np.std(confs)

        return quality_metrics

    def _compute_team_classification_metrics(
        self, detections: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute team classification quality metrics."""
        team_metrics = {}

        if "team" not in detections.columns:
            return team_metrics

        # Team distribution
        team_counts = detections["team"].value_counts()
        team_metrics["team_distribution"] = team_counts.to_dict()

        # Team assignment consistency within tracks
        track_team_consistency = []
        for track_id, group in detections.groupby("track_id"):
            teams = group["team"].dropna().unique()
            if len(teams) > 1:
                track_team_consistency.append(track_id)

        team_metrics["tracks_with_team_changes"] = len(track_team_consistency)
        team_metrics["team_consistency_rate"] = 1 - (
            len(track_team_consistency) / detections["track_id"].nunique()
        )

        return team_metrics

    def _compute_pitch_coordinate_metrics(
        self, detections: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute pitch coordinate quality metrics."""
        pitch_metrics = {}

        if "bbox_pitch" not in detections.columns:
            return pitch_metrics

        # Extract pitch coordinates
        pitch_coords = []
        for bp in detections["bbox_pitch"].dropna():
            if (
                isinstance(bp, dict)
                and "x_bottom_middle" in bp
                and "y_bottom_middle" in bp
            ):
                pitch_coords.append((bp["x_bottom_middle"], bp["y_bottom_middle"]))

        if not pitch_coords:
            return pitch_metrics

        coords = np.array(pitch_coords)

        # Coordinate statistics
        pitch_metrics["pitch_x_mean"] = np.mean(coords[:, 0])
        pitch_metrics["pitch_y_mean"] = np.mean(coords[:, 1])
        pitch_metrics["pitch_x_std"] = np.std(coords[:, 0])
        pitch_metrics["pitch_y_std"] = np.std(coords[:, 1])
        pitch_metrics["pitch_x_range"] = np.ptp(coords[:, 0])
        pitch_metrics["pitch_y_range"] = np.ptp(coords[:, 1])

        # Sanity checks for soccer pitch dimensions (approximate)
        # Standard soccer pitch is about 105m x 68m
        reasonable_x_range = pitch_metrics["pitch_x_range"] < 150  # Allow some margin
        reasonable_y_range = pitch_metrics["pitch_y_range"] < 100

        pitch_metrics["pitch_coordinates_reasonable"] = (
            reasonable_x_range and reasonable_y_range
        )

        # Check for outliers
        x_zscores = np.abs(
            (coords[:, 0] - pitch_metrics["pitch_x_mean"])
            / pitch_metrics["pitch_x_std"]
        )
        y_zscores = np.abs(
            (coords[:, 1] - pitch_metrics["pitch_y_mean"])
            / pitch_metrics["pitch_y_std"]
        )

        outlier_threshold = 3.0
        x_outliers = np.sum(x_zscores > outlier_threshold)
        y_outliers = np.sum(y_zscores > outlier_threshold)

        pitch_metrics["pitch_x_outliers"] = x_outliers
        pitch_metrics["pitch_y_outliers"] = y_outliers
        pitch_metrics["total_pitch_outliers"] = x_outliers + y_outliers

        return pitch_metrics

    def _compute_temporal_consistency_metrics(
        self, detections: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute temporal consistency metrics."""
        temporal_metrics = {}

        # Frame-to-frame consistency
        detections_by_frame = detections.groupby("image_id")

        # Track continuity across frames
        frame_track_counts = []
        for frame_id, frame_dets in detections_by_frame:
            frame_track_counts.append(len(frame_dets))

        if frame_track_counts:
            temporal_metrics["mean_tracks_per_frame"] = np.mean(frame_track_counts)
            temporal_metrics["std_tracks_per_frame"] = np.std(frame_track_counts)
            temporal_metrics["min_tracks_per_frame"] = np.min(frame_track_counts)
            temporal_metrics["max_tracks_per_frame"] = np.max(frame_track_counts)

        # Track velocity consistency (if pitch coordinates available)
        if "bbox_pitch" in detections.columns:
            track_velocities = []
            for track_id, group in detections.groupby("track_id"):
                group = group.sort_values("image_id")
                if len(group) < 2:
                    continue

                coords = []
                for bp in group["bbox_pitch"].dropna():
                    if (
                        isinstance(bp, dict)
                        and "x_bottom_middle" in bp
                        and "y_bottom_middle" in bp
                    ):
                        coords.append((bp["x_bottom_middle"], bp["y_bottom_middle"]))

                if len(coords) >= 2:
                    coords = np.array(coords)
                    displacements = np.diff(coords, axis=0)
                    velocities = np.linalg.norm(displacements, axis=1)
                    track_velocities.extend(velocities)

            if track_velocities:
                temporal_metrics["mean_track_velocity"] = np.mean(track_velocities)
                temporal_metrics["max_track_velocity"] = np.max(track_velocities)
                # Flag unrealistic velocities (> 10 m/frame ≈ 30 km/h at 30fps)
                temporal_metrics["unrealistic_velocities"] = np.sum(
                    np.array(track_velocities) > 10
                )

        return temporal_metrics

    def _log_evaluation_results(self, metrics: Dict[str, Any]) -> None:
        """Log the evaluation results in a readable format."""
        log.info("=" * 60)
        log.info("SELF-EVALUATION RESULTS (No Ground Truth Required)")
        log.info("=" * 60)

        # Basic Statistics
        log.info("📊 BASIC STATISTICS:")
        log.info(f"  Total detections: {metrics.get('total_detections', 'N/A')}")
        log.info(f"  Unique frames: {metrics.get('unique_frames', 'N/A')}")
        log.info(f"  Unique tracks: {metrics.get('unique_tracks', 'N/A')}")
        log.info(".2f")
        log.info(".1%")

        # Track Consistency
        log.info("\n🔄 TRACK CONSISTENCY:")
        log.info(".1f")
        log.info(
            f"  Track length range: {metrics.get('min_track_length', 'N/A')} - {metrics.get('max_track_length', 'N/A')} frames"
        )
        log.info(f"  Tracks > 100 frames: {metrics.get('tracks_gt_100_frames', 'N/A')}")
        log.info(f"  Tracks > 500 frames: {metrics.get('tracks_gt_500_frames', 'N/A')}")
        log.info(".2f")

        # Detection Quality
        if any(
            k.startswith(("mean_bbox", "mean_detection_confidence"))
            for k in metrics.keys()
        ):
            log.info("\n🎯 DETECTION QUALITY:")
            if "mean_bbox_area" in metrics:
                log.info(".1f")
            if "mean_detection_confidence" in metrics:
                log.info(".3f")

        # Team Classification
        if "team_distribution" in metrics:
            log.info("\n👥 TEAM CLASSIFICATION:")
            team_dist = metrics["team_distribution"]
            for team, count in team_dist.items():
                log.info(f"  Team '{team}': {count} detections")
            log.info(".1%")

        # Pitch Coordinates
        if "pitch_coordinates_reasonable" in metrics:
            log.info("\n⚽ PITCH COORDINATES:")
            log.info(
                f"  Coordinate system appears reasonable: {metrics['pitch_coordinates_reasonable']}"
            )
            log.info(".1f")
            log.info(
                f"  Total coordinate outliers: {metrics.get('total_pitch_outliers', 'N/A')}"
            )

        # Temporal Consistency
        if "mean_tracks_per_frame" in metrics:
            log.info("\n⏱️  TEMPORAL CONSISTENCY:")
            log.info(
                f"  Mean tracks per frame: {metrics.get('mean_tracks_per_frame', 'N/A'):.1f}"
            )
            if "mean_track_velocity" in metrics:
                log.info(
                    f"  Mean track velocity: {metrics.get('mean_track_velocity', 'N/A'):.2f} m/frame"
                )
                log.info(
                    f"  Unrealistic velocities: {metrics.get('unrealistic_velocities', 'N/A')}"
                )

        log.info("=" * 60)
