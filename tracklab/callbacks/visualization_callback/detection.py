"""Detection visualization classes for TrackLab."""

from typing import Any, List, Optional, Tuple, Union
import cv2
import numpy as np

from .visualizer import DetectionVisualizer
from tracklab.utils.cv2 import draw_bbox, draw_bbox_stats, draw_text


class DefaultDetection(DetectionVisualizer):
    """Default detection visualizer that draws bounding boxes with optional ID and confidence."""

    def __init__(self, print_id: bool = True, print_confidence: bool = False) -> None:
        """Initialize the default detection visualizer.

        Args:
            print_id: Whether to print track IDs on detections.
            print_confidence: Whether to print confidence scores on detections.
        """
        super().__init__()
        self.print_id = print_id
        self.print_confidence = print_confidence

    def draw_detection(
        self,
        image: np.ndarray,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw detection bounding boxes on the image.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection data.
            detection_gt: Ground truth detection data.
            metric: Matching metric between pred and gt (unused).
        """
        for detection, is_pred in zip([detection_pred, detection_gt], [True, False]):
            if detection is not None:
                color_bbox = self.color(detection, is_prediction=is_pred)
                if color_bbox:
                    # Check if track_id exists before trying to print it
                    has_track_id = (
                        hasattr(detection, "track_id") and "track_id" in detection.index
                    )
                    should_print_id = self.print_id and has_track_id

                    draw_bbox(
                        detection,
                        image,
                        color_bbox,
                        print_id=should_print_id,
                        print_confidence=self.print_confidence,
                    )


class FullDetection(DefaultDetection):
    """Detection visualizer that shows both track IDs and confidence scores."""

    def __init__(self) -> None:
        """Initialize with both ID and confidence display enabled."""
        super().__init__(print_id=True, print_confidence=True)


class DebugDetection(DetectionVisualizer):
    """Debug detection visualizer that classifies detections by correctness.

    Detections are classified by colors:
        - Green: True Positive
        - Yellow: False Positive
        - Red: False Negative
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the debug detection visualizer.

        Args:
            threshold: IoU threshold for considering a detection as true positive.
        """
        self.threshold = threshold
        super().__init__()

    def draw_detection(
        self,
        image: np.ndarray,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw debug detection visualization with correctness classification.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection data.
            detection_gt: Ground truth detection data.
            metric: IoU metric between pred and gt.
        """
        if detection_gt is not None:  # GT exists
            if detection_pred is None:  # pred is not detected
                draw_bbox(detection_gt, image, (255, 0, 0))  # FN - Red
            elif (
                metric is not None
                and metric > self.threshold
                and hasattr(detection_pred, "track_id")
                and "track_id" in detection_pred.index
                and not np.isnan(detection_pred.track_id)
            ):  # pred is correct
                draw_bbox(detection_pred, image, (0, 255, 0))  # TP - Green
            else:  # pred is not correct
                draw_bbox(detection_gt, image, (255, 0, 0))  # FN - Red
        elif (
            detection_pred is not None
            and hasattr(detection_pred, "track_id")
            and "track_id" in detection_pred.index
            and not np.isnan(detection_pred.track_id)
        ):  # no GT and pred is assigned
            draw_bbox(detection_pred, image, (255, 255, 0))  # FP - Yellow


class DetectionStats(DetectionVisualizer):
    """Detection visualizer that shows detailed tracking statistics."""

    def __init__(
        self,
        print_stats: List[str] = [
            "state",
            "hits",
            "age",
            "time_since_update",
            "matched_with",
        ],  # FIXME "costs" is too long for display
    ) -> None:
        """Initialize the detection stats visualizer.

        Args:
            print_stats: List of statistics to display on detections.
        """
        self.print_stats = print_stats
        super().__init__()

    def draw_detection(
        self,
        image: np.ndarray,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw detection with tracking statistics.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection data.
            detection_gt: Ground truth detection data (unused).
            metric: Matching metric (unused).
        """
        if detection_pred is not None:
            color_bbox = self.color(detection_pred, is_prediction=True)
            if color_bbox:
                draw_bbox_stats(
                    detection_pred,
                    image,
                    self.print_stats,
                    bbox_color=color_bbox,
                )


class SimpleDetectionStats(DetectionStats):
    """Simplified detection stats visualizer with essential tracking information."""

    def __init__(self) -> None:
        """Initialize with basic tracking statistics."""
        super().__init__(print_stats=["state", "hits", "age", "time_since_update"])


class EllipseDetection(DetectionVisualizer):
    """Detection visualizer that draws elliptical bounding boxes."""

    def __init__(self, print_id: bool = True) -> None:
        """Initialize the ellipse detection visualizer.

        Args:
            print_id: Whether to print track IDs on detections.
        """
        self.print_id = print_id
        super().__init__()

    def draw_detection(
        self,
        image: np.ndarray,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw elliptical detection visualization.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection data.
            detection_gt: Ground truth detection data.
            metric: Matching metric (unused).
        """
        for detection, is_pred in zip([detection_pred, detection_gt], [True, False]):
            if detection is not None:
                color = self.color(detection, is_prediction=is_pred)
                if color:
                    x1, y1, x2, y2 = detection.bbox.ltrb()
                    center = (int((x1 + x2) / 2), int(y2))
                    width = x2 - x1
                    cv2.ellipse(
                        image,
                        center=center,
                        axes=(int(width), int(0.35 * width)),
                        angle=0.0,
                        startAngle=-45.0,
                        endAngle=235.0,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    if self.print_id and hasattr(detection, "track_id"):
                        draw_text(
                            image,
                            f"ID: {int(detection.track_id)}",
                            center,
                            fontFace=1,
                            fontScale=1,
                            thickness=1,
                            alignH="c",
                            alignV="c",
                            color_bg=color,
                            color_txt=None,
                            alpha_bg=1,
                        )
