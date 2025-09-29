"""Keypoints visualization classes for TrackLab."""

from typing import Any, Optional
import numpy as np

from .visualizer import DetectionVisualizer
from tracklab.utils.cv2 import draw_keypoints


class DefaultKeypoints(DetectionVisualizer):
    """Default keypoints visualizer that draws keypoints with confidence filtering."""

    def __init__(self, threshold: float = 0.4, print_confidence: bool = False) -> None:
        """Initialize the default keypoints visualizer.

        Args:
            threshold: Confidence threshold for displaying keypoints.
            print_confidence: Whether to print confidence scores on keypoints.
        """
        self.threshold = threshold
        self.print_confidence = print_confidence
        super().__init__()

    def draw_detection(
        self,
        image: np.ndarray,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw keypoints for both ground truth and predictions.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection with keypoints.
            detection_gt: Ground truth detection with keypoints.
            metric: Matching metric (unused).
        """
        if detection_gt is not None:
            color_kp = self.color(detection_gt, is_prediction=False)
            if color_kp:
                draw_keypoints(
                    detection_gt,
                    image,
                    color_kp,
                    threshold=self.threshold,
                )
        if detection_pred is not None:
            color_kp = self.color(detection_pred, is_prediction=True)
            if color_kp:
                draw_keypoints(
                    detection_pred,
                    image,
                    color_kp,
                    threshold=self.threshold,
                    print_confidence=self.print_confidence,
                )


class FullKeypoints(DefaultKeypoints):
    """Keypoints visualizer that shows all keypoints with confidence scores."""

    def __init__(self) -> None:
        """Initialize with no threshold and confidence display enabled."""
        super().__init__(threshold=0.0, print_confidence=True)
