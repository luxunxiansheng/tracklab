"""Image-level visualization classes for TrackLab."""

from typing import Any, Optional
import numpy as np
import pandas as pd

from .visualizer import ImageVisualizer
from tracklab.utils.cv2 import print_count_frame, draw_ignore_region


class FrameCount(ImageVisualizer):
    """Image visualizer that displays frame count information."""

    def draw_frame(
        self,
        image: np.ndarray,
        detections_pred: Optional[pd.DataFrame],
        detections_gt: Optional[pd.DataFrame],
        image_pred: pd.DataFrame,
        image_gt: pd.DataFrame,
    ) -> np.ndarray:
        """Draw frame count on the image.

        Args:
            image: Image to draw on.
            detections_pred: Predicted detections (unused).
            detections_gt: Ground truth detections (unused).
            image_pred: Predicted image metadata (unused).
            image_gt: Ground truth image metadata.

        Returns:
            Image with frame count drawn.
        """
        print_count_frame(image, image_gt.frame, nframes=image_gt.nframes)
        return image


class IgnoreRegions(ImageVisualizer):
    """Image visualizer that draws ignore regions."""

    def draw_frame(
        self,
        image: np.ndarray,
        detections_pred: Optional[pd.DataFrame],
        detections_gt: Optional[pd.DataFrame],
        image_pred: pd.DataFrame,
        image_gt: pd.DataFrame,
    ) -> np.ndarray:
        """Draw ignore regions on the image.

        Args:
            image: Image to draw on.
            detections_pred: Predicted detections (unused).
            detections_gt: Ground truth detections (unused).
            image_pred: Predicted image metadata.
            image_gt: Ground truth image metadata (unused).

        Returns:
            Image with ignore regions drawn.
        """
        draw_ignore_region(image, image_pred)
        return image
