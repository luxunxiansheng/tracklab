"""Base visualizer classes for TrackLab."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from distinctipy import get_colors, get_rgb256


class Visualizer(ABC):
    """Abstract base class for visualization components."""

    @abstractmethod
    def draw_frame(
        self,
        image: np.ndarray,
        detections_pred: Optional[pd.DataFrame],
        detections_gt: Optional[pd.DataFrame],
        image_pred: pd.DataFrame,
        image_gt: pd.DataFrame,
    ) -> np.ndarray:
        """Draw a single frame with detections and ground truth.

        Args:
            image: Input image array.
            detections_pred: Predicted detections DataFrame.
            detections_gt: Ground truth detections DataFrame.
            image_pred: Predicted image metadata DataFrame.
            image_gt: Ground truth image metadata DataFrame.

        Returns:
            Annotated image array.
        """
        pass

    def preprocess(
        self,
        video_detections_pred: Optional[pd.DataFrame],
        video_detections_gt: Optional[pd.DataFrame],
        video_image_pred: pd.DataFrame,
        video_image_gt: pd.DataFrame,
    ) -> None:
        """Preprocess video data for visualization.

        Args:
            video_detections_pred: Predicted detections for the entire video.
            video_detections_gt: Ground truth detections for the entire video.
            video_image_pred: Predicted image metadata for the video.
            video_image_gt: Ground truth image metadata for the video.
        """
        pass

    def post_init(self, **kwargs: Any) -> None:
        """Initialize additional attributes after construction.

        Args:
            **kwargs: Additional attributes to set.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class ImageVisualizer(Visualizer, ABC):
    """Abstract base class for image-level visualizers."""

    pass


@lru_cache(maxsize=None)
def get_fixed_colors(N: int) -> Any:
    """Get a fixed set of distinct colors.

    Args:
        N: Number of colors to generate.

    Returns:
        Color array.
    """
    return get_colors(N)


class DetectionVisualizer(Visualizer, ABC):
    """Abstract base class for detection visualizers."""

    def __init__(self) -> None:
        """Initialize the detection visualizer."""
        self.colors: Optional[Dict[str, Any]] = None

    def post_init(self, colors: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize colors and other attributes.

        Args:
            colors: Color configuration dictionary.
            **kwargs: Additional attributes.
        """
        super().post_init(**kwargs)
        self.colors = colors
        if "cmap" in colors and isinstance(colors["cmap"], int):
            cmap = get_fixed_colors(colors["cmap"])
        else:
            cmap = plt.get_cmap(colors["cmap"])(np.linspace(0, 1, 256))  # type: ignore
        self.cmap = [get_rgb256(i) for i in cmap]

    def draw_frame(
        self,
        image: np.ndarray,
        detections_pred: Optional[pd.DataFrame],
        detections_gt: Optional[pd.DataFrame],
        image_pred: pd.DataFrame,
        image_gt: pd.DataFrame,
    ) -> np.ndarray:
        """Draw detections on the frame with matching between pred and gt.

        Args:
            image: Input image.
            detections_pred: Predicted detections.
            detections_gt: Ground truth detections.
            image_pred: Predicted image metadata.
            image_gt: Ground truth image metadata.

        Returns:
            Annotated image.
        """
        if detections_pred is not None and not detections_pred.empty:
            bbox_pred = torch.tensor(np.stack(detections_pred.bbox.ltrb()))  # type: ignore
        else:
            bbox_pred = torch.empty((0, 4))
        if detections_gt is not None and not detections_gt.empty:
            bbox_gt = torch.tensor(np.stack(detections_gt.bbox.ltrb()))  # type: ignore
        else:
            bbox_gt = torch.empty((0, 4))
        cost_matrix = box_iou(bbox_pred, bbox_gt)

        row_idxs, col_idxs = linear_sum_assignment(1 - cost_matrix)
        gt_rest = set(range(len(bbox_gt))) - set(col_idxs)
        for i in range(max(len(bbox_pred), len(bbox_gt))):
            if i not in row_idxs:
                metric = None
                if len(bbox_pred) < len(bbox_gt):
                    pred = None
                    gt_id = gt_rest.pop()
                    gt = detections_gt.iloc[gt_id] if detections_gt is not None else None  # type: ignore
                else:
                    pred = detections_pred.iloc[i] if detections_pred is not None else None  # type: ignore
                    gt = None
            else:
                pred = detections_pred.iloc[i] if detections_pred is not None else None  # type: ignore
                row_idx = np.min(np.nonzero(row_idxs == i)[0])  # row_idxs.index(i)
                gt = detections_gt.iloc[col_idxs[row_idx]] if detections_gt is not None else None  # type: ignore
                metric = cost_matrix[i, col_idxs[row_idx]]
            self.draw_detection(image, pred, gt, metric)
        return image

    @abstractmethod
    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        pass

    def color(self, detection, is_prediction, color_type="default"):
        assert self.colors is not None
        if color_type not in self.colors:
            raise ValueError(
                f"{color_type} not declared in the colors dict for visualization"
            )

        # Check if track_id column exists (it won't exist if tracking is disabled)
        has_track_id = hasattr(detection, "track_id") and "track_id" in detection.index

        if not has_track_id or pd.isna(detection.track_id):
            color = self.colors[color_type].get("no_id", None)
        else:
            cmap_key = "prediction" if is_prediction else "ground_truth"
            if self.colors[color_type][cmap_key] == "track_id":
                color = self.cmap[(int(detection.track_id) - 1) % len(self.cmap)]
            else:
                color = self.colors[color_type][cmap_key]
        return color
