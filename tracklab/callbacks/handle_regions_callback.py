import cv2
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Tuple

from tracklab.callbacks import Callback

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine


class IgnoredRegions(Callback):
    """Callback for marking detections in ignored regions.

    This callback identifies detections that overlap significantly with predefined
    ignore regions in the video frames and marks them as ignored.
    """

    def __init__(self, max_intersection: float = 0.9):
        """Initialize the IgnoredRegions callback.

        Args:
            max_intersection: Maximum fraction of detection area that can overlap
                with an ignore region before being marked as ignored.
        """
        self.max_intersection = max_intersection

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        """Mark detections in ignored regions at the end of video processing.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            detections: DataFrame containing detection results.
            image_pred: DataFrame containing image predictions.
        """
        image_metadatas = engine.img_metadatas[
            engine.img_metadatas.video_id == video_idx
        ]
        """
        detections.insert(-1, "ignored", detections.apply(
            lambda x: self.mark_ignored(x, image_metadatas), axis=1
        ))
        """
        if len(detections):
            detections["ignored"] = detections.apply(
                lambda x: self.mark_ignored(x, image_metadatas), axis=1
            )
        else:
            detections["ignored"] = pd.NA

    def mark_ignored(self, detection, image_metadatas: pd.DataFrame) -> bool:
        """Mark a detection as ignored if it overlaps with ignore regions.

        Args:
            detection: Detection data containing bbox and image_id.
            image_metadatas: DataFrame with image metadata including ignore regions.

        Returns:
            True if the detection should be ignored, False otherwise.
        """
        if hasattr(image_metadatas, "ignore_regions_x") and hasattr(
            image_metadatas, "ignore_regions_y"
        ):
            image_metadata = image_metadatas.loc[detection.image_id]
            return self.compute_iou(
                detection.bbox.ltrb(rounded=True),
                image_metadata.ignore_regions_x,
                image_metadata.ignore_regions_y,
            )
        return False

    def compute_iou(
        self,
        bbox_ltrb: np.ndarray,
        ignore_regions_x: Tuple,
        ignore_regions_y: Tuple,
    ) -> bool:
        """Compute intersection over union with ignore regions.

        Args:
            bbox_ltrb: Bounding box coordinates [left, top, right, bottom].
            ignore_regions_x: X coordinates of ignore region polygons.
            ignore_regions_y: Y coordinates of ignore region polygons.

        Returns:
            True if the detection area exceeds the threshold in an ignore region,
            False otherwise.
        """
        l, t, r, b = bbox_ltrb

        for ignore_region_x, ignore_region_y in zip(ignore_regions_x, ignore_regions_y):
            polygon_points = (
                np.array([ignore_region_x, ignore_region_y]).round().T.astype(int)
            )
            image_dim_max = (
                max(b, polygon_points[:, 1].max()),
                max(r, polygon_points[:, 0].max()),
            )  # height, width
            bbox_mask = np.zeros(image_dim_max, dtype=np.uint8)
            bbox_mask[t:b, l:r] = 1
            ignore_mask = np.zeros(image_dim_max, dtype=np.uint8)
            ignore_mask = cv2.fillPoly(ignore_mask, [polygon_points], (1,))
            intersection_area = np.logical_and(bbox_mask, ignore_mask).sum()
            bbox_area = (r - l) * (b - t)
            if intersection_area > self.max_intersection * bbox_area:
                return True
        return False
