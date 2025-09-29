"""YOLO Ultralytics pose estimation module for TrackLab."""

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore

log = logging.getLogger(__name__)


def collate_fn(
    batch: List[Tuple[int, Dict[str, Any]]],
) -> Tuple[List[int], Tuple[List[Any], List[Tuple[int, int]]]]:
    """Collate function for batching pose estimation data.

    Args:
        batch: List of tuples containing index and data dictionary.

    Returns:
        Tuple of indices and batched images with shapes.
    """
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOUltralyticsPose(ImageLevelModule):
    """YOLO Ultralytics pose estimation module for TrackLab.

    This module performs pose estimation using YOLO models from Ultralytics,
    detecting persons and their keypoints in images.
    """

    collate_fn = collate_fn
    input_columns: List[str] = []
    output_columns: List[str] = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
        "keypoints_xyc",
        "keypoints_conf",
    ]

    def __init__(self, cfg: Any, device: str, batch_size: int, **kwargs: Any) -> None:
        """Initialize the YOLO Ultralytics pose module.

        Args:
            cfg: Configuration object containing model path and parameters.
            device: Device to run inference on.
            batch_size: Batch size for processing.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.path_to_checkpoint)  # type: ignore
        self.model.to(device)
        self.id = 0

    @torch.no_grad()
    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Dict[str, Any]:
        """Preprocess image for pose estimation.

        Args:
            image: Input image array.
            detections: Detection DataFrame (unused for image-level processing).
            metadata: Image metadata series.

        Returns:
            Dictionary containing processed image and shape.
        """
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(
        self,
        batch: Tuple[List[Any], List[Tuple[int, int]]],
        detections: pd.DataFrame,
        metadatas: pd.DataFrame,
    ) -> List[pd.Series]:
        """Process batch and extract pose detections.

        Args:
            batch: Tuple of images and their shapes.
            detections: Detection DataFrame (unused for image-level processing).
            metadatas: Image metadata DataFrame.

        Returns:
            List of detection series with pose information.
        """
        images, shapes = batch
        results_by_image = self.model(images, verbose=False)  # type: ignore
        detections_list = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox, keypoints in zip(
                results.boxes.cpu().numpy(), results.keypoints.cpu().numpy()
            ):
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    detections_list.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                                keypoints_xyc=keypoints.data[0],
                                keypoints_conf=np.mean(keypoints.data[0, :, 2]),
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections_list
