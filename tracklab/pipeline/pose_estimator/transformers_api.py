"""Transformers-based pose estimation module for TrackLab."""

from typing import Any, Dict, List

import torch
import pandas as pd
import numpy as np

try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation  # type: ignore
except ImportError:
    AutoProcessor = None  # type: ignore
    VitPoseForPoseEstimation = None  # type: ignore

from tracklab.pipeline import DetectionLevelModule


class VITPose(DetectionLevelModule):
    """VIT-Pose estimation module using Transformers library.

    This module performs top-down pose estimation using Vision Transformer
    models, taking bounding box detections as input and estimating keypoints
    for each detected person.
    """

    input_columns: List[str] = []
    output_columns: List[str] = ["keypoints_xyc", "keypoints_conf"]

    def __init__(
        self, device: str, batch_size: int, model_name: str, **kwargs: Any
    ) -> None:
        """Initialize the VIT-Pose module.

        Args:
            device: Device to run inference on.
            batch_size: Batch size for processing.
            model_name: Name of the VIT-Pose model to use.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size)
        self.device = device
        self.image_processor = AutoProcessor.from_pretrained(f"usyd-community/{model_name}")  # type: ignore
        self.model = VitPoseForPoseEstimation.from_pretrained(f"usyd-community/{model_name}", device_map=device)  # type: ignore

    @torch.no_grad()
    def preprocess(
        self, image: Any, detection: pd.Series, metadata: pd.Series
    ) -> Dict[str, Any]:
        """Preprocess image and detection for pose estimation.

        Args:
            image: Input image array.
            detection: Detection series containing bounding box.
            metadata: Image metadata series.

        Returns:
            Dictionary containing preprocessed image and bounding box.
        """
        return {"image": image, "bbox": detection["bbox_ltwh"]}

    @torch.no_grad()
    def process(
        self, batch: Dict[str, Any], detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> pd.DataFrame:
        """Process batch and extract keypoints for detections.

        Args:
            batch: Preprocessed batch data containing images and boxes.
            detections: Detection DataFrame to update with keypoints.
            metadatas: Image metadata DataFrame.

        Returns:
            Updated detections DataFrame with keypoints.
        """
        boxes = batch["bbox"].unsqueeze(1)
        inputs = self.image_processor(batch["image"], boxes=boxes, return_tensors="pt").to(self.device)  # type: ignore
        outputs = self.model(**inputs, dataset_index=torch.tensor(boxes.shape[0] * [0], device=self.device))  # type: ignore
        pose_results = self.image_processor.post_process_pose_estimation(outputs, boxes=boxes)  # type: ignore
        keypoints_xy = np.array(
            [res[0]["keypoints"].cpu().numpy() for res in pose_results]
        )
        keypoints_c = np.array([res[0]["scores"].cpu().numpy() for res in pose_results])
        detections["keypoints_xyc"] = list(
            np.concatenate([keypoints_xy, keypoints_c[..., np.newaxis]], axis=-1)
        )
        detections["keypoints_conf"] = list(keypoints_c.mean(axis=1))
        return detections
