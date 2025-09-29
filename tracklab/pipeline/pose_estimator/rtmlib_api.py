"""RTMLib-based pose estimation modules for TrackLab."""

from typing import Any, Dict, List

import cv2
import pandas as pd
import numpy as np
import accelerate
import rtmlib

from hydra.utils import instantiate

from tracklab.utils.coordinates import generate_bbox_from_keypoints
from tracklab.pipeline import ImageLevelModule


class RTMPose(ImageLevelModule):
    """RTM-Pose estimation module using RTMLib.

    This module performs top-down pose estimation, taking bounding box
    detections as input and estimating keypoints for each detected person.
    """

    input_columns: List[str] = []
    output_columns: List[str] = ["keypoints_xyc", "keypoints_conf"]

    def __init__(self, device: str, model: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize the RTM-Pose module.

        Args:
            device: Device to run inference on.
            model: Model configuration dictionary for instantiation.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size=1)
        self.device = device
        self.model = instantiate(model, device=self.device, backend="onnxruntime")

    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Dict[str, Any]:
        """Preprocess image and detections for pose estimation.

        Args:
            image: Input image array.
            detections: Detection DataFrame containing bounding boxes.
            metadata: Image metadata series.

        Returns:
            Empty dictionary as preprocessing is handled in process method.
        """
        return {}

    def process(
        self, batch: Dict[str, Any], detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> pd.DataFrame:
        """Process detections and extract keypoints.

        Args:
            batch: Preprocessed batch data (unused).
            detections: Detection DataFrame to update with keypoints.
            metadatas: Image metadata DataFrame.

        Returns:
            Updated detections DataFrame with keypoints.
        """
        image = cv2.imread(metadatas["file_path"].values[0])  # BGR not RGB !
        bboxes = detections.bbox.ltrb().values
        keypoints, scores = self.model(image, bboxes)
        detections["keypoints_xyc"] = list(
            np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
        )
        detections["keypoints_conf"] = list(np.mean(scores, axis=1))
        return detections


class RTMO(ImageLevelModule):
    """RTM-Object detection and pose estimation module using RTMLib.

    This module performs bottom-up pose estimation and object detection,
    detecting all persons and their keypoints in images simultaneously.
    """

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

    def __init__(
        self, device: str, model: Dict[str, Any], min_confidence: float, **kwargs: Any
    ) -> None:
        """Initialize the RTMO module.

        Args:
            device: Device to run inference on.
            model: Model configuration dictionary for instantiation.
            min_confidence: Minimum confidence threshold for detections.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size=1)
        self.device = device
        self.model = instantiate(model, device=self.device, backend="onnxruntime")
        self.min_confidence = min_confidence
        self.id: int = 0

    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Dict[str, Any]:
        """Preprocess image for pose estimation.

        Args:
            image: Input image array.
            detections: Detection DataFrame (unused in preprocessing).
            metadata: Image metadata series.

        Returns:
            Empty dictionary as preprocessing is handled in process method.
        """
        return {}

    def process(
        self, batch: Dict[str, Any], detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> List[pd.Series]:
        """Process image and extract pose detections.

        Args:
            batch: Preprocessed batch data (unused).
            detections: Input detections DataFrame (unused).
            metadatas: Image metadata DataFrame.

        Returns:
            List of detection Series with pose information.
        """
        image = cv2.imread(metadatas["file_path"].values[0])  # BGR not RGB !
        shape = (image.shape[1], image.shape[0])
        keypoints, scores = self.model(image)
        detections_list = []
        for kps, score in zip(keypoints, scores):
            conf = np.mean(score)
            kps = np.concatenate([kps, score[..., np.newaxis]], axis=-1)
            if conf >= self.min_confidence:
                detections_list.append(
                    pd.Series(
                        dict(
                            image_id=metadatas["id"].values[0],
                            bbox_ltwh=generate_bbox_from_keypoints(
                                kps, (0.1, 0.03, 0.1), shape
                            ),
                            bbox_conf=conf,
                            keypoints_xyc=kps,
                            keypoints_conf=conf,
                            video_id=metadatas["video_id"].values[0],
                            category_id=1,
                        ),
                        name=self.id,
                    )
                )
                self.id += 1
        return detections_list
