"""RTMLib-based bounding box detector for TrackLab."""

from typing import Any, Dict, List

import accelerate
import rtmlib
import cv2
import pandas as pd

from hydra.utils import instantiate

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh


class RTMLibDetector(ImageLevelModule):
    """RTMLib-based bounding box detector for person detection.

    This module integrates RTMLib models for detecting persons in images.
    It provides a lightweight detection solution with ONNX runtime backend
    support for efficient inference.
    """

    input_columns: List[str] = []
    output_columns: List[str] = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, device: str, model: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize the RTMLib detector.

        Args:
            device: Device to run inference on (e.g., 'cpu', 'cuda').
            model: Model configuration dictionary for instantiation.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size=1)
        self.device = device
        self.model = instantiate(model, device=self.device, backend="onnxruntime")
        self.id: int = 0

    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Dict[str, Any]:
        """Preprocess image for RTMLib inference.

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
        """Process image and extract detections using RTMLib.

        Args:
            batch: Preprocessed batch data (unused).
            detections: Input detections DataFrame (unused).
            metadatas: Image metadata DataFrame.

        Returns:
            List of detection Series containing bounding box information.
        """
        image = cv2.imread(metadatas["file_path"].values[0])  # BGR not RGB !
        shape = (image.shape[1], image.shape[0])
        bboxes = self.model(image)
        detections_list = []
        for bbox in bboxes:
            detections_list.append(
                pd.Series(
                    dict(
                        image_id=metadatas["id"].values[0],
                        bbox_ltwh=ltrb_to_ltwh(bbox, shape),
                        bbox_conf=1.0,
                        video_id=metadatas["video_id"].values[0],
                        category_id=1,
                    ),
                    name=self.id,
                )
            )
            self.id += 1
        return detections_list
