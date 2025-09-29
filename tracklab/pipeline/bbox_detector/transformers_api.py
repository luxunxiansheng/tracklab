"""Transformers-based RT-DETR bounding box detector for TrackLab."""

from typing import Any, List

import torch
import pandas as pd

try:
    from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor  # type: ignore
except ImportError:
    RTDetrForObjectDetection = None  # type: ignore
    RTDetrV2ForObjectDetection = None  # type: ignore
    RTDetrImageProcessor = None  # type: ignore

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh


class RTDetr(ImageLevelModule):
    """RT-DETR (Real-Time DETR) detector using Transformers library.

    This module integrates RT-DETR models from the Transformers library
    for efficient real-time object detection. It supports both RT-DETR
    and RT-DETRv2 architectures for person detection tasks.
    """

    input_columns: List[str] = []
    output_columns: List[str] = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(
        self,
        device: str,
        batch_size: int,
        model_name: str,
        min_confidence: float,
        **kwargs: Any,
    ) -> None:
        """Initialize the RT-DETR detector.

        Args:
            device: Device to run inference on (e.g., 'cpu', 'cuda').
            batch_size: Batch size for processing images.
            model_name: Name of the RT-DETR model to use.
            min_confidence: Minimum confidence threshold for detections.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size)
        self.device = device
        self.image_processor = RTDetrImageProcessor.from_pretrained(f"PekingU/{model_name}")  # type: ignore
        if "v2" in model_name:
            self.model = RTDetrV2ForObjectDetection.from_pretrained(f"PekingU/{model_name}", device_map=device)  # type: ignore
        else:
            self.model = RTDetrForObjectDetection.from_pretrained(f"PekingU/{model_name}", device_map=device)  # type: ignore
        self.min_confidence = min_confidence
        self.id: int = 0

    @torch.no_grad()
    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Any:
        """Preprocess image for RT-DETR inference.

        Args:
            image: Input image array.
            detections: Detection DataFrame (unused in preprocessing).
            metadata: Image metadata series.

        Returns:
            Preprocessed image data.
        """
        return image

    @torch.no_grad()
    def process(
        self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> List[pd.Series]:
        """Process batch of images and extract detections.

        Args:
            batch: Batch of preprocessed images.
            detections: Input detections DataFrame (unused).
            metadatas: Image metadata DataFrame.

        Returns:
            List of detection Series containing bounding box information.
        """
        images = self.image_processor(batch, return_tensors="pt")
        outputs = self.model(**images)
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=[batch.shape[1:3]] * batch.shape[0],
            threshold=self.min_confidence,
        )
        detections_list = []
        for i, result in enumerate(results):
            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                if label == 0:
                    detections_list.append(
                        pd.Series(
                            dict(
                                image_id=metadatas["id"].values[i],
                                bbox_ltwh=ltrb_to_ltwh(
                                    box.numpy(), (batch.shape[2], batch.shape[1])
                                ),
                                bbox_conf=score.item(),
                                video_id=metadatas["video_id"].values[i],
                                category_id=1,
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections_list
