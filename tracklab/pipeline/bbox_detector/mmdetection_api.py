"""MMDetection-based bounding box detector for TrackLab."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from mim import get_model_info
from mim.utils import get_installed_path
from mmcv import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis.inference import init_detector

from mmengine.dataset import default_collate
from tracklab.pipeline import ImageLevelModule
from tracklab.utils import ltrb_to_ltwh
from tracklab.utils.openmmlab import get_checkpoint


class MMDetection(ImageLevelModule):
    """MMDetection-based bounding box detector for person detection.

    This module integrates MMDetection models for detecting persons in images.
    It supports various MMDetection model configurations and provides
    standardized output format for the TrackLab pipeline.
    """

    collate_fn = default_collate
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
        config_name: str,
        path_to_checkpoint: str,
        device: str,
        batch_size: int,
        min_confidence: float,
        **kwargs: Any,
    ) -> None:
        """Initialize the MMDetection detector.

        Args:
            config_name: Name of the MMDetection model configuration.
            path_to_checkpoint: Path to the model checkpoint file.
            device: Device to run inference on (e.g., 'cuda:0', 'cpu').
            batch_size: Batch size for processing images.
            min_confidence: Minimum confidence threshold for detections.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size)

        self.device = device
        self.min_confidence = min_confidence
        model_df = get_model_info(package="mmdet", configs=[config_name])
        if len(model_df) != 1:
            raise ValueError(f"Multiple values found for config_name: {config_name}")

        download_url = model_df.weight.item()
        package_path = Path(get_installed_path("mmdet"))
        path_to_config = package_path / ".mim" / model_df.config.item()
        get_checkpoint(path_to_checkpoint, download_url)
        self.model = init_detector(
            str(path_to_config), path_to_checkpoint, device=device
        )
        self.test_pipeline = get_test_pipeline_cfg(self.model.cfg.copy())  # type: ignore
        self.test_pipeline[0].type = "mmdet.LoadImageFromNDArray"
        self.test_pipeline = Compose(self.test_pipeline)
        self.current_id: int = 0

    @torch.no_grad()
    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Any:
        """Preprocess image for MMDetection inference.

        Args:
            image: Input image array.
            detections: Detection DataFrame (unused in preprocessing).
            metadata: Image metadata series.

        Returns:
            Preprocessed data for model inference.
        """
        return self.test_pipeline(dict(img=image, img_id=0))  # type: ignore

    @torch.no_grad()
    def process(
        self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> pd.DataFrame:
        """Process batch of images and extract detections.

        Args:
            batch: Preprocessed batch data.
            detections: Input detections DataFrame (unused).
            metadatas: Image metadata DataFrame.

        Returns:
            DataFrame containing detected bounding boxes.
        """
        results = self.model.test_step(batch)  # type: ignore
        img_metas = batch["data_samples"]
        shapes = [(x.ori_shape[1], x.ori_shape[0]) for x in batch["data_samples"]]
        detections_list = []
        for preds, image_shape, (_, metadata) in zip(
            results, shapes, metadatas.iterrows()
        ):
            instances = preds.pred_instances
            for score, bbox, label in zip(
                instances.scores, instances.bboxes, instances.labels
            ):
                if score < self.min_confidence or label != 0:
                    continue
                detections_list.append(
                    pd.Series(
                        dict(
                            image_id=metadata.name,
                            video_id=metadata.video_id,
                            bbox_ltwh=ltrb_to_ltwh(bbox.cpu().numpy(), image_shape),
                            bbox_conf=float(score.item()),
                            category_id=1,  # 'person' class
                        ),
                        name=self.current_id,
                    )
                )
                self.current_id += 1

        return pd.DataFrame(detections_list)
