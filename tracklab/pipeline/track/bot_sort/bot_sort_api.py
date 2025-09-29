"""BoT-SORT multi-object tracking module for TrackLab."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
from tracklab.utils.cv2 import cv2_load_image
from . import bot_sort as bot_sort

log = logging.getLogger(__name__)


class BotSORT(ImageLevelModule):
    """BoT-SORT multi-object tracking module for TrackLab.

    This module performs multi-object tracking using the BoT-SORT algorithm,
    which combines motion and appearance cues for robust tracking.
    """

    input_columns: List[str] = [
        "bbox_ltwh",
        "bbox_conf",
        "category_id",
    ]
    output_columns: List[str] = ["track_id", "track_bbox_ltwh", "track_bbox_conf"]

    def __init__(self, cfg: Any, device: str, **kwargs: Any) -> None:
        """Initialize the BoT-SORT module.

        Args:
            cfg: Configuration object containing tracking parameters.
            device: Device to run inference on.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset the tracker state to start tracking in a new video."""
        self.model = bot_sort.BoTSORT(
            Path(self.cfg.model_weights),
            self.device,
            self.cfg.fp16,
            **self.cfg.hyperparams,
        )

    @torch.no_grad()
    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Dict[str, Union[List[Any], np.ndarray]]:
        """Preprocess detections for tracking.

        Args:
            image: Input image array (unused for tracking preprocessing).
            detections: Detection DataFrame.
            metadata: Image metadata series.

        Returns:
            Dictionary containing processed detection inputs.
        """
        processed_detections = []
        if len(detections) == 0:
            return {"input": []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)  # type: ignore
            processed_detections.append(np.array([*ltrb, conf, cls, tracklab_id]))
        return {"input": np.stack(processed_detections)}

    @torch.no_grad()
    def process(
        self, batch: Dict[str, Any], detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> Union[pd.DataFrame, List[Any]]:
        """Process batch and perform tracking.

        Args:
            batch: Preprocessed batch data.
            detections: Detection DataFrame.
            metadatas: Image metadata DataFrame.

        Returns:
            DataFrame with tracking results or empty list.
        """
        if len(detections) == 0:
            return []
        inputs = batch["input"][0]  # Nx7 [l,t,r,b,conf,class,tracklab_id]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        image = cv2_load_image(metadatas["file_path"].values[0])
        results = self.model.update(inputs, image)
        results = np.asarray(results)  # N'x8 [l,t,r,b,track_id,class,conf,idx]
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 7].astype(int))
            assert set(idxs).issubset(
                detections.index
            ), "Mismatch of indexes during the tracking. The results should match the detections."
            results_df = pd.DataFrame(
                {
                    "track_bbox_ltwh": track_bbox_ltwh,
                    "track_bbox_conf": track_bbox_conf,
                    "track_id": track_ids,
                    "idxs": idxs,
                }
            )
            results_df.set_index("idxs", inplace=True, drop=True)
            return results_df
        else:
            return []
