from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
from tracklab.utils.cv2 import cv2_load_image, crop_bbox_ltwh
from tracklab.utils.attribute_voting import select_highest_voted_att

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.openmmlab import get_checkpoint

from collections import Counter


import logging


log = logging.getLogger(__name__)


class MajorityVoteTracklet(VideoLevelModule):

    input_columns = []
    output_columns = []

    def __init__(self, cfg, device, tracking_dataset=None) -> None:
        self.attributes = cfg.attributes
        for attribute in self.attributes:
            self.output_columns.append(attribute)

    @torch.no_grad()
    def process(
        self, detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> pd.DataFrame:

        detections[self.output_columns] = np.nan

        # First, handle detections WITH track_ids (tracklets)
        if "track_id" in detections.columns:
            for track_id in detections.track_id.unique():
                if pd.isna(track_id):
                    continue  # Skip NaN track_ids for now, handle them separately
                tracklet = detections[detections.track_id == track_id]
                for attribute in self.attributes:
                    det_col = f"{attribute}_detection"
                    conf_col = f"{attribute}_confidence"
                    if (
                        det_col not in detections.columns
                        or conf_col not in detections.columns
                    ):
                        continue
                    attribute_detection = tracklet[det_col]
                    attribute_confidence = tracklet[conf_col]
                    attribute_value = [
                        select_highest_voted_att(
                            attribute_detection, attribute_confidence
                        )
                    ] * len(tracklet)
                    detections.loc[tracklet.index, attribute] = attribute_value

        # Second, handle detections WITHOUT track_ids (e.g., untracked balls)
        # For these, use the detection-level attribute directly (no voting needed)
        for attribute in self.attributes:
            det_col = f"{attribute}_detection"
            if det_col in detections.columns:
                # Find detections without track_id or with NaN track_id
                if "track_id" in detections.columns:
                    no_track = detections["track_id"].isna()
                else:
                    no_track = pd.Series(
                        [True] * len(detections), index=detections.index
                    )

                # For detections without track_id, copy the detection value directly to the output attribute
                detections.loc[no_track & detections[det_col].notna(), attribute] = (
                    detections.loc[no_track & detections[det_col].notna(), det_col]
                )

        return detections
