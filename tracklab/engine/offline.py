"""Offline tracking engine for TrackLab."""

import logging
from typing import Any, Dict, Tuple
from tqdm import tqdm
import pandas as pd

from tracklab.datastruct import TrackerState
from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class OfflineTrackingEngine(TrackingEngine):
    """Offline tracking engine that processes all data at once.

    This engine loads all detections and image data upfront, then processes
    them through the tracking pipeline in batch mode.
    """

    def video_loop(
        self, tracker_state: TrackerState, video: Any, video_id: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process a video using offline tracking approach.

        Args:
            tracker_state: Current state of the tracker with detections and metadata.
            video: Video metadata or identifier.
            video_id: Unique identifier for the video.

        Returns:
            Tuple of (detections DataFrame, image predictions DataFrame).
        """
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()  # type: ignore

        load_result = tracker_state.load()
        if isinstance(load_result, pd.DataFrame):
            # JSON file case - only detections
            detections = load_result
            image_pred = tracker_state.image_metadatas[
                tracker_state.image_metadatas.video_id == video_id
            ]
        else:
            detections, image_pred = load_result
        if len(self.module_names) == 0:
            return detections, image_pred

        image_filepaths: Dict[Any, str] = {
            idx: fn for idx, fn in image_pred["file_path"].items()
        }
        model_names = self.module_names
        log.info(f"🎯 Processing {len(model_names)} modules for video {video_id}")

        for model_name in tqdm(
            model_names,
            desc=f"Processing modules for video {video_id}",
            unit="module",
            leave=False,
        ):
            log.info(f"🔄 Processing module: {model_name}")
            if self.models[model_name].level == "video":
                detections = self.models[model_name].process(  # type: ignore
                    detections, image_pred
                )
                continue
            self.datapipes[model_name].update(image_filepaths, image_pred, detections)
            self.callback(
                "on_module_start",
                task=model_name,
                dataloader=self.dataloaders[model_name],
            )

            # Get total number of batches for progress bar
            total_batches = len(self.dataloaders[model_name])
            for batch_idx, batch in enumerate(self.dataloaders[model_name]):
                detections, image_pred = self.default_step(
                    batch, model_name, detections, image_pred
                )

            self.callback("on_module_end", task=model_name, detections=detections)
            if detections.empty:
                return detections, image_pred
        return detections, image_pred
