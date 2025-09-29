import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from tracklab.callbacks.callback import Callback

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class ExporterCallback(Callback):
    """Callback that exports tracking results using a specified exporter.

    Can export tracking results after each video or at the end of the entire dataset,
    depending on the export_per_video configuration.
    """

    after_saved_state = True

    def __init__(
        self,
        exporter: Any,
        save_path: str = "exports",
        export_per_video: bool = False,
        bbox_column: str = "bbox_ltwh",
        save_classes: bool = False,
        is_ground_truth: bool = False,
        **kwargs: Any,
    ):
        """Initialize the exporter callback.

        Args:
            exporter: The exporter instance to use for exporting results.
            save_path: Base path where to save exported data.
            export_per_video: If True, export after each video. If False, export at dataset end.
            bbox_column: Column name containing bounding box data.
            save_classes: Whether to include class information in exports.
            is_ground_truth: Whether this is ground truth data.
            **kwargs: Additional parameters passed to the exporter.
        """
        self.exporter = exporter
        self.save_path = Path(save_path)
        self.export_per_video = export_per_video
        self.bbox_column = bbox_column
        self.save_classes = save_classes
        self.is_ground_truth = is_ground_truth
        self.export_kwargs = kwargs

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        """Export tracking results after each video if export_per_video is True.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            detections: Detection results for the current video.
            image_pred: Image predictions for the current video.
        """
        if self.export_per_video and engine.tracker_state.detections_pred is not None:
            # Get image IDs for this video
            video_image_ids = engine.tracker_state.image_metadatas[
                engine.tracker_state.image_metadatas["video_id"] == video_idx
            ].index

            # Filter detections for this video
            video_detections = engine.tracker_state.detections_pred[
                engine.tracker_state.detections_pred["image_id"].isin(video_image_ids)
            ]

            # Filter image metadata for this video
            video_image_metadatas = engine.tracker_state.image_metadatas.loc[
                video_image_ids
            ]

            # Create video metadata for this video
            video_video_metadatas = engine.tracker_state.video_metadatas.loc[
                [video_idx]
            ]

            # Create video-specific save path
            video_save_path = self.save_path / video_metadata["name"]

            try:
                self.exporter.export(
                    detections=video_detections,
                    image_metadatas=video_image_metadatas,
                    video_metadatas=video_video_metadatas,
                    save_path=str(video_save_path),
                    bbox_column=self.bbox_column,
                    save_classes=self.save_classes,
                    is_ground_truth=self.is_ground_truth,
                    **self.export_kwargs,
                )
                log.info(
                    f"Exported tracking results for video '{video_metadata['name']}' to {video_save_path}"
                )
            except Exception as e:
                log.error(
                    f"Failed to export results for video '{video_metadata['name']}': {e}"
                )

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        """Export tracking results for the entire dataset if export_per_video is False.

        Args:
            engine: The tracking engine instance.
        """
        if (
            not self.export_per_video
            and engine.tracker_state.detections_pred is not None
        ):
            try:
                self.exporter.export(
                    detections=engine.tracker_state.detections_pred,
                    image_metadatas=engine.tracker_state.image_metadatas,
                    video_metadatas=engine.tracker_state.video_metadatas,
                    save_path=str(self.save_path),
                    bbox_column=self.bbox_column,
                    save_classes=self.save_classes,
                    is_ground_truth=self.is_ground_truth,
                    **self.export_kwargs,
                )
                log.info(
                    f"Exported tracking results for entire dataset to {self.save_path}"
                )
            except Exception as e:
                log.error(f"Failed to export dataset results: {e}")
