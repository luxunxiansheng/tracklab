# from pytorch_lightning import Callback as PLCallback
from typing import TYPE_CHECKING, Any

import pandas as pd
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine


class Callback:
    """Base callback class for tracking engine events.

    Callbacks can be used to hook into various points in the tracking pipeline
    to perform custom operations like logging, visualization, or evaluation.
    """

    after_saved_state = False

    def on_dataset_track_start(self, engine: "TrackingEngine"):
        """Called when dataset tracking starts."""
        pass

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        """Called when dataset tracking ends."""
        pass

    def on_video_loop_start(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,  # FIXME keep ?
        # image_metadatas: pd.DataFrame,  # FIXME add ?
        video_idx: int,
        index: int,  # FIXME change name ?
    ):
        """Called when video loop starts."""
        pass

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,  # FIXME keep ?
        # image_metadatas: pd.DataFrame,  # FIXME add ?
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        """Called when video loop ends."""
        pass

    def on_image_loop_start(
        self,
        engine: "TrackingEngine",
        image_metadata: pd.Series,
        image_idx: int,
        index: int,
    ):
        """Called when image loop starts."""
        pass

    def on_image_loop_end(
        self,
        engine: "TrackingEngine",
        image_metadata: pd.Series,
        image,
        image_idx: int,
        detections: pd.DataFrame,
    ):
        """Called when image loop ends."""
        pass

    def on_module_start(
        self, engine: "TrackingEngine", task: str, dataloader: DataLoader
    ):
        """Called when module processing starts."""
        pass

    def on_module_end(
        self, engine: "TrackingEngine", task: str, detections: pd.DataFrame
    ):
        """Called when module processing ends."""
        pass

    def on_module_step_start(self, engine: "TrackingEngine", task: str, batch: Any):
        """Called when module step starts."""
        pass

    def on_module_step_end(
        self, engine: "TrackingEngine", task: str, batch: Any, detections: pd.DataFrame
    ):
        """Called when module step ends."""
        pass
