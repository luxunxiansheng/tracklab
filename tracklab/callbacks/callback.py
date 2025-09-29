"""Base callback class for TrackLab tracking engine events."""

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine


class Callback:
    """Base callback class for tracking engine events.

    Callbacks can be used to hook into various points in the tracking pipeline
    to perform custom operations like logging, visualization, or evaluation.
    Subclasses should override specific methods to implement custom behavior.
    """

    after_saved_state: bool = False

    def on_dataset_track_start(self, engine: "TrackingEngine") -> None:
        """Called when dataset tracking starts.

        Args:
            engine: The tracking engine instance.
        """
        pass

    def on_dataset_track_end(self, engine: "TrackingEngine") -> None:
        """Called when dataset tracking ends.

        Args:
            engine: The tracking engine instance.
        """
        pass

    def on_video_loop_start(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,  # FIXME keep ?
        # image_metadatas: pd.DataFrame,  # FIXME add ?
        video_idx: int,
        index: int,  # FIXME change name ?
    ) -> None:
        """Called when video loop starts.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            index: Additional index parameter.
        """
        pass

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,  # FIXME keep ?
        # image_metadatas: pd.DataFrame,  # FIXME add ?
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ) -> None:
        """Called when video loop ends.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            detections: Detection results for the video.
            image_pred: Image prediction results.
        """
        pass

    def on_image_loop_start(
        self,
        engine: "TrackingEngine",
        image_metadata: pd.Series,
        image_idx: int,
        index: int,
    ) -> None:
        """Called when image loop starts.

        Args:
            engine: The tracking engine instance.
            image_metadata: Metadata for the current image.
            image_idx: Index of the current image.
            index: Additional index parameter.
        """
        pass

    def on_image_loop_end(
        self,
        engine: "TrackingEngine",
        image_metadata: pd.Series,
        image: Any,
        image_idx: int,
        detections: pd.DataFrame,
    ) -> None:
        """Called when image loop ends.

        Args:
            engine: The tracking engine instance.
            image_metadata: Metadata for the current image.
            image: The processed image data.
            image_idx: Index of the current image.
            detections: Detection results for the image.
        """
        pass

    def on_module_start(
        self, engine: "TrackingEngine", task: str, dataloader: DataLoader
    ) -> None:
        """Called when module processing starts.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module being processed.
            dataloader: DataLoader for the module.
        """
        pass

    def on_module_end(
        self, engine: "TrackingEngine", task: str, detections: pd.DataFrame
    ) -> None:
        """Called when module processing ends.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module being processed.
            detections: Detection results from the module.
        """
        pass

    def on_module_step_start(
        self, engine: "TrackingEngine", task: str, batch: Any
    ) -> None:
        """Called when module step starts.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module being processed.
            batch: Current batch being processed.
        """
        pass

    def on_module_step_end(
        self, engine: "TrackingEngine", task: str, batch: Any, detections: pd.DataFrame
    ) -> None:
        """Called when module step ends.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module being processed.
            batch: Current batch that was processed.
            detections: Detection results from the batch.
        """
        pass
