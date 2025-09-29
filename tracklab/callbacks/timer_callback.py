import logging
import time
from datetime import timedelta
from typing import Dict, TYPE_CHECKING

import pandas as pd
from torch.utils.data import DataLoader

from tracklab.callbacks import Callback

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class Timer(Callback):
    """Callback for timing various stages of the tracking pipeline.

    Measures and logs execution times for dataset processing, individual videos,
    and pipeline modules, including frames per second calculations.
    """

    def __init__(self, **kwargs):
        """Initialize the timer callback."""
        self.start_times: Dict[str, float] = {}

    def on_dataset_track_start(self, engine: "TrackingEngine"):
        """Start timing when dataset tracking begins.

        Args:
            engine: The tracking engine instance.
        """
        self.start_times["dataset"] = time.perf_counter()

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        """Log total dataset processing time when tracking ends.

        Args:
            engine: The tracking engine instance.
        """
        time_delta = timedelta(
            seconds=time.perf_counter() - self.start_times["dataset"]
        )
        log.info(f"Dataset time : {time_delta}")

    def on_video_loop_start(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        index: int,
    ):
        """Start timing when video processing begins.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            index: Index in the video sequence.
        """
        self.start_times["video"] = time.perf_counter()

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        """Log video processing time and FPS when video ends.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            detections: Detection results.
            image_pred: Image predictions.
        """
        end = time.perf_counter()
        time_delta = timedelta(seconds=end - self.start_times["video"])
        frames = len(image_pred)
        fps = frames / (end - self.start_times["video"])
        log.info(f"Video time : {time_delta}, FPS : {fps}")

    def on_module_start(
        self, engine: "TrackingEngine", task: str, dataloader: DataLoader
    ):
        """Start timing when module processing begins.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            dataloader: Data loader for the task.
        """
        self.start_times[task] = time.perf_counter()

    def on_module_end(
        self, engine: "TrackingEngine", task: str, detections: pd.DataFrame
    ):
        """Log module processing time and FPS when module ends.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            detections: Detection results from the module.
        """
        end = time.perf_counter()
        time_delta = timedelta(seconds=end - self.start_times[task])
        frames = len(detections.image_id.unique())
        fps = frames / (end - self.start_times[task])
        log.info(f"Module {task} time : {time_delta}, FPS : {fps}")
