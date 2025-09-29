import logging
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracklab.callbacks import Callback

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class Progressbar(Callback):
    """Base class for progress bar callbacks.

    Provides a factory method to create either TQDM or Rich progress bars
    based on configuration.
    """

    def __new__(cls, use_rich: bool = False, dummy: bool = False) -> "Progressbar":
        """Create a progress bar instance.

        Args:
            use_rich: Whether to use Rich progress bars.
            dummy: Whether to create a dummy progress bar.

        Returns:
            Progress bar instance.
        """
        if dummy:
            return super().__new__(cls)
        elif not use_rich:
            return TQDMProgressbar()
        else:
            return RichProgressbar()

    def init_progress_bar(self, task: str, desc: str, length: int):
        """Initialize a progress bar for a task.

        Args:
            task: Task name.
            desc: Description for the progress bar.
            length: Total number of steps.
        """
        pass


class TQDMProgressbar(Progressbar):
    """TQDM-based progress bar callback."""

    def __init__(self, **kwargs):
        """Initialize TQDM progress bar."""
        self.pbar: Optional[tqdm] = None
        self.task_pbars: Dict[str, tqdm] = {}
        self.video_id: Optional[int] = None

    def on_dataset_track_start(self, engine: "TrackingEngine"):
        """Initialize main progress bar when dataset tracking starts.

        Args:
            engine: The tracking engine instance.
        """
        total = len(engine.video_metadatas)
        log.info(
            f"Inference will be composed of the following steps: {', '.join(x for x in engine.module_names)}"
        )
        self.pbar = tqdm(total=total, desc="Tracking videos")

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        """Close main progress bar when dataset tracking ends.

        Args:
            engine: The tracking engine instance.
        """
        if self.pbar is not None:
            self.pbar.close()

    def on_video_loop_start(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        index: int,
    ):
        """Update progress bar description when video processing starts.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            index: Index in the video sequence.
        """
        self.video_id = video_idx
        if self.pbar is not None:
            self.pbar.set_description(f"Tracking videos ({video_metadata['name']})")

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        """Update main progress bar when video processing ends.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            detections: Detection results.
            image_pred: Image predictions.
        """
        if self.pbar is not None:
            self.pbar.update()
            self.pbar.refresh()

    def on_module_start(
        self, engine: "TrackingEngine", task: str, dataloader: DataLoader
    ):
        """Initialize task-specific progress bar when module starts.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            dataloader: Data loader for the task.
        """
        desc = task.replace("_", " ").capitalize()
        if hasattr(engine.models[task], "process_video"):
            length = len(
                engine.img_metadatas[engine.img_metadatas.video_id == self.video_id]
            )
        else:
            length = len(dataloader)
        self.init_progress_bar(task, desc, length)

    def init_progress_bar(self, task: str, desc: str, length: int):
        """Initialize TQDM progress bar for a task.

        Args:
            task: Task name.
            desc: Description for the progress bar.
            length: Total number of steps.
        """
        self.task_pbars[task] = tqdm(total=length, desc=desc, leave=False, position=1)

    def on_module_step_end(
        self,
        engine: "TrackingEngine",
        task: str,
        batch: Any,
        detections: pd.DataFrame,
    ):
        """Update task progress bar after each step.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            batch: Current batch data.
            detections: Detection results.
        """
        self.task_pbars[task].update()

    def on_module_end(
        self, engine: "TrackingEngine", task: str, detections: pd.DataFrame
    ):
        """Close task progress bar when module ends.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            detections: Final detection results.
        """
        self.task_pbars[task].close()


class RichProgressbar(Progressbar):
    """Rich-based progress bar callback."""

    def __init__(self, **kwargs):
        """Initialize Rich progress bar."""
        self.pbar: Optional[Progress] = None
        self.tasks: Dict[str, TaskID] = {}
        self.video_id: Optional[int] = None

    def on_dataset_track_start(self, engine: "TrackingEngine"):
        """Initialize main Rich progress bar when dataset tracking starts.

        Args:
            engine: The tracking engine instance.
        """
        total = len(engine.video_metadatas)
        self.pbar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            speed_estimate_period=600,  # estimate speed over ten minutes
        )
        self.pbar.start()
        self.tasks["main"] = self.pbar.add_task("[yellow]Tracking videos", total=total)

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        """Stop main Rich progress bar when dataset tracking ends.

        Args:
            engine: The tracking engine instance.
        """
        if self.pbar is not None:
            self.pbar.stop()

    def on_video_loop_start(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        index: int,
    ):
        """Update progress bar description when video processing starts.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            index: Index in the video sequence.
        """
        self.video_id = video_idx
        if self.pbar is not None:
            self.pbar.update(
                self.tasks["main"],
                description=f"[yellow]Tracking videos ({video_metadata['name']})",
            )

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        """Update main progress bar when video processing ends.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the current video.
            video_idx: Index of the current video.
            detections: Detection results.
            image_pred: Image predictions.
        """
        if self.pbar is not None:
            self.pbar.update(self.tasks["main"], advance=1, refresh=True)

    def on_module_start(
        self, engine: "TrackingEngine", task: str, dataloader: DataLoader
    ):
        """Initialize task-specific progress bar when module starts.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            dataloader: Data loader for the task.
        """
        desc = task
        if hasattr(engine.models[task], "process_video"):
            length = len(
                engine.img_metadatas[engine.img_metadatas.video_id == self.video_id]
            )
        else:
            length = len(dataloader)
        self.init_progress_bar(task, desc, length)

    def init_progress_bar(self, task: str, desc: str, length: int):
        """Initialize Rich progress bar for a task.

        Args:
            task: Task name.
            desc: Description for the progress bar.
            length: Total number of steps.
        """
        if self.pbar is not None:
            self.tasks[task] = self.pbar.add_task(desc, total=length)

    def on_module_step_end(
        self,
        engine: "TrackingEngine",
        task: str,
        batch: Any,
        detections: pd.DataFrame,
    ):
        """Update task progress bar after each step.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            batch: Current batch data.
            detections: Detection results.
        """
        if self.pbar is not None:
            self.pbar.update(self.tasks[task], advance=1)

    def on_module_end(
        self, engine: "TrackingEngine", task: str, detections: pd.DataFrame
    ):
        """Remove task progress bar when module ends.

        Args:
            engine: The tracking engine instance.
            task: Name of the task/module.
            detections: Final detection results.
        """
        if self.pbar is not None:
            self.pbar.stop_task(self.tasks[task])
            self.pbar.remove_task(self.tasks[task])
