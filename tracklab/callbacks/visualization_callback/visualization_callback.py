from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING, Any, List, Tuple, Union
import logging

import cv2
import imageio
import pandas as pd
from pandas import DataFrame, Series

from tracklab.callbacks import Progressbar, Callback
from .visualizer import Visualizer
from tracklab.datastruct import TrackerState
from tracklab.utils.cv2 import final_patch, cv2_load_image

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Visualization callback for creating visual outputs from tracking results.

    This callback processes tracking results and generates various visual outputs
    including images, videos, and GIFs using a collection of visualizer modules.

    Args:
        visualizers: Dictionary of visualizer instances for drawing on frames.
        save_images: Whether to save individual frame images.
        save_videos: Whether to save video files.
        save_gifs: Whether to save animated GIFs.
        video_fps: Frame rate for video and GIF outputs.
        process_n_videos: Maximum number of videos to process.
        process_n_frames_by_video: Maximum frames per video to process.
        save_dir: Directory to save visualization outputs.
    """

    def __init__(
        self,
        visualizers: Dict[str, Visualizer],
        save_images: bool = False,
        save_videos: bool = False,
        save_gifs: bool = False,
        video_fps: int = 25,
        process_n_videos: Optional[int] = None,
        process_n_frames_by_video: Optional[int] = None,
        save_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the visualization callback.

        Args:
            visualizers: Dictionary mapping names to visualizer instances.
            save_images: Whether to save individual frames as images.
            save_videos: Whether to create MP4 video files.
            save_gifs: Whether to create animated GIF files.
            video_fps: Frames per second for video/GIF output.
            process_n_videos: Limit on number of videos to process.
            process_n_frames_by_video: Limit on frames per video.
            save_dir: Output directory for visualizations.
            **kwargs: Additional arguments passed to visualizers.
        """
        self.visualizers = visualizers
        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            # Use current working directory + visualization
            self.save_dir = Path.cwd() / "visualization"

        # Ensure save_dir is absolute
        if not self.save_dir.is_absolute():
            self.save_dir = Path.cwd() / self.save_dir

        self.save_images = save_images
        self.save_videos = save_videos
        self.save_gifs = save_gifs
        self.video_fps = video_fps
        self.max_videos = process_n_videos
        self.max_frames = process_n_frames_by_video
        for visualizer in visualizers.values():
            visualizer.post_init(**kwargs)

    def on_dataset_track_end(self, engine: "TrackingEngine") -> None:
        """Log visualization output location when dataset tracking completes.

        Args:
            engine: The tracking engine instance.
        """
        if self.save_videos or self.save_images or self.save_gifs:
            log.info(f"Visualization output at : {self.save_dir.absolute()}")

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ) -> None:
        """Process visualization for completed video.

        Args:
            engine: The tracking engine instance.
            video_metadata: Metadata for the completed video.
            video_idx: Index of the completed video.
            detections: Detection results for the video.
            image_pred: Image predictions for the video.
        """
        if self.save_videos or self.save_images or self.save_gifs:
            progress = engine.callbacks.get("progress", Progressbar(dummy=True))
            self.visualize(
                engine.tracker_state, video_idx, detections, image_pred, progress
            )
            if hasattr(progress, "on_module_end"):
                progress.on_module_end(engine, "vis", detections)

    """ 
    #TODO implement the online visualization
    previous code:
        if self.cfg.show_online:
        tracker_state = engine.tracker_state
        if tracker_state.detections_gt is not None:
            ground_truths = tracker_state.detections_gt[
                tracker_state.detections_gt.image_id == image_metadata.name
            ]
        else:
            ground_truths = None
        if len(detections) == 0:
            image = image
        else:
            detections = detections[detections.image_id == image_metadata.name]
            image = self.draw_frame(image_metadata,
                                    detections, ground_truths, "inf", image=image)
        if platform.system() == "Linux" and self.video_name not in self.windows:
            self.windows.append(self.video_name)
            cv2.namedWindow(str(self.video_name),
                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(self.video_name), image.shape[1], image.shape[0])
        cv2.imshow(str(self.video_name), image)
        cv2.waitKey(1)
    """

    def visualize(
        self,
        tracker_state: TrackerState,
        video_id: int,
        detections: pd.DataFrame,
        image_preds: pd.DataFrame,
        progress: Optional[Any] = None,
    ) -> None:
        """Generate visualizations for a video.

        Args:
            tracker_state: Current tracker state with all data.
            video_id: ID of the video to visualize.
            detections: Detection results DataFrame.
            image_preds: Image predictions DataFrame.
            progress: Progress bar callback for updates.
        """
        image_metadatas = tracker_state.image_metadatas[
            tracker_state.image_metadatas.video_id == video_id
        ]
        image_gts = tracker_state.image_gt[tracker_state.image_gt.video_id == video_id]
        nframes = len(image_metadatas)
        video_name = tracker_state.video_metadatas.loc[video_id]["name"]
        for visualizer in self.visualizers.values():
            try:
                visualizer.preproces(
                    detections,
                    tracker_state.detections_gt,
                    image_preds,
                    tracker_state.image_gt,
                )
            except Exception as e:
                log.warning(
                    f"Visualizer {Visualizer} raised error : {e} during preprocess."
                )
        total = self.max_frames or len(image_metadatas.index)
        if progress is not None and hasattr(progress, "init_progress_bar"):
            progress.init_progress_bar("vis", "Visualization", total)

        # Handle empty or malformed detections DataFrame
        if detections.empty or "image_id" not in detections.columns:
            log.debug(
                f"Detections DataFrame is empty or missing 'image_id' column. Available columns: {list(detections.columns)}"
            )
            # Create a dummy DataFrame with the expected column to allow groupby to work
            detections_for_groupby = pd.DataFrame(columns=["image_id"])
        else:
            detections_for_groupby = detections
        detection_preds_by_image = detections_for_groupby.groupby("image_id")

        # Handle empty or malformed ground truth detections DataFrame
        if (
            tracker_state.detections_gt.empty
            or "image_id" not in tracker_state.detections_gt.columns
        ):
            log.debug(
                f"Ground truth detections DataFrame is empty or missing 'image_id' column. Available columns: {list(tracker_state.detections_gt.columns)}"
            )
            detections_gt_for_groupby = pd.DataFrame(columns=["image_id"])
        else:
            detections_gt_for_groupby = tracker_state.detections_gt
        detection_gts_by_image = detections_gt_for_groupby.groupby("image_id")
        args = [
            create_draw_args(
                image_id,
                self,
                image_metadatas,
                get_group(detection_preds_by_image, image_id),
                get_group(detection_gts_by_image, image_id),
                image_gts,
                image_preds,
                nframes,
            )
            for image_id in islice(image_metadatas.index, 0, None, nframes // total)
        ]
        if self.save_videos:
            image = cv2_load_image(image_metadatas.iloc[0].file_path)
            filepath = self.save_dir / "videos" / f"{video_name}.mp4"
            try:
                # Ensure the full path exists
                filepath.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, FileNotFoundError) as e:
                log.error(f"Failed to create directory {filepath.parent}: {e}")
                log.error(f"Current working directory: {Path.cwd()}")
                log.error(f"Save directory: {self.save_dir}")
                log.error(f"Full filepath: {filepath}")
                raise

            video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter.fourcc(*"mp4v"),
                float(self.video_fps),
                (image.shape[1], image.shape[0]),
            )

        if self.save_gifs:
            gif_frames = []

        with Pool() as p:
            log.info(f"Starting visualization saving for video '{video_name}'")
            counter = 0
            for output_image, file_name in p.imap(process_frame, args):
                if self.save_images:
                    filepath = self.save_dir / "images" / str(video_name) / file_name
                    try:
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                    except (OSError, FileNotFoundError) as e:
                        log.error(f"Failed to create directory {filepath.parent}: {e}")
                        log.error(f"Current working directory: {Path.cwd()}")
                        log.error(f"Save directory: {self.save_dir}")
                        log.error(f"Full filepath: {filepath}")
                        raise
                    assert cv2.imwrite(str(filepath), output_image)
                if self.save_videos:
                    video_writer.write(output_image)
                if self.save_gifs:
                    gif_frames.append(output_image)
                counter += 1
                if counter % 10 == 0 or counter == total:
                    log.info(f"Saved {counter}/{total} frames for video '{video_name}'")
                if progress is not None and hasattr(progress, "on_module_step_end"):
                    progress.on_module_step_end(None, "vis", None, detections)

        if self.save_gifs:
            gif_filepath = self.save_dir / "gifs" / f"{video_name}.gif"
            gif_filepath.parent.mkdir(parents=True, exist_ok=True)
            # Convert BGR to RGB for imageio
            rgb_frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in gif_frames
            ]
            log.info(f"Saving GIF to {gif_filepath}")
            imageio.mimsave(str(gif_filepath), rgb_frames, fps=self.video_fps)
            log.info(f"Saved GIF for video '{video_name}' at {gif_filepath}")

    def draw_frame(
        self,
        image_metadata: pd.Series,
        detections_pred: pd.DataFrame,
        detections_gt: pd.DataFrame,
        image_pred: pd.DataFrame,
        image_gt: pd.DataFrame,
        nframes: int,
    ) -> Any:
        """Draw visualizations on a single frame.

        Args:
            image_metadata: Metadata for the current image.
            detections_pred: Predicted detections for the frame.
            detections_gt: Ground truth detections for the frame.
            image_pred: Predicted image data.
            image_gt: Ground truth image data.
            nframes: Total number of frames in the video.

        Returns:
            The processed image with visualizations drawn.
        """
        image = cv2_load_image(image_metadata.file_path)
        for visualizer in self.visualizers.values():
            try:
                visualizer.draw_frame(
                    image, detections_pred, detections_gt, image_pred, image_gt
                )
            except Exception as e:
                log.warning(
                    f"Visualizer {type(visualizer).__name__} raised error : {e} during drawing."
                )
        return final_patch(image)


def create_draw_args(
    image_id: int,
    instance: "VisualizationCallback",
    image_metadatas: DataFrame,
    detections_pred: DataFrame,
    detections_gt: DataFrame,
    image_gts: DataFrame,
    image_preds: DataFrame,
    nframes: int,
) -> Tuple[Any, ...]:
    """Create arguments for frame drawing.

    Args:
        image_id: ID of the image to process.
        instance: Visualization callback instance.
        image_metadatas: DataFrame with image metadata.
        detections_pred: Predicted detections.
        detections_gt: Ground truth detections.
        image_gts: Ground truth image data.
        image_preds: Predicted image data.
        nframes: Total number of frames.

    Returns:
        Tuple of arguments for process_frame function.
    """
    image_metadata = image_metadatas.loc[image_id]
    image_gt = image_gts.loc[image_id]
    image_pred = image_preds.loc[image_id]
    return (
        instance,
        image_metadata,
        detections_pred,
        detections_gt,
        image_pred,
        image_gt,
        nframes,
    )


def process_frame(args: Tuple) -> Tuple[Any, str]:
    """Process a single frame for visualization.

    Args:
        args: Tuple of arguments from create_draw_args.

    Returns:
        Tuple of (processed_image, filename).
    """
    (
        instance,
        image_metadata,
        detections_pred,
        detections_gt,
        image_pred,
        image_gt,
        nframes,
    ) = args
    frame = instance.draw_frame(
        image_metadata, detections_pred, detections_gt, image_pred, image_gt, nframes
    )

    return frame, Path(image_metadata.file_path).name


def get_group(g: Any, key: int) -> DataFrame:
    """Get a group from a pandas GroupBy object.

    Args:
        g: GroupBy object.
        key: Key to retrieve.

    Returns:
        DataFrame for the group, or empty DataFrame if key not found.
    """
    if key in g.groups:
        return g.get_group(key)
    return pd.DataFrame(columns=["bbox_ltwh"])
