"""TrackEval-based evaluator for TrackLab tracking evaluation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import trackeval
from tabulate import tabulate
from tracklab.callbacks.evaluation_callback.evaluator import Evaluator as EvaluatorBase
from hydra.utils import instantiate

log = logging.getLogger(__name__)


class TrackEvalEvaluator(EvaluatorBase):
    """Evaluator using the TrackEval library for comprehensive tracking evaluation.

    This evaluator uses the TrackEval library (https://github.com/JonathonLuiten/TrackEval)
    to perform standardized evaluation of tracking predictions. It works with exported
    tracking data and provides comprehensive metrics for tracking performance assessment.
    """

    def __init__(
        self,
        tracking_dataset: Optional[Any] = None,
        export_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the TrackEval evaluator.

        Args:
            tracking_dataset: The tracking dataset instance.
            export_path: Path where tracking results were exported.
            *args: Additional positional arguments.
            **kwargs: Configuration parameters from Hydra.
        """
        # Handle Hydra instantiate pattern where all config comes via kwargs
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.create(kwargs)
        self.export_path = export_path

        self.show_progressbar: bool = self.cfg.get("show_progressbar", True)
        self.eval_set: str = self.cfg.get("eval_set", "val")
        self.dataset_path: Optional[str] = self.cfg.get("dataset_path", None)
        if self.dataset_path is None:
            raise ValueError("dataset_path must be specified in the config")
        self.tracking_dataset = tracking_dataset
        self.trackeval_dataset_name: str = self.cfg.dataset.dataset_class
        self.trackeval_dataset_class = getattr(
            trackeval.datasets, self.trackeval_dataset_name
        )

    def run(self, tracker_state: Any) -> None:
        log.info(
            "Starting evaluation using TrackEval library (https://github.com/JonathonLuiten/TrackEval)"
        )

        tracker_name = "tracklab"
        # save_classes = self.trackeval_dataset_class.__name__ != 'MotChallenge2DBox'

        # Get the path where predictions were exported by the export_callback
        if self.export_path is None:
            log.error(
                "No export_path provided. TrackEvalEvaluator requires exported predictions."
            )
            return

        # Use the export path as the TRACKERS_FOLDER
        export_save_path = Path(self.export_path)

        # Make path absolute if it's relative
        if not export_save_path.is_absolute():
            export_save_path = Path.cwd() / export_save_path

        if not export_save_path.exists():
            # Try to create the directory in case export callback hasn't run yet
            try:
                export_save_path.mkdir(parents=True, exist_ok=True)
                log.info(f"Created export directory: {export_save_path}")
            except Exception as e:
                log.error(
                    f"Export path {export_save_path} does not exist and could not be created: {e}. Make sure export_callback runs before evaluation."
                )
                return

        dataset_config = self.trackeval_dataset_class.get_default_dataset_config()
        for key, value in self.cfg.dataset.items():
            dataset_config[key] = value

        # Set up tracker folder configuration
        dataset_config["TRACKERS_FOLDER"] = str(export_save_path)
        dataset_config["TRACKER_SUB_FOLDER"] = ""
        dataset_config["OUTPUT_FOLDER"] = str(export_save_path / "results")

        if tracker_state.detections_gt is None or len(tracker_state.detections_gt) == 0:
            log.warning(
                f"Stopping evaluation because the current split ({self.eval_set}) has no ground truth detections."
            )
            return

        # Build TrackEval dataset
        dataset_config["SEQ_INFO"] = tracker_state.video_metadatas.set_index("name")[
            "nframes"
        ].to_dict()
        dataset_config["BENCHMARK"] = (
            self.trackeval_dataset_name
        )  # required for trackeval.datasets.MotChallenge2DBox

        dataset_config["GT_FOLDER"] = self.dataset_path  # Location of GT data
        dataset_config["GT_LOC_FORMAT"] = (
            "{gt_folder}/{seq}/Labels-GameState.json"  # '{gt_folder}/{seq}/gt/gt.txt'
        )

        # Create a single tracker list pointing to the exported data
        dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]

        # Create the expected TrackEval directory structure
        # TrackEval expects: TRACKERS_FOLDER/{benchmark_name}-{split}/{tracker_name}/files
        # But our export saves directly to export_path/files
        # So we need to create the expected directory structure and copy/symlink files
        benchmark_split_name = f"{dataset_config['BENCHMARK']}-{self.eval_set}"
        trackeval_tracker_path = export_save_path / benchmark_split_name / tracker_name
        trackeval_tracker_path.mkdir(parents=True, exist_ok=True)

        # Create symlinks for all JSON files from export_path to the expected tracker path
        json_files = list(export_save_path.glob("*.json"))
        log.info(
            f"Found {len(json_files)} JSON files in {export_save_path}: {[f.name for f in json_files]}"
        )

        for json_file in json_files:
            symlink_target = trackeval_tracker_path / json_file.name
            if not symlink_target.exists():
                try:
                    symlink_target.symlink_to(json_file.resolve())
                    log.info(f"Created symlink: {symlink_target} -> {json_file}")
                except Exception as e:
                    log.warning(f"Could not create symlink for {json_file}: {e}")
                    # If symlink fails, try copying the file
                    try:
                        import shutil

                        shutil.copy2(json_file, symlink_target)
                        log.info(f"Copied file: {json_file} -> {symlink_target}")
                    except Exception as copy_e:
                        log.error(f"Could not copy file {json_file}: {copy_e}")

        if len(json_files) == 0:
            log.warning(
                f"No JSON files found in {export_save_path}. Export callback may not have run yet."
            )

        dataset = self.trackeval_dataset_class(dataset_config)

        # Build metrics
        metrics_config = {
            "METRICS": set(self.cfg.metrics),
            "PRINT_CONFIG": False,
            "THRESHOLD": 0.5,
        }
        metrics_list = []
        for metric_name in self.cfg.metrics:
            try:
                metric = getattr(trackeval.metrics, metric_name)
                metrics_list.append(metric(metrics_config))
            except AttributeError:
                log.warning(f"Skipping evaluation for unknown metric: {metric_name}")

        # Build evaluator
        eval_config = trackeval.Evaluator.get_default_eval_config()
        for key, value in self.cfg.eval.items():
            if key == "NUM_PARALLEL_CORES":
                value = max(1, int(value))
            eval_config[key] = value
        evaluator = trackeval.Evaluator(eval_config)

        # Run evaluation
        output_res, output_msg = evaluator.evaluate(
            [dataset], metrics_list, show_progressbar=self.show_progressbar
        )
        log.info(output_msg)

        # Log results
        results = output_res[dataset.get_name()][tracker_name]
        if results is None:
            log.error(
                f"Evaluation failed for tracker '{tracker_name}' on dataset '{dataset.get_name()}'"
            )
            log.error(
                "This is likely due to a multiprocessing/pickling issue. Try setting USE_PARALLEL: False in your eval config."
            )
            return

        # if the dataset has the process_trackeval_results method, use it to process the results
        if self.tracking_dataset is not None and hasattr(
            self.tracking_dataset, "process_trackeval_results"
        ):
            self.tracking_dataset.process_trackeval_results(
                results, dataset_config, eval_config
            )


def _print_results(
    res_combined: Dict[str, float],
    res_by_video: Optional[Dict[str, Dict[str, float]]] = None,
    scale_factor: float = 1.0,
    title: str = "",
    print_by_video: bool = False,
) -> None:
    """Print evaluation results in a formatted table.

    Args:
        res_combined: Combined results across all videos.
        res_by_video: Results broken down by individual videos.
        scale_factor: Factor to scale metric values for display.
        title: Title for the results table.
        print_by_video: Whether to print per-video results.
    """
    headers = list(res_combined.keys())
    data = [format_metric(name, res_combined[name], scale_factor) for name in headers]
    log.info(f"{title}\n" + tabulate([data], headers=headers, tablefmt="plain"))
    if print_by_video and res_by_video:
        data = []
        for video_name, res in res_by_video.items():
            video_data = [video_name] + [
                format_metric(name, res[name], scale_factor) for name in headers
            ]
            data.append(video_data)
        headers = ["video"] + list(headers)
        log.info(
            f"{title} by videos\n" + tabulate(data, headers=headers, tablefmt="plain")
        )


def format_metric(
    metric_name: str, metric_value: Union[int, float], scale_factor: float
) -> Union[int, float]:
    """Format a metric value for display.

    Args:
        metric_name: Name of the metric.
        metric_value: Raw metric value.
        scale_factor: Factor to scale the value.

    Returns:
        Formatted metric value.
    """
    if (
        "TP" in metric_name
        or "FN" in metric_name
        or "FP" in metric_name
        or "TN" in metric_name
    ):
        if metric_name == "MOTP":
            return np.around(metric_value * scale_factor, 3)
        return int(metric_value)
    else:
        return np.around(metric_value * scale_factor, 3)
