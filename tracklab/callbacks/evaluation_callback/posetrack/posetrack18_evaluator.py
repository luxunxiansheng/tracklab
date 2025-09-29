"""PoseTrack18 evaluator for TrackLab pose tracking evaluation."""

import logging
import os
from typing import Any, TYPE_CHECKING

import numpy as np
from tabulate import tabulate

from tracklab.callbacks.evaluation_callback.evaluator import Evaluator as EvaluatorBase
from tracklab.utils import wandb

if TYPE_CHECKING:
    from tracklab.datastruct import TrackerState

try:
    from poseval.eval_helpers import (
        load_data_dir,
        Joint,
        mapmetrics2dict,
        precmetrics2dict,
        recallmetrics2dict,
        motmetrics2dict,
    )
    from poseval.evaluateAP import evaluateAP
    from poseval.evaluateTracking import evaluateTracking
except ImportError:
    poseval = None  # type: ignore

from .posetrack21_evaluator import PoseTrack21Evaluator as PTEvaluator

log = logging.getLogger(__name__)


class PoseTrack18Evaluator(EvaluatorBase):
    """Evaluator for PoseTrack18 dataset pose tracking evaluation.

    This evaluator performs comprehensive evaluation of pose tracking results
    on the PoseTrack18 dataset, including pose estimation and pose tracking
    metrics using the official PoseTrack evaluation toolkit.
    """

    def __init__(self, cfg: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the PoseTrack18 evaluator.

        Args:
            cfg: Configuration object containing evaluation parameters.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg

    def run(self, tracker_state: "TrackerState") -> None:
        """Run PoseTrack18 evaluation on the tracker state.

        Args:
            tracker_state: The tracker state containing predictions and ground truth.
        """
        log.info("Starting evaluation on PoseTrack18")
        images = PTEvaluator._images(tracker_state.image_metadatas)
        category = PTEvaluator._category(tracker_state.video_metadatas)
        seqs = list(tracker_state.video_metadatas.name)
        bbox_column = self.cfg.bbox_column_for_eval
        eval_pose_on_all = self.cfg.eval_pose_on_all
        if self.cfg.eval_pose_estimation:
            annotations = PTEvaluator._annotations_pose_estimation_eval(
                tracker_state.detections_pred,
                tracker_state.image_metadatas,
                bbox_column,
                eval_pose_on_all,
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_estimation"
            )
            PTEvaluator._save_json(images, annotations, category, trackers_folder)

            # Bounding box evaluation
            bbox_map = PTEvaluator.compute_bbox_map(
                tracker_state.detections_pred,
                tracker_state.detections_gt,
                tracker_state.image_metadatas,
            )
            map(bbox_map.pop, ["map_per_class", "mar_100_per_class"])
            headers = bbox_map.keys()
            data = [np.round(100 * bbox_map[name].item(), 2) for name in headers]
            log.info(
                "Pose estimation - bbox metrics\n"
                + tabulate([data], headers=headers, tablefmt="plain")
            )
            wandb.log_metric(bbox_map, "PoseTrack18/bbox/AP")

            # Bounding box evaluation
            bbox_map = PTEvaluator.compute_bbox_map(
                tracker_state.detections_pred,
                tracker_state.detections_gt,
                tracker_state.image_metadatas,
            )
            map(bbox_map.pop, ["map_per_class", "mar_100_per_class"])
            headers = bbox_map.keys()
            data = [np.round(100 * bbox_map[name].item(), 2) for name in headers]
            log.info(
                "Pose estimation - bbox metrics\n"
                + tabulate([data], headers=headers, tablefmt="plain")
            )
            wandb.log_metric(bbox_map, "PoseTrack18/bbox/AP")

            # Keypoint evaluation
            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            apAll, preAll, recAll = evaluateAP(
                gtFramesAll, prFramesAll, "", False, False
            )
            res_combined = mapmetrics2dict(apAll)
            PTEvaluator._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints average precision",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack18/kp/AP")
            res_combined = precmetrics2dict(preAll)
            PTEvaluator._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints precision",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack18/kp/precision")
            res_combined = recallmetrics2dict(recAll)
            PTEvaluator._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose estimation - keypoints recall",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack18/kp/recall")

        if self.cfg.eval_pose_tracking:
            annotations = PTEvaluator._annotations_tracking_eval(
                tracker_state.detections_pred,
                tracker_state.image_metadatas,
                bbox_column,
            )
            trackers_folder = os.path.join(
                self.cfg.posetrack_trackers_folder, "pose_tracking"
            )
            PTEvaluator._save_json(images, annotations, category, trackers_folder)

            argv = ["", self.cfg.posetrack_gt_folder, trackers_folder]
            gtFramesAll, prFramesAll = load_data_dir(argv, seqs)
            metricsAll = evaluateTracking(gtFramesAll, prFramesAll, "", False, False)
            metrics = np.zeros([Joint().count + 4, 1])
            for i in range(Joint().count + 1):
                metrics[i, 0] = metricsAll["mota"][0, i]
            metrics[Joint().count + 1, 0] = metricsAll["motp"][0, Joint().count]
            metrics[Joint().count + 2, 0] = metricsAll["pre"][0, Joint().count]
            metrics[Joint().count + 3, 0] = metricsAll["rec"][0, Joint().count]
            res_combined = motmetrics2dict(metrics)
            PTEvaluator._print_results(
                res_combined,
                scale_factor=1.0,
                title="Pose tracking - keypoints MOTA",
                print_by_video=self.cfg.print_by_video,
            )
            wandb.log_metric(res_combined, "PoseTrack18/kp/MOTA")
