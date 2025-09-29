"""MMPose-based pose estimation modules for TrackLab."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch
import numpy as np
import pandas as pd

import mmcv  # type: ignore
from mmcv.parallel import scatter  # type: ignore
from mmengine.dataset import Compose, default_collate
from mim import get_model_info  # type: ignore
from mim.utils import get_installed_path  # type: ignore
from mmpose.apis.inference import dataset_meta_from_config  # type: ignore
from mmpose.apis import init_model, init_pose_model  # type: ignore
from mmpose.core.post_processing import oks_nms  # type: ignore
from mmpose.datasets.dataset_info import DatasetInfo  # type: ignore
from mmpose.datasets.pipelines import Compose as MMPoseCompose  # type: ignore

from tracklab.pipeline import ImageLevelModule, DetectionLevelModule
from tracklab.utils.openmmlab import get_checkpoint
from tracklab.utils.coordinates import sanitize_keypoints, generate_bbox_from_keypoints

import logging

log = logging.getLogger(__name__)


def mmpose_collate(batch: List[Any]) -> Any:
    """Custom collate function for MMPose batch processing.

    Args:
        batch: List of batch items to collate.

    Returns:
        Collated batch data.
    """
    return default_collate(batch)  # type: ignore


@torch.no_grad()
class BottomUpMMPose(ImageLevelModule):
    """Bottom-up MMPose pose estimation module.

    This module performs bottom-up pose estimation, detecting all persons
    and their keypoints in an image simultaneously without requiring
    bounding box detections as input.
    """

    collate_fn = mmpose_collate
    output_columns: List[str] = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
        "keypoints_xyc",
        "keypoints_conf",
    ]

    def __init__(self, cfg: Any, device: str, batch_size: int) -> None:
        """Initialize the bottom-up MMPose module.

        Args:
            cfg: Configuration object containing model settings.
            device: Device to run inference on.
            batch_size: Batch size for processing.
        """
        super().__init__(batch_size)
        get_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.device = device if device != "cpu" else -1
        self.model = init_pose_model(cfg.path_to_config, cfg.path_to_checkpoint, device)
        self.id: int = 0

        self.cfg = self.model.cfg
        self.dataset_info = DatasetInfo(self.cfg.dataset_info)

        self.test_pipeline = MMPoseCompose(self.cfg.test_pipeline)

    @torch.no_grad()
    def preprocess(self, metadata: pd.Series) -> Any:
        """Preprocess image metadata for pose estimation.

        Args:
            metadata: Image metadata series containing file path.

        Returns:
            Preprocessed data for model inference.
        """
        image = cv2.imread(
            metadata.file_path, flags=cv2.IMREAD_COLOR_BGR
        )  # BGR not RGB !
        data = {
            "dataset": self.dataset_info.dataset_name,
            "img": image,
            "ann_info": {
                "image_size": np.array(self.model.cfg.data_cfg["image_size"]),
                "heatmap_size": self.model.cfg.data_cfg.get("heatmap_size", None),
                "num_joints": self.model.cfg.data_cfg["num_joints"],
                "flip_index": self.dataset_info.flip_index,
                "skeleton": self.dataset_info.skeleton,
            },
        }
        return self.test_pipeline(data)

    @torch.no_grad()
    def process(self, batch: Any, metadatas: pd.DataFrame) -> List[pd.Series]:
        """Process batch and extract pose detections.

        Args:
            batch: Preprocessed batch data.
            metadatas: Image metadata DataFrame.

        Returns:
            List of detection Series with pose information.
        """
        batch = scatter(batch, [self.device])[0]
        images = list(batch["img"].unsqueeze(0).permute(1, 0, 2, 3, 4))
        detections = []
        for image, img_metas, (_, metadata) in zip(
            images, batch["img_metas"], metadatas.iterrows()
        ):
            result = self.model(
                img=image,
                img_metas=[img_metas],
                return_loss=False,
                return_heatmap=False,
            )
            pose_results = []
            for idx, pred in enumerate(result["preds"]):
                area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                    np.max(pred[:, 1]) - np.min(pred[:, 1])
                )
                pose_results.append(
                    {
                        "keypoints": pred[:, :3],
                        "score": result["scores"][idx],
                        "area": area,
                    }
                )

            # pose nms
            score_per_joint = self.model.cfg.model.test_cfg.get(
                "score_per_joint", False
            )
            keep = oks_nms(
                pose_results,
                self.cfg.nms_threshold,
                self.dataset_info.sigmas,
                score_per_joint=score_per_joint,
            )
            pose_results = [pose_results[_keep] for _keep in keep]

            for pose in pose_results:
                if pose["score"] >= self.cfg.min_confidence:
                    image_shape = (image.shape[2], image.shape[1])
                    keypoints = sanitize_keypoints(pose["keypoints"], image_shape)
                    bbox = generate_bbox_from_keypoints(
                        keypoints, self.cfg.bbox.extension_factor, image_shape
                    )
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                keypoints_xyc=keypoints,
                                keypoints_conf=pose["score"],
                                bbox_ltwh=bbox,
                                bbox_conf=pose["score"],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections


class TopDownMMPose(DetectionLevelModule):
    """Top-down MMPose pose estimation module.

    This module performs top-down pose estimation, taking bounding box
    detections as input and estimating keypoints for each detected person.
    """

    collate_fn = default_collate
    input_columns: List[str] = ["bbox_ltwh", "bbox_conf"]
    output_columns: List[str] = ["keypoints_xyc", "keypoints_conf"]

    def __init__(
        self,
        device: str,
        batch_size: int,
        config_name: str,
        path_to_checkpoint: str,
        vis_kp_threshold: float = 0.4,
        min_num_vis_kp: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize the top-down MMPose module.

        Args:
            device: Device to run inference on.
            batch_size: Batch size for processing.
            config_name: Name of the MMPose model configuration.
            path_to_checkpoint: Path to model checkpoint.
            vis_kp_threshold: Threshold for keypoint visibility.
            min_num_vis_kp: Minimum number of visible keypoints required.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size)
        model_df = get_model_info(package="mmpose", configs=[config_name])
        if len(model_df) != 1:
            raise ValueError("Multiple values found for the config name")
        download_url = model_df.weight.item()
        package_path = Path(get_installed_path("mmpose"))
        path_to_config = package_path / ".mim" / model_df.config.item()
        get_checkpoint(path_to_checkpoint, download_url)
        self.model = init_model(str(path_to_config), path_to_checkpoint, device)
        self.vis_kp_threshold = vis_kp_threshold
        self.min_num_vis_kp = min_num_vis_kp
        self.dataset_info = dataset_meta_from_config(self.model.cfg, "test")

        # self.dataset_info = DatasetInfo(self.model.cfg.dataset_info)
        self.test_pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)

    @torch.no_grad()
    def preprocess(self, image: Any, detection: pd.Series, metadata: pd.Series) -> Any:
        """Preprocess image and detection for pose estimation.

        Args:
            image: Input image array.
            detection: Detection series containing bounding box.
            metadata: Image metadata series.

        Returns:
            Preprocessed data for model inference.
        """
        data_info = dict(img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        data_info["bbox"] = detection.bbox.ltrb()[None]
        data_info["bbox_score"] = np.array(detection.bbox_conf)[None]
        data_info.update(self.model.dataset_meta)

        return self.test_pipeline(data_info)

    @torch.no_grad()
    def process(
        self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> pd.DataFrame:
        """Process batch and extract keypoints for detections.

        Args:
            batch: Preprocessed batch data.
            detections: Detection DataFrame to update with keypoints.
            metadatas: Image metadata DataFrame.

        Returns:
            Updated detections DataFrame with keypoints.
        """
        results = self.model.test_step(batch)
        kps_xyc = []
        kps_conf = []
        for result in results:
            result = result.pred_instances
            keypoints = result.keypoints[0]
            visibility_scores = result.keypoints_visible[0]
            visibility_scores[visibility_scores < self.vis_kp_threshold] = 0
            keypoints_xyc = np.concatenate(
                [keypoints, visibility_scores[:, None]], axis=-1
            )
            if len(np.nonzero(visibility_scores)[0]) < self.min_num_vis_kp:
                conf = 0
            else:
                conf = np.mean(visibility_scores[visibility_scores != 0])
            kps_xyc.append(keypoints_xyc)
            kps_conf.append(conf)
        detections["keypoints_conf"] = kps_conf
        detections["keypoints_xyc"] = kps_xyc
        return detections
