"""PRT ReID module for TrackLab."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf

try:
    from yacs.config import CfgNode as CN
except ImportError:
    CN = None  # type: ignore

from tracklab.pipeline import DetectionLevelModule

# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.collate import default_collate
from .prtreid_dataset import ReidDataset

try:
    from prtreid.scripts.main import build_config, build_torchreid_model_engine
    from prtreid.tools.feature_extractor import FeatureExtractor
    from tracklab.utils.collate import Unbatchable
    import prtreid
    from prtreid.utils.tools import extract_test_embeddings
    from prtreid.data.datasets import configure_dataset_class
    from prtreid.scripts.default_config import engine_run_kwargs
except ImportError:
    build_config = None  # type: ignore
    build_torchreid_model_engine = None  # type: ignore
    FeatureExtractor = None  # type: ignore
    Unbatchable = None  # type: ignore
    prtreid = None  # type: ignore
    extract_test_embeddings = None  # type: ignore
    configure_dataset_class = None  # type: ignore
    engine_run_kwargs = None  # type: ignore

from tracklab.utils.download import download_file


class PRTReId(DetectionLevelModule):
    """PRT ReID module for TrackLab.

    This module performs person re-identification using pose-guided
    feature extraction and role detection for soccer players.
    """

    collate_fn = default_collate
    input_columns: List[str] = ["bbox_ltwh"]
    output_columns: List[str] = [
        "embeddings",
        "visibility_scores",
        "body_masks",
        "role_detection",
        "role_confidence",
    ]
    forget_columns: List[str] = ["embeddings", "body_masks"]
    role_mapping: Dict[Optional[str], int] = {
        "ball": 0,
        "goalkeeper": 1,
        "other": 2,
        "player": 3,
        "referee": 4,
        None: -1,
    }

    def __init__(
        self,
        cfg: Any,
        tracking_dataset: Any,
        dataset: Any,
        device: str,
        save_path: str,
        job_id: int,
        use_keypoints_visibility_scores_for_reid: bool,
        training_enabled: bool,
        batch_size: int,
    ) -> None:
        """Initialize the PRT ReID module.

        Args:
            cfg: Configuration object for the model.
            tracking_dataset: Tracking dataset configuration.
            dataset: Dataset configuration.
            device: Device to run inference on.
            save_path: Path to save model outputs.
            job_id: Job identifier.
            use_keypoints_visibility_scores_for_reid: Whether to use keypoints visibility.
            training_enabled: Whether training is enabled.
            batch_size: Batch size for processing.
        """
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        tracking_dataset.name = dataset.name
        tracking_dataset.nickname = dataset.nickname
        self.dataset_cfg = dataset
        self.use_keypoints_visibility_scores_for_reid = (
            use_keypoints_visibility_scores_for_reid
        )
        tracking_dataset.name = self.dataset_cfg.name
        tracking_dataset.nickname = self.dataset_cfg.nickname
        additional_args = {
            "tracking_dataset": tracking_dataset,
            "reid_config": self.dataset_cfg,
            "role_mapping": self.role_mapping,
            "pose_model": None,
        }
        prtreid.data.register_image_dataset(  # type: ignore
            tracking_dataset.name,
            configure_dataset_class(ReidDataset, **additional_args),  # type: ignore
            tracking_dataset.nickname,
        )
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
        self.download_models(
            load_weights=self.cfg.model.load_weights,
            pretrained_path=self.cfg.model.bpbreid.hrnet_pretrained_path,
            backbone=self.cfg.model.bpbreid.backbone,
        )
        self.inverse_role_mapping = {v: k for k, v in self.role_mapping.items()}
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config=self.cfg)  # type: ignore
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = training_enabled
        self.feature_extractor = None
        self.model = None

    def download_models(
        self, load_weights: str, pretrained_path: str, backbone: str
    ) -> None:
        """Download model weights if not present locally.

        Args:
            load_weights: Path to the model weights file.
            pretrained_path: Path to pretrained backbone weights.
            backbone: Backbone model name.
        """
        if Path(load_weights).name == "prtreid-soccernet-baseline.pth.tar":
            md5 = "9633825232bc89f23a94522c5561650e"
            download_file(
                "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1",
                local_filename=str(load_weights),  # Convert Path to str
                md5=md5,
            )
        if backbone == "hrnet32":
            md5 = "58ea12b0420aa3adaa2f74114c9f9721"
            path = Path(pretrained_path) / "hrnetv2_w32_imagenet_pretrained.pth"
            download_file(
                "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1",
                local_filename=str(path),  # Convert Path to str
                md5=md5,
            )

    @torch.no_grad()
    def preprocess(
        self, image: Any, detection: pd.Series, metadata: pd.Series
    ) -> Dict[str, Any]:  # Tensor RGB (1, 3, H, W)
        """Preprocess image and detection for ReID feature extraction.

        Args:
            image: Input image array.
            detection: Detection series containing bounding box.
            metadata: Image metadata series.

        Returns:
            Dictionary containing preprocessed batch data.
        """
        mask_w, mask_h = 32, 64
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])  # type: ignore
        batch = {
            "img": crop,
        }

        return batch

    @torch.no_grad()
    def process(
        self, batch: Dict[str, Any], detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> pd.DataFrame:
        """Process batch and extract ReID embeddings and role information.

        Args:
            batch: Preprocessed batch data.
            detections: Detection DataFrame to update.
            metadatas: Image metadata DataFrame.

        Returns:
            DataFrame with ReID features and role detections.
        """
        im_crops = batch["img"]
        im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]
        if "masks" in batch:
            external_parts_masks = batch["masks"]
            external_parts_masks = external_parts_masks.cpu().detach().numpy()
        else:
            external_parts_masks = None
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(  # type: ignore
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.device,
                image_size=(self.cfg.data.height, self.cfg.data.width),
                model=self.model,
                verbose=False,  # FIXME @Vladimir
            )
        reid_result = self.feature_extractor(
            im_crops, external_parts_masks=external_parts_masks
        )
        embeddings, visibility_scores, body_masks, _, role_cls_scores = (
            extract_test_embeddings(reid_result, self.test_embeddings)  # type: ignore
        )

        role_scores_: Any = []
        role_scores_.append(
            role_cls_scores["globl"].cpu() if role_cls_scores is not None else None
        )
        role_scores_ = (
            torch.cat(role_scores_, 0) if role_scores_[0] is not None else None
        )
        roles = [torch.argmax(i).item() for i in role_scores_] if role_scores_ is not None else []  # type: ignore
        roles = [self.inverse_role_mapping[int(index)] for index in roles] if roles else []  # type: ignore
        role_confidence = [torch.max(i).item() for i in role_scores_] if role_scores_ is not None else []  # type: ignore

        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()
        body_masks = body_masks.cpu().detach().numpy()

        if self.use_keypoints_visibility_scores_for_reid:
            kp_visibility_scores = batch["visibility_scores"].numpy()
            if visibility_scores.shape[1] > kp_visibility_scores.shape[1]:
                kp_visibility_scores = np.concatenate(
                    [np.ones((visibility_scores.shape[0], 1)), kp_visibility_scores],
                    axis=1,
                )
            visibility_scores = np.float32(kp_visibility_scores)

        reid_df = pd.DataFrame(
            {
                "embeddings": list(embeddings),
                "visibility_scores": list(visibility_scores),  # type: ignore
                "body_masks": list(body_masks),
                "role_detection": roles,
                "role_confidence": role_confidence,
            },
            index=detections.index,
        )
        return reid_df

    def train(self) -> None:
        """Train the PRT ReID model.

        Args:
            None

        Returns:
            None
        """
        self.engine, self.model = build_torchreid_model_engine(self.cfg)  # type: ignore
        self.engine.run(**engine_run_kwargs(self.cfg))  # type: ignore
