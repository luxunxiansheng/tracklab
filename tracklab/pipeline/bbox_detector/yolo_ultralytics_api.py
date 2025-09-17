import logging
from typing import Any
import torch
import pandas as pd
from ultralytics import YOLO
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
import yaml
from pathlib import Path

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOUltralytics(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]
    training_enabled = False  # Set to True in config to enable training

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.training_config = cfg.get("training", {})

        # Load or initialize model
        if hasattr(cfg, "path_to_checkpoint") and cfg.path_to_checkpoint:
            self.model = YOLO(cfg.path_to_checkpoint)
        else:
            # Initialize with default model if no checkpoint provided
            self.model = YOLO("yolov8n.pt")  # or any default model

        self.model.to(device)
        self.id = 0

    def train(self, tracking_dataset, pipeline, evaluator, dataset_config):
        """Train YOLO model following TrackLab's training pattern"""
        log.info("Starting YOLO training...")

        # Get training configuration
        train_cfg = self.training_config
        if not train_cfg:
            log.warning("No training configuration found, using defaults")
            train_cfg = {
                "epochs": 100,
                "batch_size": 16,
                "img_size": 640,
                "data_path": None,
            }

        # Prepare training data
        data_path = train_cfg.get("data_path")
        if not data_path:
            # Try to infer from dataset config or create default
            data_path = self._prepare_training_data(tracking_dataset, dataset_config)

        # Set up training arguments
        training_args = {
            "data": data_path,
            "epochs": train_cfg.get("epochs", 100),
            "batch": train_cfg.get("batch_size", 16),
            "imgsz": train_cfg.get("img_size", 640),
            "device": self.device,
            "workers": 8,
            "project": "tracklab_training",
            "name": f'yolo_{self.cfg.get("model_type", "custom")}',
            "exist_ok": True,
            "verbose": True,
        }

        # Add optional training parameters
        if "optimizer" in train_cfg:
            training_args["optimizer"] = train_cfg["optimizer"]
        if "lr0" in train_cfg:
            training_args["lr0"] = train_cfg["lr0"]
        if "lrf" in train_cfg:
            training_args["lrf"] = train_cfg["lrf"]

        # Add augmentation if specified
        if "augmentation" in train_cfg:
            aug = train_cfg["augmentation"]
            training_args.update(
                {
                    "hsv_h": aug.get("hsv_h", 0.015),
                    "hsv_s": aug.get("hsv_s", 0.7),
                    "hsv_v": aug.get("hsv_v", 0.4),
                    "degrees": aug.get("degrees", 0.0),
                    "translate": aug.get("translate", 0.1),
                    "scale": aug.get("scale", 0.5),
                    "shear": aug.get("shear", 0.0),
                    "perspective": aug.get("perspective", 0.0),
                    "flipud": aug.get("flipud", 0.0),
                    "fliplr": aug.get("fliplr", 0.5),
                    "mosaic": aug.get("mosaic", 1.0),
                    "mixup": aug.get("mixup", 0.0),
                }
            )

        log.info(f"Training YOLO with config: {training_args}")

        # Start training
        results = self.model.train(**training_args)

        # Update model path to trained weights
        trained_model_path = (
            Path("tracklab_training")
            / f'yolo_{self.cfg.get("model_type", "custom")}'
            / "weights"
            / "best.pt"
        )
        if trained_model_path.exists():
            log.info(f"Trained model saved to: {trained_model_path}")
            # Update config to use trained model for future inference
            self.cfg.path_to_checkpoint = str(trained_model_path)
            self.model = YOLO(str(trained_model_path))

        return results

    def _prepare_training_data(self, tracking_dataset, dataset_config):
        """Prepare training data from TrackLab dataset"""
        # This is a simplified implementation
        # In practice, you'd convert TrackLab's detection data to YOLO format

        log.info("Preparing training data from TrackLab dataset...")

        # Create a basic data.yaml
        data_yaml = {
            "path": "./training_data",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "person", 1: "ball"},  # Default classes
        }

        # Save data configuration
        data_path = Path("./training_data/data.yaml")
        data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(data_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        log.info(f"Created data configuration at {data_path}")
        return str(data_path)

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        results_by_image = self.model(images, verbose=False)
        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # check for `person` class
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections
