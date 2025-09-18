"""
YOLO Ultralytics detector for TrackLab with automatic class detection.

"""

import logging
from typing import Any, Dict, Union, List
from tracklab.datastruct.tracking_dataset import TrackingDataset
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.pipeline.module import Pipeline
from tracklab.utils.coordinates import ltrb_to_ltwh
from pathlib import Path
from tqdm import tqdm

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

    def __init__(self, cfg, device, batch_size, training_enabled=False, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.training_enabled = training_enabled
        self.id = 0

        # Load or initialize model
        if hasattr(cfg, "path_to_checkpoint") and cfg.path_to_checkpoint:
            self.model = YOLO(cfg.path_to_checkpoint)
        else:
            # Initialize with default model if no checkpoint provided
            default_model = getattr(cfg, "default_model", "yolov8n.pt")
            self.model = YOLO(default_model)

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(
        self,
        batch: tuple[list, tuple[list, list]],
        detections: pd.DataFrame,
        metadatas: pd.DataFrame,
    ) -> pd.DataFrame:
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

    def train(
        self,
        tracking_dataset: TrackingDataset,
        pipeline: Pipeline,
        evaluator: object,
        dataset_config: dict,
    ) -> None:
        """Train the YOLO model using the TrackingDataset.

        Args:
            tracking_dataset: The TrackingDataset containing train/valid/test sets
            pipeline: The TrackLab pipeline
            evaluator: The evaluator for validation
            dataset_config: Configuration for the dataset
        """
        import os
        import yaml
        from pathlib import Path
        import tempfile
        import shutil

        log.info("Starting YOLO training with TrackingDataset...")

        # Get training configuration
        train_cfg = getattr(self.cfg, "training", {})
        epochs = train_cfg.get("epochs", 50)
        batch_size = train_cfg.get("batch_size", 16)
        img_size = train_cfg.get("img_size", 640)

        # Create temporary directory for YOLO dataset format
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yolo_dataset_path = temp_path / "dataset"
            yolo_dataset_path.mkdir()

            log.info("📊 Step 1/3: Converting TrackingDataset to YOLO format...")
            # Convert TrackingDataset to YOLO format
            yolo_data_yaml = self._prepare_yolo_dataset(
                tracking_dataset,
                yolo_dataset_path,
                dataset_config,
                tracking_dataset.dataset_path,
            )
            log.info("✅ Dataset conversion completed!")

            log.info("🎯 Step 2/3: Running YOLO training...")
            # Train the model
            self._run_yolo_training(yolo_data_yaml, epochs, batch_size, img_size)
            log.info("✅ Model training completed!")

        log.info("🎉 YOLO training pipeline completed successfully!")

    def _prepare_yolo_dataset(
        self,
        tracking_dataset: TrackingDataset,
        output_path: Path,
        dataset_config: dict,
        dataset_path: Union[str, Path],
    ) -> Path:
        """Convert TrackingDataset to YOLO format and create dataset YAML.

        Args:
            tracking_dataset: The TrackingDataset to convert
            output_path: Path to save the YOLO dataset
            dataset_config: Dataset configuration
            dataset_path: Path to the original dataset

        Returns:
            Path to the created dataset YAML file
        """
        import yaml

        # Create directories for YOLO format
        images_train_dir = output_path / "images" / "train"
        images_val_dir = output_path / "images" / "valid"
        images_test_dir = output_path / "images" / "test"
        labels_train_dir = output_path / "labels" / "train"
        labels_val_dir = output_path / "labels" / "valid"
        labels_test_dir = output_path / "labels" / "test"

        for dir_path in [
            images_train_dir,
            images_val_dir,
            images_test_dir,
            labels_train_dir,
            labels_val_dir,
            labels_test_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Process each split
        splits_info = {}
        available_splits = [
            split_name
            for split_name in ["train", "valid", "test"]
            if split_name in tracking_dataset.sets
        ]

        log.info(
            f"Processing {len(available_splits)} dataset splits: {available_splits}"
        )
        for split_name in tqdm(
            available_splits, desc="Processing dataset splits", unit="split"
        ):
            tracking_set = tracking_dataset.sets[split_name]
            splits_info[split_name] = self._process_tracking_set(
                tracking_set, output_path, split_name, dataset_path
            )

        # Create dataset YAML
        data_yaml = {
            "path": str(output_path),
            "train": "images/train",
            "val": "images/valid",
            "test": "images/test",
            "names": {0: "person"},  # Only person class for bbox_detector
            "nc": 1,  # Number of classes
        }

        yaml_path = output_path / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        log.info(f"📄 Created YOLO dataset YAML at {yaml_path}")
        log.info(f"📊 Dataset summary: {splits_info}")
        return yaml_path

    from tracklab.datastruct.tracking_dataset import TrackingSet

    def _process_tracking_set(
        self,
        tracking_set: TrackingSet,
        output_path: Path,
        split_name: str,
        dataset_path: Union[str, Path],
    ) -> dict:
        """Process a TrackingSet and convert to YOLO format.

        Args:
            tracking_set: The TrackingSet to process
            output_path: Base output path
            split_name: Name of the split (train/valid/test)
            dataset_path: Path to the original dataset

        Returns:
            Dictionary with processing statistics
        """
        from PIL import Image
        import cv2

        images_dir = output_path / "images" / split_name
        labels_dir = output_path / "labels" / split_name

        processed_count = 0
        total_detections = 0

        # Group detections by image
        image_groups = tracking_set.detections_gt.groupby("image_id")
        total_images = len(image_groups)

        log.info(f"Processing {total_images} images for {split_name} split")

        for image_id, detections in tqdm(
            image_groups,
            desc=f"Processing {split_name} images",
            unit="img",
            total=total_images,
        ):
            try:
                # Get image metadata
                image_meta = tracking_set.image_metadatas.loc[image_id]

                # Load image
                image_path = Path(dataset_path) / image_meta.file_path
                if not image_path.exists():
                    log.warning(f"Image not found: {image_path}")
                    continue

                # Copy image to YOLO format directory
                img = cv2.imread(str(image_path))
                if img is None:
                    log.warning(f"Could not load image: {image_path}")
                    continue

                height, width = img.shape[:2]
                yolo_image_path = images_dir / f"{image_id}.jpg"
                cv2.imwrite(str(yolo_image_path), img)

                # Create label file
                label_path = labels_dir / f"{image_id}.txt"
                with open(label_path, "w") as f:
                    # Process detections with progress bar
                    for _, detection in tqdm(
                        detections.iterrows(),
                        desc=f"Processing detections for image {image_id}",
                        unit="det",
                        total=len(detections),
                        leave=False,
                    ):
                        # Filter and merge categories to person only
                        category_id = self._map_category_to_person(
                            detection, tracking_set
                        )

                        if category_id == 0:  # person class
                            # Convert bbox to YOLO format (normalized)
                            bbox = detection.bbox_ltwh
                            if isinstance(bbox, np.ndarray):
                                left, top, w, h = bbox
                            else:
                                left, top, w, h = bbox

                            x_center = (left + w / 2) / width
                            y_center = (top + h / 2) / height
                            w_norm = w / width
                            h_norm = h / height

                            # Ensure values are within [0, 1]
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            w_norm = max(0, min(1, w_norm))
                            h_norm = max(0, min(1, h_norm))

                            f.write(
                                f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                            )
                            total_detections += 1

                processed_count += 1

            except Exception as e:
                log.warning(f"Error processing image {image_id}: {e}")
                continue

        log.info(
            f"✅ Processed {processed_count}/{total_images} images for {split_name} split with {total_detections} detections"
        )
        return {
            "processed_images": processed_count,
            "total_detections": total_detections,
        }

    def _map_category_to_person(self, detection_row: dict, tracking_set: object) -> int:
        """Map any category to person class (0) for bbox_detector training.

        Args:
            detection_row: Row from detections DataFrame
            tracking_set: The TrackingSet containing category information

        Returns:
            Mapped category ID (0 for person, -1 to skip)
        """
        # For bbox_detector, we only want person detections
        # Map all person-related categories to class 0

        # Check if we have role information
        if "role" in detection_row:
            role = detection_row["role"]
            if role in ["player", "goalkeeper"]:
                return 0  # YOLO person class
            else:
                # Skip non-person categories (referee, ball, other)
                return -1

        # Fallback to category name checking
        if "category" in detection_row:
            category = str(detection_row["category"]).lower()
            if any(
                keyword in category for keyword in ["player", "goalkeeper", "person"]
            ):
                return 0  # YOLO person class
            else:
                return -1

        # Final fallback to category_id
        # This assumes category_id 1+ are person-related (common in many datasets)
        category_id = detection_row.get("category_id", -1)
        if category_id > 0:  # Assume positive category_ids are person-related
            return 0
        else:
            return -1

    def _run_yolo_training(
        self, data_yaml_path: Path, epochs: int, batch_size: int, img_size: int
    ) -> Any:
        """Run YOLO training with the prepared dataset.

        Args:
            data_yaml_path: Path to the dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Image size for training
        """
        from ultralytics import YOLO

        log.info(
            f"Starting YOLO training with {epochs} epochs, batch size {batch_size}"
        )

        # Get training configuration from cfg
        train_cfg = getattr(self.cfg, "training", {})

        # Set up training arguments
        train_args = {
            "data": str(data_yaml_path),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "device": self.device,
            "workers": train_cfg.get("workers", 8),
            "optimizer": train_cfg.get("optimizer", "AdamW"),
            "lr0": train_cfg.get("lr0", 0.0001),
            "lrf": train_cfg.get("lrf", 0.01),
            "momentum": train_cfg.get("momentum", 0.9),
            "weight_decay": train_cfg.get("weight_decay", 0.0005),
            "warmup_epochs": train_cfg.get("warmup_epochs", 3),
            "warmup_momentum": train_cfg.get("warmup_momentum", 0.8),
            "warmup_bias_lr": train_cfg.get("warmup_bias_lr", 0.1),
            "freeze": train_cfg.get("freeze", 10),
            "amp": train_cfg.get("amp", True),
            "cache": train_cfg.get("cache", False),
            "val": train_cfg.get("val", True),
            "save_period": train_cfg.get("save_period", 10),
            "patience": train_cfg.get("patience", 10),
            "plots": train_cfg.get("plots", True),
            "verbose": train_cfg.get("verbose", True),
        }

        # Add augmentation settings if available
        if "augmentation" in train_cfg:
            aug = train_cfg["augmentation"]
            train_args.update(
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
                    "copy_paste": aug.get("copy_paste", 0.0),
                }
            )

        # Train the model
        log.info("🚀 Starting YOLO model training...")
        results = self.model.train(**train_args)
        log.info("✅ YOLO training completed!")

        # Save the trained model
        if hasattr(self.cfg, "save_path") and self.cfg.save_path:
            save_path = Path(self.cfg.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            log.info(f"Saved trained model to {save_path}")
        else:
            # Save with default name
            default_save_path = Path("models") / "yolo_finetuned.pt"
            default_save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(default_save_path)
            log.info(f"Saved trained model to {default_save_path}")

        return results
