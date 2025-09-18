"""
YOLO Ultralytics detector for TrackLab with configurable class mappings.

TRAINING REQUIREMENTS:
- The DATASET is responsible for providing proper train/val/test splits
- The detector assumes the dataset has already validated splits exist
- Split names are flexible - dataset can use any naming convention
- eval_set from dataset config is respected for validation split selection
- No fallback logic in detector - dataset handles split management

CONFIG EXAMPLE:
  class_mapping:
    names: {0: "person", 1: "ball"}
    source_to_yolo: {0: 0, 1: 1}
    yolo_to_source: {0: 0, 1: 1}
"""

import logging
from typing import Any
import torch
import pandas as pd
from ultralytics import YOLO
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
import yaml
from pathlib import Path
import shutil
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

        # Initialize path configuration
        self._setup_paths()

        # Load or initialize model
        if hasattr(cfg, "path_to_checkpoint") and cfg.path_to_checkpoint:
            self.model = YOLO(cfg.path_to_checkpoint)
        else:
            # Initialize with default model if no checkpoint provided
            default_model = getattr(cfg, "default_model", "yolov8n.pt")
            self.model = YOLO(default_model)

        # Initialize class mappings (will be set during training or from config)
        self.yolo_to_source_mapping = None
        self.source_to_yolo_mapping = None

    def _cleanup_memory(self):
        """Clean up GPU memory after training operations"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log.debug("GPU memory cleaned up")

    def _setup_paths(self):
        """Setup and standardize path configuration following TrackLab's framework"""
        # Use Hydra's current working directory pattern (following TrackLab framework)
        # TrackLab uses outputs/${experiment_name}/${date}/${time} structure
        import os
        from datetime import datetime

        # Get current working directory (set by Hydra)
        cwd = Path.cwd()

        # Follow TrackLab's output directory structure
        # If we're in a Hydra output dir, use it; otherwise create our own
        if "outputs" in str(cwd):
            # We're already in a Hydra output directory
            base_output_dir = cwd
        else:
            # Create output directory following TrackLab pattern
            timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            experiment_name = self.cfg.get("experiment_name", "yolo_training")
            base_output_dir = cwd / "outputs" / experiment_name / timestamp

        # Get base paths from config or use TrackLab-style defaults
        self.base_training_dir = Path(
            self.cfg.get("base_training_dir", base_output_dir / "yolo_training")
        )
        self.base_data_dir = Path(
            self.cfg.get("base_data_dir", base_output_dir / "yolo_data")
        )
        self.base_incremental_dir = Path(
            self.cfg.get("base_incremental_dir", base_output_dir / "yolo_incremental")
        )

        # Model type for naming - use dataset name if available, otherwise fallback to model_type or "custom"
        self.model_type = self.cfg.get("dataset_name") or self.cfg.get(
            "model_type", "custom"
        )

        # Create standard path templates
        self.training_paths = {
            "project": str(self.base_training_dir),
            "name": f"yolo_{self.model_type}",
            "weights_dir": self.base_training_dir
            / f"yolo_{self.model_type}"
            / "weights",
            "best_model": self.base_training_dir
            / f"yolo_{self.model_type}"
            / "weights"
            / "best.pt",
            "data_yaml": self.base_data_dir / "data.yaml",
            "incremental_project": str(self.base_incremental_dir),
            "incremental_name": "incremental_yolo",
            "incremental_best": self.base_incremental_dir
            / "incremental_yolo"
            / "weights"
            / "best.pt",
        }

        log.debug(f"Initialized training paths: {self.training_paths}")

    def train(self, tracking_dataset, pipeline, evaluator, dataset_config):
        """Train YOLO model following TrackLab's simple training pattern

        The dataset is responsible for providing proper train/val/test splits.
        This method assumes the dataset has already validated and prepared the splits.

        Args:
            tracking_dataset: Dataset object with train/val/test splits
            pipeline: TrackLab pipeline object
            evaluator: Evaluation object
            dataset_config: Dataset configuration

        Raises:
            ValueError: If dataset doesn't provide required splits
        """
        log.info("Starting YOLO training...")

        # Get eval_set from dataset config if available (e.g., MOT17 uses "train" for eval)
        dataset_eval_set = None
        try:
            # Try dictionary access first (most common case)
            dataset_eval_set = dataset_config["eval_set"]
        except (KeyError, TypeError, AttributeError):
            # Fall back to attribute access
            dataset_eval_set = getattr(dataset_config, "eval_set", None)

        if dataset_eval_set:
            log.info(f"📊 Dataset specifies eval_set: '{dataset_eval_set}'")
            log.info(
                f"✅ Using '{dataset_eval_set}' split for validation (from dataset config)"
            )

        # Dataset should provide these splits - just use them
        available_sets = list(tracking_dataset.sets.keys())
        log.info(f"Available dataset splits: {available_sets}")

        # Assume dataset provides standard splits, but be flexible
        train_split = dataset_config.get("train_split", "train")
        val_split = dataset_config.get(
            "val_split",
            dataset_eval_set if dataset_eval_set in available_sets else "val",
        )
        test_split = dataset_config.get("test_split", "test")

        # Basic validation - dataset should handle the heavy lifting
        if train_split not in available_sets:
            raise ValueError(
                f"Dataset must provide '{train_split}' split. Available: {available_sets}"
            )

        if (
            val_split not in available_sets
            and "val" not in available_sets
            and "eval" not in available_sets
        ):
            raise ValueError(
                f"Dataset must provide validation split ('{val_split}', 'val', or 'eval'). Available: {available_sets}"
            )

        log.info("✅ Dataset validation passed - proceeding with training")

        # Get training configuration (simple TrackLab style)
        train_cfg = self.training_config
        if not train_cfg:
            log.warning("No training configuration found, using defaults")
            train_cfg = {"epochs": 100, "batch_size": 16, "img_size": 640}

        # Prepare training data
        data_path = train_cfg.get("data_path")
        if not data_path:
            data_path = self._prepare_training_data(tracking_dataset, dataset_config)

        # Simple training arguments (TrackLab style)
        training_args = {
            "data": data_path,
            "epochs": train_cfg.get("epochs", 100),
            "batch": train_cfg.get("batch_size", 16),
            "imgsz": train_cfg.get("img_size", 640),
            "device": self.device,
            "project": str(self.base_training_dir),
            "name": f"yolo_{self.model_type}",
            "exist_ok": True,
            "verbose": True,
        }

        # Add optional training parameters if specified
        for param in ["optimizer", "lr0", "lrf", "freeze"]:
            if param in train_cfg:
                training_args[param] = train_cfg[param]

        log.info(f"Training YOLO with: {training_args}")

        try:
            # Train the model (simple and direct)
            results = self.model.train(**training_args)

            # Update model to use trained weights
            best_model_path = self.training_paths["best_model"]
            if best_model_path.exists():
                log.info(f"Training completed. Loading best model: {best_model_path}")
                self.model = YOLO(str(best_model_path))
                self.cfg.path_to_checkpoint = str(best_model_path)

            return results

        except Exception as e:
            log.error(f"Training failed: {e}")
            raise
        finally:
            self._cleanup_memory()

    def _prepare_training_data(self, tracking_dataset, dataset_config):
        """Prepare training data from TrackLab dataset in YOLO format

        The dataset is responsible for providing proper splits. This method
        assumes the dataset has already validated and prepared the splits.

        Args:
            tracking_dataset: Dataset with train/val/test splits
            dataset_config: Dataset configuration

        Returns:
            str: Path to generated data.yaml file
        """
        import shutil

        log.info("Preparing training data from TrackLab dataset...")

        # Set up class mappings for this training session
        self.source_to_yolo_mapping = self._get_source_to_yolo_mapping(dataset_config)
        self.yolo_to_source_mapping = self._get_yolo_to_source_mapping(dataset_config)

        # Create directory structure using standardized paths
        base_path = self.base_data_dir
        images_dir = base_path / "images"
        labels_dir = base_path / "labels"

        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

        # Get available sets from dataset
        available_sets = list(tracking_dataset.sets.keys())
        if not available_sets:
            raise ValueError("Dataset must provide training splits")

        # Get eval_set from dataset config if available
        dataset_eval_set = None
        try:
            dataset_eval_set = dataset_config["eval_set"]
        except (KeyError, TypeError, AttributeError):
            dataset_eval_set = getattr(dataset_config, "eval_set", None)

        # Determine which splits to use
        train_split = "train"
        val_split = dataset_eval_set if dataset_eval_set in available_sets else "val"
        test_split = "test"

        # Get class mapping from dataset config or use defaults
        class_mapping = self._get_class_mapping(dataset_config)

        log.info(f"Available dataset splits: {available_sets}")
        log.info(f"Using train split: '{train_split}', val split: '{val_split}'")

        # Process the splits provided by the dataset
        splits_to_process = {}

        # Always try to get train split
        if train_split in available_sets:
            splits_to_process["train"] = tracking_dataset.sets[train_split]

        # Get validation split (prefer eval_set, fallback to val/eval)
        if val_split in available_sets:
            splits_to_process["val"] = tracking_dataset.sets[val_split]
        elif "val" in available_sets:
            splits_to_process["val"] = tracking_dataset.sets["val"]
        elif "eval" in available_sets:
            splits_to_process["val"] = tracking_dataset.sets["eval"]

        # Get test split if available
        if test_split in available_sets:
            splits_to_process["test"] = tracking_dataset.sets[test_split]

        if not splits_to_process.get("train"):
            raise ValueError(f"Dataset must provide '{train_split}' split")

        if not splits_to_process.get("val"):
            raise ValueError(f"Dataset must provide validation split")

        # Process each split
        for split_name, dataset_split in splits_to_process.items():
            log.info(f"Processing {split_name} split")
            self._process_dataset_split(
                dataset_split,
                split_name,
                class_mapping,
                images_dir,
                labels_dir,
                dataset_config,
            )

        # Create data.yaml
        data_yaml = {
            "path": str(base_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": class_mapping,
            "nc": len(class_mapping),
        }

        data_path = self.training_paths["data_yaml"]
        with open(data_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        log.info(f"Created YOLO training data at {base_path}")
        log.info(f"Class mapping: {class_mapping}")
        return str(data_path)

    def _get_class_mapping(self, dataset_config):
        """Get class mapping from dataset config or use defaults"""
        # Check if class mapping is specified in cfg
        if hasattr(self.cfg, "class_mapping") and self.cfg.class_mapping:
            class_cfg = self.cfg.class_mapping
            if "names" in class_cfg:
                return class_cfg["names"]

        # Try to extract from dataset config
        if hasattr(dataset_config, "get") and "names" in dataset_config:
            return dataset_config["names"]

        # Default mapping for common tracking datasets
        return {0: "person", 1: "ball"}

    def _get_source_to_yolo_mapping(self, dataset_config):
        """Get source dataset category_id to YOLO class_id mapping"""
        # Check if mapping is specified in cfg
        if hasattr(self.cfg, "class_mapping") and self.cfg.class_mapping:
            class_cfg = self.cfg.class_mapping
            if "source_to_yolo" in class_cfg:
                return class_cfg["source_to_yolo"]

        # Default mapping (identity for simple cases)
        return {0: 0, 1: 1}

    def _get_yolo_to_source_mapping(self, dataset_config):
        """Get YOLO class_id to source dataset category_id mapping"""
        # Check if mapping is specified in cfg
        if hasattr(self.cfg, "class_mapping") and self.cfg.class_mapping:
            class_cfg = self.cfg.class_mapping
            if "yolo_to_source" in class_cfg:
                return class_cfg["yolo_to_source"]

        # Default mapping (identity for simple cases)
        return {0: 0, 1: 1}

    def _process_dataset_split(
        self,
        dataset_split,
        split_name,
        class_mapping,
        images_dir,
        labels_dir,
        dataset_config,
    ):
        """Process a complete dataset split (train/val/test)"""
        # Count total images across all videos in this split
        total_images = 0
        for video_id in dataset_split.video_metadatas.index:
            video_images = dataset_split.image_metadatas[
                dataset_split.image_metadatas.video_id == video_id
            ]
            total_images += len(video_images)

        log.info(f"Processing {total_images} images in {split_name} split")

        with tqdm(
            total=total_images, desc=f"Processing {split_name} split", unit="image"
        ) as pbar:
            for video_id in dataset_split.video_metadatas.index:
                video_images = dataset_split.image_metadatas[
                    dataset_split.image_metadatas.video_id == video_id
                ]
                video_detections = dataset_split.detections_gt[
                    dataset_split.detections_gt.video_id == video_id
                ]

                for image_id, image_row in video_images.iterrows():
                    # Copy image (data loader should handle image processing)
                    image_path = image_row["file_path"]
                    if not Path(image_path).exists():
                        log.warning(f"Image {image_path} not found, skipping")
                        pbar.update(1)
                        continue

                    image_filename = f"{image_id}.jpg"
                    dest_image_path = images_dir / split_name / image_filename
                    shutil.copy2(image_path, dest_image_path)

                    # Create annotation file (delegate coordinate processing to data loader)
                    label_filename = f"{image_id}.txt"
                    label_path = labels_dir / split_name / label_filename

                    # Get detections for this image
                    image_dets = video_detections[video_detections.image_id == image_id]

                    # Write YOLO annotations (data loader should handle coordinate normalization)
                    self._write_yolo_annotations(label_path, image_dets, dataset_config)

                    pbar.update(1)

    def _write_yolo_annotations(self, label_path, image_dets, dataset_config):
        """Write YOLO format annotations for an image

        NOTE: This is a temporary implementation. Best practice would be to:
        1. Use a dedicated YOLODataLoader class for coordinate transformations
        2. Handle different coordinate systems (ltwh, xyxy, normalized, etc.)
        3. Validate coordinate ranges and image dimensions
        4. Support different annotation formats
        """
        with open(label_path, "w") as f:
            if len(image_dets) > 0:
                for _, det in image_dets.iterrows():
                    # Get class id using configurable mapping
                    category_id = det.get("category_id", 1)
                    source_to_yolo = self._get_source_to_yolo_mapping(dataset_config)

                    if category_id in source_to_yolo:
                        class_id = source_to_yolo[category_id]
                        # TODO: Use data loader for proper coordinate normalization
                        # For now, assume bbox is already in YOLO format or will be processed by loader
                        bbox = det["bbox_ltwh"]
                        if isinstance(bbox, str):
                            bbox = [float(x) for x in bbox.strip("[]").split()]

                        # Write YOLO format: class_id center_x center_y width height
                        # Data loader should handle proper normalization
                        x, y, w, h = bbox
                        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    # Skip categories we don't want

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
        detection_list = []

        # Use configured mapping or fallback to defaults
        if self.yolo_to_source_mapping is not None:
            yolo_to_source = self.yolo_to_source_mapping
        else:
            # Fallback to identity mapping if no training was done
            yolo_to_source = {0: 0, 1: 1}

        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # Check if this is one of our target classes and meets confidence threshold
                yolo_class = int(bbox.cls[0])
                if (
                    yolo_class in yolo_to_source
                    and bbox.conf[0] >= self.cfg.min_confidence
                ):
                    # Map back to source dataset category_id
                    category_id = yolo_to_source[yolo_class]

                    detection_list.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=category_id,
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detection_list
