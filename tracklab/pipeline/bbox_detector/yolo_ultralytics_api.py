"""
YOLO Ultralytics detector for TrackLab with automatic class detection.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

from tracklab.datastruct.tracking_dataset import TrackingDataset
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.pipeline.module import Pipeline
from tracklab.utils.coordinates import ltrb_to_ltwh

try:
    from mmcv.ops import soft_nms
except ImportError:
    soft_nms = None

log = logging.getLogger(__name__)


def collate_fn(
    batch: List[Tuple[Any, Dict[str, Any]]],
) -> Tuple[List[Any], Tuple[List[Any], List[Tuple[int, int]]]]:
    """Collate function for batching detection data.

    Args:
        batch: List of (index, data) tuples.

    Returns:
        Tuple of (indices, (images, shapes)).
    """
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOUltralytics(ImageLevelModule):
    """YOLO Ultralytics detector for object detection in TrackLab.

    This module uses YOLO models from Ultralytics for detecting objects
    in images with configurable post-processing options.
    """

    collate_fn = collate_fn
    input_columns: List[str] = []
    output_columns: List[str] = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(
        self,
        cfg: Any,
        device: str,
        batch_size: int,
        training_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the YOLO Ultralytics detector.

        Args:
            cfg: Configuration object with model and processing settings.
            device: Device to run the model on (e.g., 'cpu', 'cuda').
            batch_size: Batch size for processing.
            training_enabled: Whether training mode is enabled.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.training_enabled = training_enabled
        self.id = 0

        # Extract TTA and post-processing configuration
        self.enable_tta: bool = False
        if hasattr(cfg, "tta") and cfg.tta:
            self.enable_tta = True

        # Extract post-processing configuration
        self.enable_soft_nms: bool = False
        self.size_filter_min_area_ratio: float = 0.0
        if hasattr(cfg, "postproc"):
            postproc = cfg.postproc
            if hasattr(postproc, "soft_nms"):
                self.enable_soft_nms = bool(postproc.soft_nms)
            if hasattr(postproc, "size_filter_min_area_ratio"):
                self.size_filter_min_area_ratio = float(
                    postproc.size_filter_min_area_ratio
                )

        # Load or initialize model - YOLO11x specific optimizations
        if hasattr(cfg, "path_to_checkpoint") and cfg.path_to_checkpoint:
            self.model = YOLO(cfg.path_to_checkpoint)
        else:
            # For YOLO11x, use smaller batch size but optimize other parameters
            default_model = getattr(cfg, "default_model", "yolo11x.pt")
            self.model = YOLO(default_model)

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series) -> Dict[str, Any]:
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(
        self,
        batch: tuple[list, tuple[list, list]],
        detections_df: pd.DataFrame,
        metadatas: pd.DataFrame,
    ) -> List[pd.Series]:
        images, shapes = batch

        # Enable TTA if configured
        augment = self.enable_tta
        results_by_image = self.model(images, augment=augment, verbose=False)

        detections_out: List[pd.Series] = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            # Check for missing video_id in metadata
            if not hasattr(metadata, "video_id"):
                log.warning(
                    f"Metadata missing video_id for image {metadata.name}: {dict(metadata)}"
                )

            # Extract detections for this image
            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes
                # Filter for person (class 0) and ball (class 1) with sufficient confidence
                valid_classes = [0, 1]  # 0: person, 1: ball
                for cls_id in valid_classes:
                    cls_mask = (boxes.cls == cls_id) & (
                        boxes.conf >= self.cfg.min_confidence
                    )
                    if cls_mask.sum() > 0:
                        cls_boxes = boxes.xyxy[cls_mask]
                        cls_confs = boxes.conf[cls_mask]

                        for i in range(len(cls_boxes)):
                            detection = {
                                "bbox": cls_boxes[i].cpu().numpy(),  # [x1, y1, x2, y2]
                                "conf": cls_confs[i].cpu().numpy(),
                                "image_id": metadata.name,
                                "video_id": getattr(metadata, "video_id", 0),
                                "shape": shape,
                                "cls": cls_id,
                            }
                            detections.append(detection)

                # Apply post-processing if configured
                if detections:
                    detections = self._apply_post_processing(
                        detections, tuple(shape)
                    )  # Convert to TrackLab format
                for detection in detections:
                    category_id = (
                        1 if detection["cls"] == 0 else 2
                    )  # 1 for person, 2 for ball
                    detections_out.append(
                        pd.Series(
                            dict(
                                image_id=detection["image_id"],
                                bbox_ltwh=ltrb_to_ltwh(
                                    detection["bbox"], detection["shape"]
                                ),
                                bbox_conf=detection["conf"],
                                video_id=detection["video_id"],
                                category_id=category_id,
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1

        return detections_out

    def _apply_post_processing(
        self, detections: List[Dict], image_shape: tuple
    ) -> List[Dict]:
        """
        Apply post-processing to detections including soft NMS and size filtering.

        Args:
            detections: List of detection dictionaries
            image_shape: (width, height) of the image

        Returns:
            Filtered list of detections
        """
        if not detections:
            return detections

        # Convert detections to format expected by soft_nms
        if self.enable_soft_nms:
            # Prepare data for soft_nms: [x1, y1, x2, y2, score]
            boxes = []
            scores = []
            for det in detections:
                boxes.append(det["bbox"])
                scores.append(det["conf"])

            boxes = np.array(boxes)
            scores = np.array(scores)

            # Apply soft NMS
            if soft_nms is not None:
                try:
                    # soft_nms returns (dets, indices) where dets has shape (N, 5) with [x1, y1, x2, y2, score]
                    dets, indices = soft_nms(
                        boxes=boxes.astype(np.float32),  # Ensure float32 type
                        scores=scores.astype(np.float32),  # Ensure float32 type
                        iou_threshold=0.5,  # Standard IoU threshold
                        sigma=0.5,  # Soft NMS sigma parameter
                        min_score=0.001,  # Minimum score to keep
                    )

                    # Update detections with soft NMS results
                    # dets[:, :4] are the boxes, dets[:, 4] are the updated scores
                    filtered_detections = []
                    for i, (det_result, orig_idx) in enumerate(zip(dets, indices)):
                        box = det_result[:4]  # Extract box coordinates
                        score = det_result[4]  # Extract updated score from 5th column
                        if (
                            score > self.cfg.min_confidence
                        ):  # Re-apply confidence threshold
                            det = detections[orig_idx].copy()  # Use original index
                            det["bbox"] = box
                            det["conf"] = score
                            filtered_detections.append(det)

                    detections = filtered_detections

                except Exception as e:
                    log.warning(f"Soft NMS failed, using original detections: {e}")
            else:
                log.warning("Soft NMS not available, using original detections")

        # Apply size filtering
        if self.size_filter_min_area_ratio > 0:
            image_area = image_shape[0] * image_shape[1]  # width * height
            min_area = image_area * self.size_filter_min_area_ratio

            filtered_detections = []
            for det in detections:
                bbox = det["bbox"]
                bbox_area = (bbox[2] - bbox[0]) * (
                    bbox[3] - bbox[1]
                )  # (x2-x1) * (y2-y1)

                if bbox_area >= min_area:
                    filtered_detections.append(det)

            log.debug(
                f"Size filtering: {len(detections)} -> {len(filtered_detections)} detections"
            )
            detections = filtered_detections

        return detections

    def train(
        self,
        tracking_dataset: TrackingDataset,
        pipeline: Pipeline,
        dataset_config: dict,
    ) -> None:
        """Train the YOLO model using the TrackingDataset.

        Args:
            tracking_dataset: The TrackingDataset containing train/valid/test sets
            pipeline: The TrackLab pipeline
            evaluator: The evaluator for validation
            dataset_config: Configuration for the dataset
        """
        from pathlib import Path

        log.info("Starting YOLO training with TrackingDataset...")

        # Get training configuration optimized for YOLO11x
        train_cfg = getattr(self.cfg, "training", {})
        epochs = train_cfg.get("epochs", 50)
        batch_size = train_cfg.get("batch_size", 16)  # Smaller batch for YOLO11x
        img_size = train_cfg.get("img_size", 640)

        # Determine dataset directory: use data_path if provided, otherwise use persistent dir
        data_path = train_cfg.get("data_path")
        if data_path:
            dataset_base_path = Path(data_path)
        else:
            # Use persistent directory in the project data folder
            dataset_base_path = Path.cwd() / "data" / "yolo_training_dataset"

        dataset_base_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Using dataset directory: {dataset_base_path}")

        # Use the persistent directory directly
        yolo_dataset_path = dataset_base_path / "dataset"
        yolo_dataset_path.mkdir(exist_ok=True)

        # Convert TrackingDataset to YOLO format
        yolo_data_yaml = self._prepare_yolo_dataset(
            tracking_dataset,
            yolo_dataset_path,
            dataset_config,
            tracking_dataset.dataset_path,
        )

        # Train the model
        self._run_yolo_training(yolo_data_yaml, epochs, batch_size, img_size)

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
            "names": {0: "person", 1: "ball"},  # Person and ball classes
            "nc": 2,  # Number of classes
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

        import shutil
        import os
        from PIL import Image

        images_dir = output_path / "images" / split_name
        labels_dir = output_path / "labels" / split_name

        processed_count = 0
        total_detections = 0
        skipped_images = 0

        # Group detections by image
        if tracking_set.detections_gt is None or tracking_set.detections_gt.empty:
            log.warning(f"No ground truth detections found for {split_name} split")
            return {"processed_images": 0, "total_detections": 0, "skipped_images": 0}

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

                # Load image (ensure file path is a string to satisfy type checkers)
                file_path_str = str(image_meta.file_path)
                image_path = Path(dataset_path) / file_path_str
                if not image_path.exists():
                    log.warning(f"Image not found: {image_path}")
                    skipped_images += 1
                    continue

                # Get image dimensions first
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception as e:
                    log.warning(
                        f"Could not read image dimensions for {image_path}: {e}"
                    )
                    skipped_images += 1
                    continue

                # Create symlink to image instead of copying (much faster for large datasets)
                suffix = image_path.suffix or ".jpg"
                yolo_image_path = images_dir / f"{image_id}{suffix}"

                # Remove existing file/symlink if it exists
                if yolo_image_path.exists() or yolo_image_path.is_symlink():
                    try:
                        yolo_image_path.unlink(missing_ok=True)
                    except Exception:
                        pass  # Ignore cleanup errors

                try:
                    # Use relative symlink if possible, absolute otherwise
                    try:
                        # Try relative symlink first
                        rel_path = os.path.relpath(image_path, images_dir)
                        yolo_image_path.symlink_to(rel_path)
                    except (OSError, ValueError):
                        # Fall back to absolute symlink
                        yolo_image_path.symlink_to(image_path)
                except Exception as e:
                    log.warning(
                        f"Could not create symlink for {image_path} to {yolo_image_path}: {e}"
                    )
                    skipped_images += 1
                    continue

                # Create label file
                label_path = labels_dir / f"{image_id}.txt"
                labels_written = 0
                with open(label_path, "w") as f:
                    # Process detections
                    for _, detection in detections.iterrows():
                        # Filter and merge categories to person only
                        category_id = self._map_category_to_person(
                            detection, tracking_set
                        )

                        if category_id in [0, 1]:  # person or ball class
                            # Convert bbox to YOLO format (normalized)
                            bbox = detection.bbox_ltwh
                            if isinstance(bbox, np.ndarray):
                                left, top, w, h = bbox
                            else:
                                left, top, w, h = bbox

                            # Validate bbox values - comprehensive check
                            if (
                                any(v < 0 for v in [left, top, w, h])
                                or w <= 0
                                or h <= 0
                                or left + w > width
                                or top + h > height
                            ):
                                log.warning(
                                    f"Invalid bbox for image {image_id}: {bbox} (image: {width}x{height})"
                                )
                                continue

                            x_center = (left + w / 2) / width
                            y_center = (top + h / 2) / height
                            w_norm = w / width
                            h_norm = h / height

                            # Ensure values are within [0, 1]
                            x_center = max(0.0, min(1.0, x_center))
                            y_center = max(0.0, min(1.0, y_center))
                            w_norm = max(0.0, min(1.0, w_norm))
                            h_norm = max(0.0, min(1.0, h_norm))

                            f.write(
                                f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                            )
                            labels_written += 1
                            total_detections += 1

                # Clean up empty label files
                if labels_written == 0:
                    try:
                        label_path.unlink(missing_ok=True)
                    except Exception:
                        pass  # Ignore cleanup errors

                processed_count += 1

            except Exception as e:
                log.warning(f"Error processing image {image_id}: {e}")
                skipped_images += 1
                continue

        log.info(
            f"✅ Processed {processed_count}/{total_images} images for {split_name} split with {total_detections} detections"
        )
        if skipped_images > 0:
            log.warning(f"⚠️ Skipped {skipped_images} images due to errors")

        return {
            "processed_images": processed_count,
            "total_detections": total_detections,
            "skipped_images": skipped_images,
        }

    def _map_category_to_person(
        self, detection_row: pd.Series, tracking_set: object
    ) -> int:
        """Map any category to person class (0) for bbox_detector training.

        Args:
            detection_row: Row from detections DataFrame
            tracking_set: The TrackingSet containing category information

        Returns:
            Mapped category ID (0 for person, -1 to skip)
        """
        # For bbox_detector, we want to be permissive and accept most detections as person class
        # since we're training a general person detector

        # Check if we have role information - accept all human roles
        if "role" in detection_row:
            role = detection_row["role"]
            if role in ["player", "goalkeeper", "referee", "person", "human"]:
                return 0  # YOLO person class
            elif role in ["ball", "football", "soccer_ball"]:
                return 1  # YOLO ball class
            else:
                # For unknown roles, assume they might be person-related
                return 0

        # Fallback to category name checking - be more permissive
        if "category" in detection_row:
            category = str(detection_row["category"]).lower()
            # Skip only clearly non-person/ball categories
            if any(keyword in category for keyword in ["goal", "field", "line"]):
                return -1
            elif "ball" in category:
                return 1
            else:
                # Accept all other categories as potentially person-related
                return 0

        # Check category_id - be very permissive for bbox detection
        category_id = detection_row.get("category_id", -1)

        # Accept any positive category_id as person-related for bbox detection training
        if category_id >= 0:
            return 0
        else:
            # If no clear category information, assume it's a person detection
            return 0

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
        from ultralytics import YOLO, settings

        # Set Ultralytics cache directory to avoid downloading to root
        cache_dir = Path(self.cfg.path_to_checkpoint).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings.update(weights_dir=str(cache_dir))

        # Change working directory to cache_dir for downloads
        import os

        original_cwd = os.getcwd()
        os.chdir(cache_dir)

        log.info(
            f"Starting YOLO training with {epochs} epochs, batch size {batch_size}"
        )

        # Get training configuration from cfg
        train_cfg = getattr(self.cfg, "training", {})

        # Set up training arguments

        # Set up training arguments optimized for YOLO11x large model
        train_args = {
            "data": str(data_yaml_path),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "device": self.device,
            "workers": train_cfg.get("workers", 12),  # Moderate workers for stability
            "optimizer": train_cfg.get("optimizer", "AdamW"),
            "lr0": train_cfg.get("lr0", 0.0005),  # Conservative LR for large model
            "lrf": train_cfg.get("lrf", 0.1),  # Higher final LR ratio
            "momentum": train_cfg.get("momentum", 0.9),  # Standard momentum
            "weight_decay": train_cfg.get("weight_decay", 0.0005),
            "warmup_epochs": train_cfg.get("warmup_epochs", 5),  # Longer warmup
            "warmup_momentum": train_cfg.get("warmup_momentum", 0.8),
            "warmup_bias_lr": train_cfg.get("warmup_bias_lr", 0.05),
            "freeze": train_cfg.get("freeze", 24),  # Freeze more layers for stability
            "amp": train_cfg.get("amp", True),  # Essential for large models
            "cache": train_cfg.get("cache", True),  # Critical for speed
            "val": train_cfg.get("val", True),
            "save_period": train_cfg.get("save_period", 5),  # Save more frequently
            "patience": train_cfg.get("patience", 15),  # More patience for large model
            "plots": train_cfg.get("plots", False),
            "verbose": train_cfg.get("verbose", False),
            "resume": train_cfg.get("resume", False),
            "cos_lr": True,  # Cosine LR for better convergence
            "close_mosaic": 15,  # Close mosaic later for large model
            "overlap_mask": False,
            "mask_ratio": 1,
            "dropout": 0.1,  # Light dropout for regularization
            "nbs": 64,  # Nominal batch size
            "hsv_h": 0.01,  # Reduced augmentations for stability
            "hsv_s": 0.6,
            "hsv_v": 0.3,
            "degrees": 0.0,  # No rotation for person detection
            "translate": 0.05,  # Reduced translation
            "scale": 0.3,  # Reduced scale
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,  # Keep horizontal flip
            "mosaic": 0.8,  # Reduced mosaic probability
            "mixup": 0.0,  # No mixup for stability
            "copy_paste": 0.0,
        }

        # YOLO11x specific: Add gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()  # Reduce memory usage

        # Train the model
        log.info("🚀 Starting YOLO11x model training...")
        results = self.model.train(**train_args)
        log.info("✅ YOLO11x training completed!")

        # Restore original working directory
        os.chdir(original_cwd)

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
