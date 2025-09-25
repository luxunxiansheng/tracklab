import json
import os
from pathlib import Path
from typing import Optional
import logging

import pandas as pd

from .base_exporter import BaseExporter

log = logging.getLogger(__name__)


class GSExporter(BaseExporter):
    """
    Exporter for SoccerNet Game State format.
    Exports tracking data in JSON format similar to COCO.
    """

    def export(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        save_path: str,
        bbox_column: str = "bbox_ltwh",
        save_classes: bool = False,
        is_ground_truth: bool = False,
        **kwargs,
    ) -> None:
        """
        Export detections in GS JSON format.

        Args:
            detections: DataFrame containing detection data
            image_metadatas: DataFrame containing image metadata
            video_metadatas: DataFrame containing video metadata
            save_path: Path where to save the exported data
            bbox_column: Column name containing bounding box data
            save_classes: Whether to include class information
            is_ground_truth: Whether this is ground truth data
            **kwargs: Additional parameters
        """
        gs_data = self._gs_encoding(
            detections, image_metadatas, video_metadatas, bbox_column, save_classes
        )

        save_path_obj = self._ensure_directory(save_path)

        for video_id, video in video_metadatas.iterrows():
            file_path = save_path_obj / f"{video['name']}.json"
            video_data = gs_data[video_id]
            with open(file_path, "w") as f:
                json.dump(video_data, f, indent=2)

    def _gs_encoding(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        bbox_column: str,
        save_classes: bool,
    ) -> dict:
        """
        Convert detections to GS JSON format.
        """
        # Merge detections with image metadata
        image_metadatas = image_metadatas.copy()
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            detections.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
            suffixes=("", "_y"),
        )

        # Drop rows with missing required fields
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "frame",
                "track_id",
                bbox_column,
            ],
            how="any",
            inplace=True,
        )

        if len_before_drop != len(df):
            log.warning(f"Dropped {len_before_drop - len(df)} rows with NA values")

        # Convert track_id to int
        df["track_id"] = df["track_id"].astype(int)

        # Group by video
        video_data = {}
        for video_id, video_df in df.groupby("video_id"):
            video = video_metadatas.loc[video_id]
            images = []
            annotations = []

            for _, row in video_df.iterrows():
                # Add image if not already added
                image_id = f"{video['name']}_{int(row['frame']):06d}"
                if not any(img["image_id"] == image_id for img in images):
                    images.append(
                        {
                            "image_id": image_id,
                            "has_labeled_pitch": True,  # Assume labeled
                            "has_labeled_camera": True,
                            "has_labeled_person": True,
                        }
                    )

                # Add annotation
                bbox = row[bbox_column]
                pitch_bbox = row.get(
                    "bbox_pitch_ltwh", bbox
                )  # Use pitch bbox if available, else image bbox
                annotation = {
                    "image_id": image_id,
                    "category_id": 1,  # person
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],
                    "bbox_pitch": {
                        "x_bottom_left": float(pitch_bbox[0]),
                        "y_bottom_left": float(pitch_bbox[1]),
                        "x_top_right": float(pitch_bbox[0] + pitch_bbox[2]),
                        "y_top_right": float(pitch_bbox[1] + pitch_bbox[3]),
                    },
                    "track_id": int(row["track_id"]),
                    "confidence": float(row.get("bbox_conf", 1.0)),
                    "supercategory": "object",
                    "attributes": {"role": "player", "team": None, "jersey": None},
                }
                # Compute additional bbox_pitch fields
                bbox_pitch = annotation["bbox_pitch"]
                bbox_pitch["x_bottom_middle"] = (
                    bbox_pitch["x_bottom_left"] + bbox_pitch["x_top_right"]
                ) / 2
                bbox_pitch["y_bottom_middle"] = (
                    bbox_pitch["y_bottom_left"]
                    + (bbox_pitch["y_top_right"] - bbox_pitch["y_bottom_left"]) * 0.9
                )
                bbox_pitch["x_bottom_right"] = bbox_pitch["x_top_right"]
                bbox_pitch["y_bottom_right"] = bbox_pitch["y_bottom_middle"]
                if save_classes and "category_id" in row:
                    annotation["category_id"] = int(row["category_id"])
                annotations.append(annotation)

            # Sort images by frame
            def get_frame_number(image):
                return int(image["image_id"].split("_")[-1])

            images = sorted(images, key=get_frame_number)

            video_data[video_id] = {
                "categories": [{"id": 1, "name": "person"}],
                "images": images,
                "predictions": annotations,
            }

        return video_data
