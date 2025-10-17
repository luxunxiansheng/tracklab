"""SoccerNet Game State format exporter for TrackLab tracking data."""

import json
import zipfile
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from .base_exporter import BaseExporter

log = logging.getLogger(__name__)


def transform_bbox_image(bbox: Any) -> Optional[Dict[str, float]]:
    """Transform bbox format for SoccerNet GS.

    Args:
        bbox: Bounding box in LTWH format (left, top, width, height).

    Returns:
        Dictionary with transformed bbox coordinates, or None if invalid.
    """
    try:
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
            # Convert from ltwh (left, top, width, height) to center-based format
            # Note: Using "x", "y" keys instead of "x_center", "y_center" to match trackeval library expectations
            left, top, width, height = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            return {
                "x": left,
                "y": top,
                "w": width,
                "h": height,
                "x_center": left + width / 2,
                "y_center": top + height / 2,
            }
        else:
            # Invalid bbox format
            return None
    except (TypeError, IndexError, ValueError):
        return None


class GSExporter(BaseExporter):
    """Exporter for SoccerNet Game State format.

    Exports tracking data in JSON format matching SoccerNet GameState requirements.
    This format is used for evaluating tracking performance on soccer videos with
    additional metadata like player roles, jersey numbers, and team information.
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
        save_zip: bool = False,
        **kwargs,
    ) -> None:
        """
        Export detections in SoccerNet Game State JSON format.

        Args:
            detections: DataFrame containing detection data
            image_metadatas: DataFrame containing image metadata
            video_metadatas: DataFrame containing video metadata
            save_path: Path where to save the exported data
            bbox_column: Column name containing bounding box data
            save_classes: Whether to include class information
            is_ground_truth: Whether this is ground truth data
            save_zip: Whether to create a zip file of the results
            **kwargs: Additional parameters
        """
        if is_ground_truth:
            return

        save_path_obj = self._ensure_directory(save_path)

        # Process detections with SoccerNet encoding
        detections_encoded = self._soccernet_encoding(
            detections.copy(), supercategory="object"
        )
        camera_metadata = self._soccernet_encoding(
            image_metadatas.copy(), supercategory="camera"
        )
        pitch_metadata = self._soccernet_encoding(
            image_metadatas.copy(), supercategory="pitch"
        )

        predictions = pd.concat(
            [detections_encoded, camera_metadata, pitch_metadata], ignore_index=True
        )

        zf_save_path = save_path_obj.parent.parent / f"{save_path_obj.parent.name}.zip"

        for video_id, video in video_metadatas.iterrows():
            file_path = save_path_obj / f"{video['name']}.json"
            video_predictions_df = predictions[
                predictions["video_id"] == str(video_id)
            ].copy()

            if not video_predictions_df.empty:
                video_predictions_df["image_id_int"] = video_predictions_df[
                    "image_id"
                ].astype(int)
                video_predictions_df.sort_values(
                    by=["image_id_int", "id"], inplace=True
                )
                video_predictions_df.drop(columns=["image_id_int"], inplace=True)
                video_predictions = [
                    {
                        k: int(v) if k == "track_id" else v
                        for k, v in m.items()
                        if np.all(pd.notna(v))
                    }
                    for m in video_predictions_df.to_dict(orient="records")
                ]

                with file_path.open("w") as fp:
                    json.dump({"predictions": video_predictions}, fp, indent=2)

                if save_zip:
                    with zipfile.ZipFile(
                        zf_save_path, "a", compression=zipfile.ZIP_DEFLATED
                    ) as zf:
                        zf.write(
                            file_path, arcname=f"{save_path_obj.name}/{file_path.name}"
                        )

    def _soccernet_encoding(
        self, dataframe: pd.DataFrame, supercategory: str
    ) -> pd.DataFrame:
        """Convert dataframe to SoccerNet Game State encoding format.

        Args:
            dataframe: Input dataframe to encode.
            supercategory: Type of data ("object", "camera", or "pitch").

        Returns:
            Encoded dataframe in SoccerNet format.
        """
        dataframe["supercategory"] = supercategory
        dataframe = dataframe.replace({np.nan: None})

        if supercategory == "object":
            # Remove detections that don't have mandatory columns
            # Detections with no track_id will therefore be removed and not count as FP at evaluation
            # Exception: Balls (category_id=2) are kept even without track_id
            mandatory_columns: List[str] = []
            if "bbox_ltwh" in dataframe.columns:
                mandatory_columns.append("bbox_ltwh")
            if "bbox_pitch" in dataframe.columns:
                mandatory_columns.append("bbox_pitch")

            # For track_id: keep balls even if untracked
            if "track_id" in dataframe.columns:
                # Create mask for rows to keep: either has track_id OR is a ball
                if "category_id" in dataframe.columns:
                    keep_mask = dataframe["track_id"].notna() | (
                        dataframe["category_id"] == 2
                    )
                    # For balls without track_id, assign a unique negative ID
                    ball_no_track = (dataframe["category_id"] == 2) & dataframe[
                        "track_id"
                    ].isna()
                    if ball_no_track.any():
                        # Assign unique negative track IDs to untracked balls
                        ball_indices = dataframe[ball_no_track].index
                        dataframe.loc[ball_indices, "track_id"] = -(
                            ball_indices.values + 1
                        )
                    # Keep only rows in keep_mask
                    dataframe = dataframe[keep_mask]
                else:
                    mandatory_columns.append("track_id")

            if mandatory_columns:
                dataframe.dropna(
                    subset=mandatory_columns,
                    how="any",
                    inplace=True,
                )

            # Add track_id if missing (for detection-only runs)
            if "track_id" not in dataframe.columns:
                dataframe["track_id"] = -1  # Default for detections without tracking

            # Rename columns if they exist
            rename_dict: Dict[str, str] = {}
            if "bbox_ltwh" in dataframe.columns:
                rename_dict["bbox_ltwh"] = "bbox_image"
            if "jersey_number" in dataframe.columns:
                rename_dict["jersey_number"] = "jersey"
            dataframe = dataframe.rename(columns=rename_dict)

            dataframe["track_id"] = dataframe["track_id"]
            dataframe["attributes"] = [
                {
                    "role": x.get("role")
                    or x.get("role_detection")  # Use role_detection if role is None/NaN
                    or (
                        "ball"
                        if x.get("category_id") == 2
                        else "person" if x.get("category_id") == 1 else None
                    ),
                    "jersey": x.get("jersey"),
                    "team": x.get("team"),
                }
                for n, x in dataframe.iterrows()
            ]
            dataframe["id"] = dataframe.index

            # Keep only relevant columns that exist
            columns_to_keep = [
                "id",
                "image_id",
                "video_id",
                "track_id",
                "supercategory",
                "category_id",
                "attributes",
            ]
            if "bbox_image" in dataframe.columns:
                columns_to_keep.append("bbox_image")
            if "bbox_pitch" in dataframe.columns:
                columns_to_keep.append("bbox_pitch")
            dataframe = dataframe[dataframe.columns.intersection(columns_to_keep)]

            dataframe = dataframe.reset_index(drop=True)

            if "bbox_image" in dataframe.columns:
                # Transform bbox format
                for idx in dataframe.index:
                    bbox = dataframe.at[idx, "bbox_image"]
                    transformed = transform_bbox_image(bbox)
                    if transformed is not None:
                        dataframe.at[idx, "bbox_image"] = transformed  # type: ignore
                    else:
                        # Invalid bbox, mark for removal
                        dataframe.at[idx, "bbox_image"] = None

                # Remove rows with invalid bbox_image
                dataframe.dropna(subset=["bbox_image"], inplace=True)

        elif supercategory == "camera":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 6
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "01")
            dataframe = dataframe[
                dataframe.columns.intersection(
                    [
                        "id",
                        "image_id",
                        "video_id",
                        "supercategory",
                        "category_id",
                        "parameters",
                        "relative_mean_reproj",
                        "accuracy@5",
                    ]
                )
            ]

        elif supercategory == "pitch":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 5
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "00")
            dataframe = dataframe[
                dataframe.columns.intersection(
                    [
                        "id",
                        "image_id",
                        "video_id",
                        "supercategory",
                        "category_id",
                        "lines",
                    ]
                )
            ]

        # Convert IDs to string format
        dataframe["video_id"] = dataframe["video_id"].apply(str)
        dataframe["image_id"] = dataframe["image_id"].apply(str)
        dataframe["id"] = dataframe["id"].apply(str)

        # Convert numpy arrays to lists for JSON serialization
        dataframe = dataframe.map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

        return dataframe
