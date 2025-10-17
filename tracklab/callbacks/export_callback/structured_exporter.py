"""Structured data format exporter for TrackLab tracking data.

This exporter creates a structured JSON format with frame-by-frame tracking data,
including player positions, teams, possession information, and time metadata.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .base_exporter import BaseExporter

log = logging.getLogger(__name__)


class StructuredExporter(BaseExporter):
    """Exporter for structured frame-by-frame tracking data.

    Exports tracking data in a structured JSON format where each frame contains:
    - possession: Information about ball possession
    - frame: Frame number
    - data: List of tracked objects with their properties
    - period: Game period (if available)
    - time: Timestamp in the video (if available)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        include_pitch_coords: bool = True,
        fps: float = 25.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the structured exporter.

        Args:
            config: Configuration dictionary for the exporter.
            include_pitch_coords: Whether to include pitch coordinates (x, y).
            fps: Frames per second for time calculation.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config, **kwargs)
        self.include_pitch_coords = (
            config.get("include_pitch_coords", include_pitch_coords)
            if config
            else include_pitch_coords
        )
        self.fps = config.get("fps", fps) if config else fps

    def export(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        save_path: str,
        bbox_column: str = "bbox_ltwh",
        save_classes: bool = False,
        is_ground_truth: bool = False,
        **kwargs: Any,
    ) -> None:
        """Export detections in structured JSON format.

        Args:
            detections: DataFrame containing detection and tracking data.
            image_metadatas: DataFrame containing image metadata.
            video_metadatas: DataFrame containing video metadata.
            save_path: Path where to save the exported data.
            bbox_column: Column name containing bounding box data.
            save_classes: Whether to include class information in export.
            is_ground_truth: Whether this is ground truth data.
            **kwargs: Additional format-specific parameters.
        """
        if is_ground_truth:
            return

        save_path_obj = self._ensure_directory(save_path)

        # Process each video
        for video_id, video in video_metadatas.iterrows():
            file_path = save_path_obj / f"{video['name']}_structured.json"

            # Get frames for this video
            video_image_ids = image_metadatas[
                image_metadatas["video_id"] == video_id
            ].index

            # Filter detections for this video
            video_detections = detections[
                detections["image_id"].isin(video_image_ids)
            ].copy()

            # Build structured data
            structured_data = self._build_structured_data(
                video_detections, image_metadatas, video_id
            )

            # Save to JSON
            with file_path.open("w") as fp:
                json.dump(structured_data, fp, indent=2)

            log.info(f"Exported structured data to {file_path}")

    def _build_structured_data(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_id: Any,
    ) -> List[Dict[str, Any]]:
        """Build the structured data format from detections.

        Args:
            detections: Detection data for the video.
            image_metadatas: Image metadata for the video.
            video_id: ID of the video being processed.

        Returns:
            List of frame dictionaries in structured format.
        """
        structured_frames = []

        # Get all image IDs for this video
        video_images = image_metadatas[image_metadatas["video_id"] == video_id].copy()

        # Sort by frame number if available, otherwise by index
        if "frame_id" in video_images.columns:
            video_images = video_images.sort_values("frame_id")
        else:
            video_images = video_images.sort_index()

        # Process each frame
        for image_id, image_info in video_images.iterrows():
            frame_detections = detections[detections["image_id"] == image_id]

            # Get frame number
            frame_num = int(image_info.get("frame_id", image_id))

            # Calculate time stamp (format: "M:SS.FF")
            time_str = self._calculate_time_str(frame_num, self.fps)

            # Get period if available
            period = int(image_info.get("period", 1)) if "period" in image_info else 1

            # Build detection data for this frame
            frame_data = []
            possession_info = {"trackable_object": None, "group": None}

            for _, det in frame_detections.iterrows():
                det_dict = self._build_detection_dict(det)
                if det_dict:
                    frame_data.append(det_dict)

                    # Update possession if this is the ball
                    if det_dict.get("type") == "ball":
                        possession_info["trackable_object"] = det_dict.get(
                            "trackable_object"
                        )

            # Create frame entry
            frame_entry = {
                "possession": possession_info,
                "frame": frame_num,
                "data": frame_data,
                "period": period,
                "time": time_str,
            }

            structured_frames.append(frame_entry)

        return structured_frames

    def _build_detection_dict(self, detection: pd.Series) -> Optional[Dict[str, Any]]:
        """Build a detection dictionary from a detection series.

        Args:
            detection: Single detection as a pandas Series.

        Returns:
            Dictionary with detection information, or None if invalid.
        """
        det_dict = {}

        # Add pitch coordinates if available and enabled
        if self.include_pitch_coords and "bbox_pitch" in detection:
            bbox_pitch = detection["bbox_pitch"]

            # Handle dictionary format (e.g., {'x_bottom_middle': ..., 'y_bottom_middle': ...})
            if isinstance(bbox_pitch, dict):
                try:
                    # Prefer bottom_middle, then bottom_left, then center coordinates
                    if (
                        "x_bottom_middle" in bbox_pitch
                        and "y_bottom_middle" in bbox_pitch
                    ):
                        det_dict["x"] = float(bbox_pitch["x_bottom_middle"])
                        det_dict["y"] = float(bbox_pitch["y_bottom_middle"])
                    elif (
                        "x_bottom_left" in bbox_pitch and "y_bottom_left" in bbox_pitch
                    ):
                        det_dict["x"] = float(bbox_pitch["x_bottom_left"])
                        det_dict["y"] = float(bbox_pitch["y_bottom_left"])
                    elif "x" in bbox_pitch and "y" in bbox_pitch:
                        det_dict["x"] = float(bbox_pitch["x"])
                        det_dict["y"] = float(bbox_pitch["y"])
                except (TypeError, ValueError, KeyError):
                    pass  # Skip if conversion fails

            # Handle list/tuple/array format [x, y]
            elif (
                isinstance(bbox_pitch, (list, tuple, np.ndarray))
                and len(bbox_pitch) >= 2
            ):
                try:
                    det_dict["x"] = (
                        float(bbox_pitch[0]) if not pd.isna(bbox_pitch[0]) else 0.0
                    )
                    det_dict["y"] = (
                        float(bbox_pitch[1]) if not pd.isna(bbox_pitch[1]) else 0.0
                    )
                except (TypeError, IndexError, ValueError):
                    pass  # Skip if conversion fails

        # Add trackable_object (unique ID for this detection)
        # Use the detection's index or a unique identifier
        if detection.name is not None:
            det_dict["trackable_object"] = int(detection.name)  # type: ignore
        else:
            det_dict["trackable_object"] = 0

        # Add track_id (persistent ID across frames)
        if "track_id" in detection and pd.notna(detection["track_id"]):
            det_dict["track_id"] = int(detection["track_id"])
        else:
            # No tracking, use negative ID
            det_dict["track_id"] = -1

        # Determine type based on category_id
        category_id = detection.get("category_id", 1)
        if category_id == 0 or detection.get("role") == "ball":  # Ball is category_id 0
            det_dict["type"] = "ball"
            det_dict["team"] = None
        else:
            det_dict["type"] = "player"
            # Add team information
            team = detection.get("team")
            if pd.notna(team):
                det_dict["team"] = str(team)
            else:
                det_dict["team"] = None

        return det_dict

    def _calculate_time_str(self, frame_num: int, fps: float) -> str:
        """Calculate time string from frame number.

        Args:
            frame_num: Frame number.
            fps: Frames per second.

        Returns:
            Time string in format "M:SS.FF" (minutes:seconds.centiseconds).
        """
        total_seconds = frame_num / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        centiseconds = int((total_seconds % 1) * 100)

        return f"{minutes}:{seconds:02d}.{centiseconds:02d}"
