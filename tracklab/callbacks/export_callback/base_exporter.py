"""Base exporter classes for TrackLab data export functionality."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Base class for exporting tracking data in various formats.

    This abstract base class defines the interface for all data exporters
    in the TrackLab system, providing a standardized way to export detection
    and tracking results to different file formats.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize the base exporter.

        Args:
            config: Configuration dictionary for the exporter.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or kwargs

    @abstractmethod
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
        """Export detections to the specified format.

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
        pass

    def _ensure_directory(self, path: str) -> Path:
        """Ensure the directory exists and return Path object.

        Args:
            path: Directory path to ensure exists.

        Returns:
            Path object for the ensured directory.
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
