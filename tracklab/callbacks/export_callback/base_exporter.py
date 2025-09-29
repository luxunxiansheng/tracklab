import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


class BaseExporter(ABC):
    """
    Base class for exporting tracking data in various formats.
    """

    def __init__(self, config=None):
        self.config = config or {}

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
        **kwargs,
    ) -> None:
        """
        Export detections to the specified format.

        Args:
            detections: DataFrame containing detection data
            image_metadatas: DataFrame containing image metadata
            video_metadatas: DataFrame containing video metadata
            save_path: Path where to save the exported data
            bbox_column: Column name containing bounding box data
            save_classes: Whether to include class information
            is_ground_truth: Whether this is ground truth data
            **kwargs: Additional format-specific parameters
        """
        pass

    def _ensure_directory(self, path: str) -> Path:
        """Ensure the directory exists and return Path object."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
