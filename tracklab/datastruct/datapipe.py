"""Data pipeline for TrackLab engine processing."""

from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from torch.utils.data import Dataset

from tracklab.pipeline.module import Module
from tracklab.utils.cv2 import cv2_load_image


class EngineDatapipe(Dataset):
    """Dataset class for feeding data to pipeline modules during tracking.

    This datapipe handles data loading and preprocessing for different module
    levels (detection-level and image-level processing).
    """

    def __init__(self, model: Module) -> None:
        """Initialize the datapipe with a pipeline module.

        Args:
            model: The pipeline module that will process the data.
        """
        self.model = model
        self.image_filepaths: Optional[Dict[Any, str]] = None
        self.img_metadatas: Optional[pd.DataFrame] = None
        self.detections: Optional[pd.DataFrame] = None

    def update(
        self,
        image_filepaths: Dict[Any, str],
        img_metadatas: pd.DataFrame,
        detections: Optional[pd.DataFrame],
    ) -> None:
        """Update the datapipe with new data for processing.

        Args:
            image_filepaths: Mapping from image IDs to file paths.
            img_metadatas: DataFrame containing image metadata.
            detections: DataFrame containing detection data (can be None).
        """
        # Clean up previous data
        del self.img_metadatas
        del self.detections

        self.image_filepaths = image_filepaths
        self.img_metadatas = img_metadatas
        self.detections = detections

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            Number of items based on the module's processing level.

        Raises:
            ValueError: If the module level is not supported.
        """
        if self.model.level == "detection":
            return len(self.detections) if self.detections is not None else 0
        elif self.model.level == "image":
            return len(self.img_metadatas) if self.img_metadatas is not None else 0
        else:
            raise ValueError(
                f"You should provide the appropriate level for your module, "
                f"not '{self.model.level}'"
            )

    def __getitem__(self, idx: int) -> Tuple[Union[int, str], Any]:
        """Get a data sample for processing.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the sample ID and preprocessed data.

        Raises:
            ValueError: If the module level is not supported.
        """
        if self.model.level == "detection":
            if (
                self.detections is None
                or self.img_metadatas is None
                or self.image_filepaths is None
            ):
                raise ValueError(
                    "Detections, image metadata, and file paths must be set"
                )

            detection = self.detections.iloc[idx]
            metadata = self.img_metadatas.loc[detection.image_id]
            image = cv2_load_image(self.image_filepaths[metadata.name])

            sample = (
                detection.name,
                self.model.preprocess(  # type: ignore
                    image=image, detection=detection, metadata=metadata
                ),
            )
            return sample

        elif self.model.level == "image":
            if self.img_metadatas is None or self.image_filepaths is None:
                raise ValueError("Image metadata and file paths must be set")

            metadata = self.img_metadatas.iloc[idx]

            # Filter detections for this image if available
            if self.detections is not None and len(self.detections) > 0:
                detections = self.detections[self.detections.image_id == metadata.name]
            else:
                detections = self.detections

            image = cv2_load_image(self.image_filepaths[metadata.name])

            sample = (
                self.img_metadatas.index[idx],
                self.model.preprocess(  # type: ignore
                    image=image, detections=detections, metadata=metadata
                ),
            )
            return sample

        else:
            raise ValueError("Please provide appropriate level.")
