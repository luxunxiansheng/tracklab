"""PoseTrack18 dataset for TrackLab pose tracking evaluation."""

from typing import Any

from .posetrack21 import PoseTrack21


class PoseTrack18(PoseTrack21):
    """PoseTrack18 dataset class for pose tracking evaluation.

    This class inherits from PoseTrack21 and configures it for PoseTrack18
    dataset format and evaluation protocols.
    """

    def __init__(
        self, dataset_path: str, annotation_path: str, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize PoseTrack18 dataset.

        Args:
            dataset_path: Path to the dataset directory.
            annotation_path: Path to the annotation files.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            dataset_path, annotation_path, posetrack_version=18, *args, **kwargs
        )
