import logging
from typing import Dict, List, Optional, Union

from .common import MOT

log = logging.getLogger(__name__)

categories_list: List[Dict[str, Union[int, str]]] = [
    {"id": 1, "name": "bee"},
]


class Bee24(MOT):
    """Bee24 dataset class."""

    name = "Bee24"
    nickname = "bee24"

    def __init__(
        self,
        dataset_path: str,
        nvid: int = -1,
        nframes: int = -1,
        vids_dict: Optional[Dict[str, List[str]]] = None,
        public_dets_subpath: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize Bee24 dataset.

        Args:
            dataset_path: Path to the dataset.
            nvid: Number of videos to use.
            nframes: Number of frames per video.
            vids_dict: Dictionary of video IDs per split.
            public_dets_subpath: Subpath for public detections.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        log.info(f"Loading Bee24 dataset from {dataset_path}.")
        super().__init__(
            dataset_path,
            categories_list,
            nvid,
            nframes,
            vids_dict,
            public_dets_subpath,
            *args,
            **kwargs,
        )
