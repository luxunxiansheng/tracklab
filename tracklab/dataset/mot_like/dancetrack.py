import logging
from typing import Dict, List, Optional, Union

from .common import MOT

log = logging.getLogger(__name__)

categories_list: List[Dict[str, Union[int, str]]] = [
    {"id": 1, "name": "pedestrian"},
]


class DanceTrack(MOT):
    """DanceTrack dataset class.

    Public detections notes:
    - train:
        - my_det.txt: official YoloX weights from DanceTrack, trained on train set, ran by Baptiste to get these detections
    - val:
        - my_det.txt: official YoloX weights from DanceTrack, trained on train set, ran by Baptiste to get these detections
        - yolox_dets.txt: official YoloX detections from GHOST, trained on train set
    - test
        - my_det.txt: official YoloX weights from DanceTrack, trained on train set, ran by Baptiste to get these detections
        - yolox_dets.txt: official YoloX detections from GHOST, trained on train (+val?) set
    """

    # 40 train videos
    # 25 val videos
    # 35 test videos
    name = "DanceTrack"
    nickname = "dt"

    def __init__(
        self,
        dataset_path: str,
        nvid: int = -1,
        nframes: int = -1,
        vids_dict: Optional[Dict[str, List[str]]] = None,
        public_dets_subpath: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Initialize DanceTrack dataset.

        Args:
            dataset_path: Path to the dataset.
            nvid: Number of videos to use.
            nframes: Number of frames per video.
            vids_dict: Dictionary of video IDs per split.
            public_dets_subpath: Subpath for public detections.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        log.info(f"Loading DanceTrack dataset from {dataset_path}.")
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
