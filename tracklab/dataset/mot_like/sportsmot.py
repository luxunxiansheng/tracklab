import logging
from typing import Dict, List, Optional, Union

from .common import MOT

log = logging.getLogger(__name__)

categories_list: List[Dict[str, Union[int, str]]] = [
    {"id": 1, "name": "pedestrian"},
]


class SportsMOT(MOT):
    """SportsMOT dataset class.

    Public detections notes:
    - test
        - dets.txt: official detections from deep_eiou
        - diffmot_yolox_x.txt: official detections from diffmot, trained on train set
        - diffmot_yolox_x_mix.txt: official detections from diffmot, trained on train + val set
    - val
        - deep_eiou.txt: official YoloX weights from deep_eiou, ran by Baptiste to get hese detections. Train on train+val set
        - diffmot_yolox_x.txt: official detections from diffmot, trained on train set
        - diffmot_yolox_x_mix.txt: official detections from diffmot, trained on train + val set
    - train
        - deep_eiou.txt: official YoloX weights from deep_eiou, ran by Baptiste to get hese detections. Train on train+val set
    """

    # 45 train videos
    # 45 val videos
    # 150 test videos
    name = "SportsMOT"
    nickname = "sm"

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
        """Initialize SportsMOT dataset.

        Args:
            dataset_path: Path to the dataset.
            nvid: Number of videos to use.
            nframes: Number of frames per video.
            vids_dict: Dictionary of video IDs per split.
            public_dets_subpath: Subpath for public detections.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        log.info(f"Loading SportsMOT dataset from {dataset_path}.")
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
