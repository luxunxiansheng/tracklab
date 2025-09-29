import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tracklab.datastruct import TrackingDataset, TrackingSet

log = logging.getLogger(__name__)


class PoseTrack21(TrackingDataset):
    """PoseTrack dataset class for pose tracking evaluation.

    Train set: 43603 images
    Val set: 20161 images
    Test set: ??? images
    """

    name = "posetrack21"
    nickname = "ptt"

    def __init__(
        self,
        dataset_path: str,
        annotation_path: str,
        posetrack_version: int = 21,
        *args,
        **kwargs,
    ) -> None:
        """Initialize PoseTrack dataset.

        Args:
            dataset_path: Path to the dataset images directory.
            annotation_path: Path to the annotations directory.
            posetrack_version: Version of PoseTrack dataset (18, 21, etc.).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.dataset_path = Path(dataset_path)
        assert (
            self.dataset_path.exists()
        ), f"'{self.dataset_path}' directory does not exist"
        self.annotation_path = Path(annotation_path)
        assert (
            self.annotation_path.exists()
        ), f"'{self.annotation_path}' directory does not exist"

        # Dynamically discover splits by listing subdirectories in annotation_path
        potential_splits = [
            d
            for d in os.listdir(self.annotation_path)
            if (self.annotation_path / d).is_dir()
        ]
        sets = {}
        for split in potential_splits:
            split_path = self.annotation_path / split
            if split_path.exists():
                sets[split] = load_tracking_set(
                    split_path, self.dataset_path, posetrack_version
                )
            else:
                log.warning(
                    f"Split '{split}' directory does not exist at '{split_path}'."
                )

        super().__init__(dataset_path, sets, *args, **kwargs)


def load_tracking_set(
    anns_path: Path, dataset_path: Path, posetrack_version: int = 21
) -> TrackingSet:
    """Load a tracking set from PoseTrack annotations.

    Args:
        anns_path: Path to the annotations directory.
        dataset_path: Path to the dataset images directory.
        posetrack_version: Version of PoseTrack dataset.

    Returns:
        TrackingSet containing the loaded data.
    """
    # Load annotations into Pandas dataframes
    video_metadatas, image_metadatas, detections_gt = load_annotations(anns_path)
    # Fix formatting of dataframes to be compatible with tracklab
    video_metadatas, image_metadatas, detections_gt = fix_formatting(
        video_metadatas, image_metadatas, detections_gt, dataset_path, posetrack_version
    )
    return TrackingSet(
        video_metadatas,
        image_metadatas,
        detections_gt,
    )


def load_annotations(
    anns_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PoseTrack annotations from JSON files.

    Args:
        anns_path: Path to the annotations directory.

    Returns:
        Tuple of (video_metadatas, image_metadatas, detections_gt) DataFrames.
    """
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, f"No annotations files found in {anns_path}"
    detections_gt = []
    image_metadatas = []
    video_metadatas = []
    for path in anns_files_list:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            detections_gt.extend(data_dict["annotations"])
            image_metadatas.extend(data_dict["images"])
            video_metadatas.append(
                {
                    "id": data_dict["images"][0]["vid_id"],
                    "nframes": len(data_dict["images"]),
                    "name": path.stem,
                    "categories": data_dict["categories"],
                }
            )

    return (
        pd.DataFrame(video_metadatas),
        pd.DataFrame(image_metadatas),
        pd.DataFrame(detections_gt),
    )


def fix_formatting(
    video_metadatas: pd.DataFrame,
    image_metadatas: pd.DataFrame,
    detections_gt: pd.DataFrame,
    dataset_path: Path,
    posetrack_version: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fix formatting of PoseTrack dataframes for TrackLab compatibility.

    Args:
        video_metadatas: Video metadata DataFrame.
        image_metadatas: Image metadata DataFrame.
        detections_gt: Ground truth detections DataFrame.
        dataset_path: Path to the dataset images directory.
        posetrack_version: Version of PoseTrack dataset.

    Returns:
        Tuple of formatted (video_metadatas, image_metadatas, detections_gt).
    """
    image_id = "image_id" if posetrack_version == 21 else "frame_id"

    # Videos
    video_metadatas.set_index("id", drop=True, inplace=True)

    # Images
    image_metadatas["file_name"] = image_metadatas["file_name"].apply(
        lambda x: os.path.join(dataset_path, x)
    )
    image_metadatas["frame"] = image_metadatas["file_name"].apply(
        lambda x: int(os.path.basename(x).split(".")[0]) + 1
    )
    image_metadatas.rename(
        columns={"vid_id": "video_id", "file_name": "file_path"},
        inplace=True,
    )
    image_metadatas.set_index("id", drop=True, inplace=True)

    # Detections
    detections_gt.drop(["bbox_head"], axis=1, inplace=True)
    detections_gt.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
    detections_gt.bbox_ltwh = detections_gt.bbox_ltwh.apply(lambda x: np.array(x))  # type: ignore
    detections_gt.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
    detections_gt.keypoints_xyc = detections_gt.keypoints_xyc.apply(  # type: ignore
        lambda x: np.reshape(np.array(x), (-1, 3))
    )  # type: ignore
    detections_gt.set_index("id", drop=True, inplace=True)
    detections_gt = detections_gt.merge(
        image_metadatas[["video_id"]], how="left", left_on="image_id", right_index=True
    )

    return video_metadatas, image_metadatas, detections_gt
