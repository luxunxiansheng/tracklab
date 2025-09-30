import copy
import logging
import os
from abc import ABC
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Any

import numpy as np
import pandas as pd

from tracklab.utils import wandb

log = logging.getLogger(__name__)


class SetsDict(dict):
    """Dictionary subclass that provides better error messages for missing dataset splits."""

    def __getitem__(self, key: str) -> Any:
        if key not in self:
            raise KeyError(
                f"Trying to access a '{key}' split of the dataset that is not available. "
                f"Available splits are {list(self.keys())}. "
                f"Make sur this split name is correct or is available in the dataset folder."
            )
        return super().__getitem__(key)


@dataclass
class TrackingSet:
    """Represents a set of tracking data including videos, images, and detections."""

    video_metadatas: pd.DataFrame
    image_metadatas: pd.DataFrame
    detections_gt: Optional[pd.DataFrame]
    image_gt: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["video_id"])
    )
    detections_public: Optional[pd.DataFrame] = None
    detections_pred: Optional[pd.DataFrame] = None

    def filter_videos(self, keep_video_ids: List[int]) -> None:
        """Filter the tracking set to keep only specified video IDs.

        Args:
            keep_video_ids: List of video IDs to keep.
        """
        self.video_metadatas = self.video_metadatas.loc[keep_video_ids]
        self.image_metadatas = self.image_metadatas[
            self.image_metadatas.video_id.isin(keep_video_ids)
        ]
        if self.detections_gt is not None:
            self.detections_gt = self.detections_gt[
                self.detections_gt.video_id.isin(keep_video_ids)
            ]
        self.image_gt = self.image_gt[self.image_gt.video_id.isin(keep_video_ids)]


class TrackingDataset(ABC):
    """Abstract base class for tracking datasets."""

    def __init__(
        self,
        dataset_path: str,
        sets: Dict[str, TrackingSet],
        nvid: int = -1,
        nframes: int = -1,
        vids_dict: Optional[Dict[str, List[str]]] = None,
        *,
        set_split_idxs: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        """Initialize the tracking dataset.

        Args:
            dataset_path: Path to the dataset.
            sets: Dictionary of dataset splits.
            nvid: Number of videos to subsample (-1 for all).
            nframes: Number of frames per video to subsample (-1 for all).
            vids_dict: Dictionary of video names per split.
            set_split_idxs: Indices for dataset splits.
            **kwargs: Additional arguments.
        """
        set_split_idxs = set_split_idxs or {}
        self.dataset_path = Path(dataset_path)
        self.sets = SetsDict(sets)
        sub_sampled_sets = SetsDict()
        for set_name, split in self.sets.items():
            vid_list = (
                vids_dict[set_name]
                if vids_dict is not None and set_name in vids_dict
                else None
            )
            sub_sampled_sets[set_name] = self._subsample(split, nvid, nframes, vid_list)
        assert (len(set_split_idxs) == 0) or (
            nvid == -1
        ), "Splitting the dataset and setting nvid to a different value is not supported"
        self.training_sets = copy.deepcopy(self.sets)
        self.sets = sub_sampled_sets
        self.set_splits = {}
        self.set_split_idxs = set_split_idxs

        for set_name, split_idx in set_split_idxs.items():
            self.set_splits[set_name] = []
            self._split_set(set_name)
            self.sets[set_name] = self.set_splits[set_name][split_idx]
            self.training_sets[set_name] = self.set_splits[set_name][split_idx]

    def _split_set(self, set_name: str, num_splits: int = 2) -> None:
        """Split a dataset set into multiple parts for cross-validation.

        Args:
            set_name: Name of the set to split.
            num_splits: Number of splits to create.
        """
        video_groups = [[] for i in range(num_splits)]
        people_in_video = [set() for i in range(num_splits)]
        for video_id, _ in (
            self.sets[set_name]
            .detections_gt.groupby("video_id")
            .person_id.nunique()
            .sort_values(ascending=False)
            .items()
        ):
            video_df = self.sets[set_name].detections_gt.loc[
                self.sets[set_name].detections_gt.video_id == video_id
            ]
            for person_id in np.unique(video_df.person_id):
                group_idxs = np.nonzero(
                    [np.isin(person_id, x) for x in people_in_video]
                )[0]
                if len(group_idxs) > 0:
                    current_group = group_idxs[0]
                    break
            else:
                current_group = np.argmin(
                    [len(x) for x in video_groups]
                )  # group to put it in

            video_groups[current_group].append(video_id)
            people_in_video[current_group].update(video_df.person_id)

        self.train_sets = []
        for video_ids in video_groups:
            current_set = copy.deepcopy(self.sets[set_name])
            current_set.filter_videos(video_ids)
            self.set_splits[set_name].append(current_set)

    def _subsample(
        self,
        tracking_set: TrackingSet,
        nvid: int,
        nframes: int,
        vids_names: Optional[List[str]],
    ) -> TrackingSet:
        """Subsample a tracking set to reduce its size.

        Args:
            tracking_set: The tracking set to subsample.
            nvid: Number of videos to keep.
            nframes: Number of frames per video to keep.
            vids_names: List of specific video names to keep.

        Returns:
            Subsampled tracking set.
        """
        if (
            nvid < 1
            and nframes < 1
            and (vids_names is None or len(vids_names) == 0)
            or tracking_set is None
        ):
            return tracking_set

        # filter videos:
        if vids_names is not None and len(vids_names) > 0:
            assert set(vids_names).issubset(
                tracking_set.video_metadatas.name.unique()
            ), f"Some videos to process {set(vids_names) - set(tracking_set.video_metadatas.name.unique())} does not exist in the tracking set"
            videos_to_keep = tracking_set.video_metadatas[
                tracking_set.video_metadatas.name.isin(vids_names)
            ].index
            tiny_video_metadatas = tracking_set.video_metadatas.loc[videos_to_keep]
        elif nvid > 0:  # keep 'nvid' videos
            videos_to_keep = tracking_set.video_metadatas.sample(
                nvid, random_state=2
            ).index
            tiny_video_metadatas = tracking_set.video_metadatas.loc[videos_to_keep]
        else:  # keep all videos
            videos_to_keep = tracking_set.video_metadatas.index
            tiny_video_metadatas = tracking_set.video_metadatas

        # filter images:
        # keep only images from videos to keep
        tiny_image_metadatas = tracking_set.image_metadatas[
            tracking_set.image_metadatas.video_id.isin(videos_to_keep)
        ]
        tiny_image_gt = tracking_set.image_gt[
            tracking_set.image_gt.video_id.isin(videos_to_keep)
        ]

        # keep only images from first nframes
        if nframes > 0:
            tiny_image_metadatas = tiny_image_metadatas.groupby("video_id").head(
                nframes
            )
            tiny_image_gt = tiny_image_gt.groupby("video_id").head(nframes)

        # filter detections:
        tiny_detections = None
        if (
            tracking_set.detections_gt is not None
            and not tracking_set.detections_gt.empty
        ):
            tiny_detections = tracking_set.detections_gt[
                tracking_set.detections_gt.image_id.isin(tiny_image_metadatas.index)
            ]

        assert (
            len(tiny_video_metadatas) > 0
        ), "No videos left after subsampling the tracking set"
        assert (
            len(tiny_image_metadatas) > 0
        ), "No images left after subsampling the tracking set"

        tiny_tracking_set = TrackingSet(
            tiny_video_metadatas,
            tiny_image_metadatas,
            tiny_detections,
            tiny_image_gt,
        )

        if (
            hasattr(tracking_set, "detections_public")
            and tracking_set.detections_public is not None
            and not tracking_set.detections_public.empty
        ):
            tiny_public_detections = tracking_set.detections_public[
                tracking_set.detections_public.image_id.isin(tiny_image_metadatas.index)
            ]
            tiny_tracking_set.detections_public = tiny_public_detections

        if (
            hasattr(tracking_set, "detections_pred")
            and tracking_set.detections_pred is not None
            and not tracking_set.detections_pred.empty
        ):
            tiny_pred_detections = tracking_set.detections_pred[
                tracking_set.detections_pred.image_id.isin(tiny_image_metadatas.index)
            ]
            tiny_tracking_set.detections_pred = tiny_pred_detections

        return tiny_tracking_set

    @staticmethod
    def _mot_encoding(
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        bbox_column: str,
    ) -> pd.DataFrame:
        """Encode detections in MOT format.

        Args:
            detections: Detection dataframe.
            image_metadatas: Image metadata dataframe.
            video_metadatas: Video metadata dataframe.
            bbox_column: Name of the bbox column.

        Returns:
            DataFrame in MOT format.
        """
        detections = detections.copy()
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            detections.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
            suffixes=("", "_y"),
        )
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "frame",
                "track_id",
                bbox_column,
            ],
            how="any",
            inplace=True,
        )

        if len_before_drop != len(df):
            log.warning(
                "Dropped {} rows with NA values".format(len_before_drop - len(df))
            )
        df["track_id"] = df["track_id"].astype(int)
        df["bb_left"] = df[bbox_column].apply(lambda x: x[0])
        df["bb_top"] = df[bbox_column].apply(lambda x: x[1])
        df["bb_width"] = df[bbox_column].apply(lambda x: x[2])
        df["bb_height"] = df[bbox_column].apply(lambda x: x[3])
        df = df.assign(x=-1, y=-1, z=-1)
        return df

    def process_trackeval_results(
        self,
        results: Dict[str, Any],
        dataset_config: Dict[str, Any],
        eval_config: Dict[str, Any],
    ) -> None:
        """Process and log TrackEval results.

        Args:
            results: Evaluation results dictionary.
            dataset_config: Dataset configuration.
            eval_config: Evaluation configuration.
        """
        log.info(f"TrackEval results = {results}")
        wandb.log(results)

    def __str__(self) -> str:
        set_str = []
        for set_name, set_data in self.sets.items():
            if set_data is not None:
                set_str.append(f"{set_name} set: {len(set_data.video_metadatas)}")
        return self.__class__.__name__ + "= " + "; ".join(set_str)
