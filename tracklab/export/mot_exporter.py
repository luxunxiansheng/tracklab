import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .base_exporter import BaseExporter


class MOTExporter(BaseExporter):
    """
    Exporter for MOT Challenge format.
    Exports tracking data in the standard MOT Challenge format.
    """

    def export(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        save_path: str,
        bbox_column: str = "bbox_ltwh",
        save_classes: bool = False,
        is_ground_truth: bool = False,
        save_zip: bool = True,
        **kwargs,
    ) -> None:
        """
        Export detections in MOT Challenge format.

        Args:
            detections: DataFrame containing detection data
            image_metadatas: DataFrame containing image metadata
            video_metadatas: DataFrame containing video metadata
            save_path: Path where to save the exported data
            bbox_column: Column name containing bounding box data
            save_classes: Whether to include class information
            is_ground_truth: Whether this is ground truth data
            save_zip: Whether to save as zip (not implemented yet)
            **kwargs: Additional parameters
        """
        mot_df = self._mot_encoding(
            detections, image_metadatas, video_metadatas, bbox_column
        )

        save_path_obj = self._ensure_directory(save_path)

        # MOT Challenge format = <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        for video_id, video in video_metadatas.iterrows():
            file_path = save_path_obj / f"{video['name']}.txt"
            file_df = mot_df[mot_df["video_id"] == video_id].copy()

            # MOT Challenge format starts at frame 1
            if not file_df.empty:
                file_df["frame"] = file_df["frame"] + 1
                file_df.sort_values(by="frame", inplace=True)

                # Choose which column to use for the class field
                clazz = "category_id" if save_classes else "x"

                file_df[
                    [
                        "frame",
                        "track_id",
                        "bb_left",
                        "bb_top",
                        "bb_width",
                        "bb_height",
                        "bbox_conf",
                        clazz,
                        "y",
                        "z",
                    ]
                ].to_csv(
                    file_path,
                    header=False,
                    index=False,
                )
            else:
                # Create empty file
                file_path.touch()

    def _mot_encoding(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        bbox_column: str,
    ) -> pd.DataFrame:
        """
        Convert detections to MOT format DataFrame.
        """
        df = detections.copy()

        # Extract bbox coordinates
        df["bb_left"] = df[bbox_column].apply(lambda x: x[0])
        df["bb_top"] = df[bbox_column].apply(lambda x: x[1])
        df["bb_width"] = df[bbox_column].apply(lambda x: x[2])
        df["bb_height"] = df[bbox_column].apply(lambda x: x[3])

        # Add placeholder columns for MOT format
        df = df.assign(x=-1, y=-1, z=-1)

        return df
