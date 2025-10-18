import cv2
import numpy as np
from pathlib import Path

from tracklab.utils.cv2 import draw_text
from .visualizer import ImageVisualizer

from tracklab.pipeline.calibration.sn_calibration_baseline.soccerpitch import (
    SoccerPitch,
)

import logging

log = logging.getLogger(__name__)

pitch_file = Path(__file__).parent / "Radar.png"


class Pitch(ImageVisualizer):
    def draw_frame(
        self, image, detections_pred, detections_gt, image_pred, image_gt
    ) -> None:
        draw_pitch(image, detections_pred, detections_gt, image_pred)


class Radar(ImageVisualizer):
    def draw_frame(
        self, image, detections_pred, detections_gt, image_pred, image_gt
    ) -> None:
        for detection, group in zip(
            [detections_pred, detections_gt], ["Predictions", "Ground Truth"]
        ):
            if detection is not None and "bbox_pitch" in detection:
                draw_radar_view(image, detection, group=group)


class Minimap(ImageVisualizer):
    def draw_frame(
        self, image, detections_pred, detections_gt, image_pred, image_gt
    ) -> None:
        # Create a small minimap image
        pitch_width = 105 + 2 * 10  # pitch size + 2 * margin
        pitch_height = 68 + 2 * 5  # pitch size + 2 * margin
        scale = 8
        minimap = (
            np.ones((pitch_height * scale, pitch_width * scale, 3), dtype=np.uint8)
            * 255
        )
        if pitch_file is not None:
            minimap = cv2.resize(
                cv2.imread(str(pitch_file)), (pitch_width * scale, pitch_height * scale)
            )
        # Draw predictions only
        if detections_pred is not None and "bbox_pitch" in detections_pred:
            draw_radar_view_minimap(minimap, detections_pred, scale=scale)
        # Replace the image with the minimap
        image[:] = cv2.resize(minimap, (image.shape[1], image.shape[0]))


def draw_radar_view_minimap(radar_img, detections, scale=8) -> None:
    pitch_width = 105 + 2 * 10
    pitch_height = 68 + 2 * 5
    radar_center_x = int(pitch_width * scale / 2)
    radar_center_y = int(pitch_height * scale / 2)
    for name, detection in detections.iterrows():
        # Set color based on role/team
        if "role" in detection and detection.role == "ball":
            color = (0, 255, 255)  # Yellow for ball
        elif "role" in detection and "team" in detection:
            color = (0, 0, 255) if detection.team == "left" else (255, 0, 0)
        else:
            color = (0, 0, 0)
        bbox_name = "bbox_pitch"
        if (
            not isinstance(detection[bbox_name], dict)
            or detection[bbox_name]["x_bottom_middle"] is None
        ):
            continue
        x_middle = np.clip(detection[bbox_name]["x_bottom_middle"], -10000, 10000)
        y_middle = np.clip(detection[bbox_name]["y_bottom_middle"], -10000, 10000)
        cat = None
        if "jersey_number" in detection and detection.jersey_number is not None:
            if "role" in detection and detection.role == "player":
                if isinstance(detection.jersey_number, float) and np.isnan(
                    detection.jersey_number
                ):
                    cat = None
                else:
                    cat = f"{int(detection.jersey_number)}"

        if "role" in detection:
            if detection.role == "ball":
                cat = "B"  # Ball marker
            elif detection.role == "goalkeeper":
                cat = "GK"
            elif detection.role == "referee":
                cat = "RE"
                color = (238, 210, 2)
            elif detection.role == "other":
                cat = "OT"
                color = (0, 255, 0)
        if cat is not None:
            draw_text(
                radar_img,
                cat,
                (
                    radar_center_x + int(x_middle * scale),
                    radar_center_y + int(y_middle * scale),
                ),
                1,
                int(0.3 * scale),  # Increased ball size from 0.2 to 0.3
                color_txt=(255, 255, 255),
                color_bg=color,
                alignH="c",
                alignV="c",
            )
        else:
            cv2.circle(
                radar_img,
                (
                    radar_center_x + int(x_middle * scale),
                    radar_center_y + int(y_middle * scale),
                ),
                scale,
                color=color,
                thickness=-1,
            )


def draw_pitch(
    patch,
    detections_pred,
    detections_gt,
    image_pred,
    line_thickness=3,
) -> None:
    # Draw the lines on the image pitch
    if "lines" in image_pred and isinstance(image_pred["lines"], dict):
        image_height, image_width, _ = patch.shape
        for name, line in image_pred["lines"].items():
            if name == "Circle central" and len(line) > 4:
                points = np.array(
                    [
                        (int(p["x"] * image_width), int(p["y"] * image_height))
                        for p in line
                    ]
                )
                ellipse = cv2.fitEllipse(points)
                cv2.ellipse(
                    patch,
                    ellipse,
                    color=SoccerPitch.palette[name],
                    thickness=line_thickness,
                )
            else:
                for j in np.arange(len(line) - 1):
                    cv2.line(
                        patch,
                        (
                            int(line[j]["x"] * image_width),
                            int(line[j]["y"] * image_height),
                        ),
                        (
                            int(line[j + 1]["x"] * image_width),
                            int(line[j + 1]["y"] * image_height),
                        ),
                        color=SoccerPitch.palette[name],
                        thickness=line_thickness,  # TODO : make this a parameter
                    )


def draw_radar_view(patch, detections, scale=4, delta=32, group="Ground Truth") -> None:
    pitch_width = 105 + 2 * 10  # pitch size + 2 * margin
    pitch_height = 68 + 2 * 5  # pitch size + 2 * margin
    sign = -1 if group == "Ground Truth" else +1
    y_delta = 3
    # Use actual image dimensions instead of hardcoded 1920x1080
    img_height, img_width = patch.shape[:2]
    radar_center_x = int(img_width / 2 - pitch_width * scale / 2 * sign - delta * sign)
    radar_center_y = int(img_height - pitch_height * scale / 2 - y_delta)
    radar_top_x = int(radar_center_x - pitch_width * scale / 2)
    radar_top_y = int(img_height - pitch_height * scale - y_delta)
    radar_width = int(pitch_width * scale)
    radar_height = int(pitch_height * scale)
    if pitch_file is not None:
        radar_img = cv2.resize(
            cv2.imread(str(pitch_file)), (pitch_width * scale, pitch_height * scale)
        )
        cv2.line(
            radar_img, (0, 0), (0, radar_img.shape[0]), thickness=6, color=(0, 0, 255)
        )
        cv2.line(
            radar_img,
            (radar_img.shape[1], 0),
            (radar_img.shape[1], radar_img.shape[0]),
            thickness=6,
            color=(255, 0, 0),
        )
    else:
        radar_img = np.ones((pitch_height * scale, pitch_width * scale, 3)) * 255

    # Check if radar region is within image bounds
    if (
        radar_top_x < 0
        or radar_top_y < 0
        or radar_top_x + radar_width > img_width
        or radar_top_y + radar_height > img_height
    ):
        return  # Skip radar if it doesn't fit

    alpha = 0.3
    patch[
        radar_top_y : radar_top_y + radar_height,
        radar_top_x : radar_top_x + radar_width,
        :,
    ] = cv2.addWeighted(
        patch[
            radar_top_y : radar_top_y + radar_height,
            radar_top_x : radar_top_x + radar_width,
            :,
        ],
        1 - alpha,
        radar_img,
        alpha,
        0.0,
    )
    patch[
        radar_top_y : radar_top_y + radar_height,
        radar_top_x : radar_top_x + radar_width,
        :,
    ] = cv2.addWeighted(
        patch[
            radar_top_y : radar_top_y + radar_height,
            radar_top_x : radar_top_x + radar_width,
            :,
        ],
        1 - alpha,
        radar_img,
        alpha,
        0.0,
    )
    draw_text(
        patch,
        group,
        (radar_center_x, radar_top_y - 5),
        0,
        1,
        1,
        color_txt=(255, 255, 255),
        color_bg=None,
        alignH="c",
        alignV="t",
    )
    for name, detection in detections.iterrows():
        # Set color based on role/team
        if "role" in detection and detection.role == "ball":
            color = (0, 255, 255)  # Yellow for ball
        elif "role" in detection and "team" in detection:
            color = (0, 0, 255) if detection.team == "left" else (255, 0, 0)
        else:
            color = (0, 0, 0)
        bbox_name = "bbox_pitch"
        if (
            not isinstance(detection[bbox_name], dict)
            or detection[bbox_name]["x_bottom_middle"] is None
        ):
            continue
        x_middle = np.clip(detection[bbox_name]["x_bottom_middle"], -10000, 10000)
        y_middle = np.clip(detection[bbox_name]["y_bottom_middle"], -10000, 10000)
        cat = None
        if "jersey_number" in detection and detection.jersey_number is not None:
            if "role" in detection and detection.role == "player":
                if isinstance(detection.jersey_number, float) and np.isnan(
                    detection.jersey_number
                ):
                    cat = None
                else:
                    cat = f"{int(detection.jersey_number)}"

        if "role" in detection:
            if detection.role == "ball":
                cat = "B"  # Ball marker
            elif detection.role == "goalkeeper":
                cat = "GK"
            elif detection.role == "referee":
                cat = "RE"
                color = (238, 210, 2)
            elif detection.role == "other":
                cat = "OT"
                color = (0, 255, 0)
        if cat is not None:
            draw_text(
                patch,
                cat,
                (
                    radar_center_x + int(x_middle * scale),
                    radar_center_y + int(y_middle * scale),
                ),
                1,
                0.3 * scale,  # Increased ball size from 0.2 to 0.3
                1,
                color_txt=color,
                color_bg=None,
                alignH="c",
                alignV="b",
            )
        else:
            cv2.circle(
                patch,
                (
                    radar_center_x + int(x_middle * scale),
                    radar_center_y + int(y_middle * scale),
                ),
                scale,
                color=color,
                thickness=-1,
            )
