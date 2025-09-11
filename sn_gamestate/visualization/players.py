import cv2
import pandas as pd

from distinctipy import get_rgb256

from tracklab.visualization import Visualizer, DefaultDetection, EllipseDetection, get_fixed_colors
from tracklab.utils.cv2 import draw_text

import logging

log = logging.getLogger(__name__)

class TeamVisualizer(Visualizer):
    def post_init(self, colors, **kwargs):
        super().post_init(colors, **kwargs)
        self.colors = colors
        cmap = get_fixed_colors(colors["cmap"])
        self.cmap = [get_rgb256(i) for i in cmap]

    def color(self, detection, is_prediction, color_type="default"):
        assert self.colors is not None
        if color_type not in self.colors:
            raise ValueError(f"{color_type} not declared in the colors dict for visualization")
        if pd.isna(detection.track_id):
            color = self.colors[color_type].no_id
        else:
            cmap_key = "prediction" if is_prediction else "ground_truth"
            if self.colors[color_type][cmap_key] == "track_id":
                color = self.cmap[(int(detection.track_id) - 1) % len(self.cmap)]
            elif self.colors[color_type][cmap_key] == "team":
                if hasattr(detection, "role") and detection.role == "referee":
                    return self.colors["team"][cmap_key]["referee"]
                elif hasattr(detection, "team") and detection.team in ["left", "right"]:
                    return self.colors["team"][cmap_key][detection.team]
                else:
                    return self.colors["team"]["no_team"]
            else:
                color = self.colors[color_type][cmap_key]
        return color

class Player(TeamVisualizer, DefaultDetection):
    pass

class PlayerEllipse(TeamVisualizer, EllipseDetection):
    pass

class CompletePlayerEllipse(TeamVisualizer, EllipseDetection):
    def __init__(self, display_track_id=True, display_jersey=True, display_role=True, display_team=False):
        self.display_list = [
            "track_id" if display_track_id else None,
            "jersey_number" if display_jersey else None,
            "role" if display_role else None,
            "team" if display_team else None,
        ]
        self.display_list = [item for item in self.display_list if item]
        super().__init__()

    def draw_detection(self, image, detection_pred, detection_gt, metric=None):
        for detection, is_pred in zip([detection_pred, detection_gt], [True, False]):
            if detection is not None:
                color = self.color(detection, is_prediction=is_pred)
                if color:
                    x1, y1, x2, y2 = detection.bbox.ltrb()
                    center = (int((x1 + x2) / 2), int(y2))
                    width = x2 - x1
                    cv2.ellipse(
                        image,
                        center=center,
                        axes=(int(width), int(0.35 * width)),
                        angle=0.0,
                        startAngle=-45.0,
                        endAngle=235.0,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    txt = [pprint(v, getattr(detection, v, lambda: None)) for v in self.display_list]
                    txt = "\n".join([v for v in txt if v != ""])
                    draw_text(
                        image,
                        txt,
                        (center[0], center[1]),
                        fontFace=1,
                        fontScale=0.75,
                        thickness=1,
                        alignH="c",
                        alignV="c",
                        color_bg=color,
                        color_txt=None,
                        alpha_bg=0.6,
                    )

def pprint(key, value):
    if key == "track_id" and not pd.isna(value):
        return f"ID: {int(value)}"
    elif key == "jersey_number" and not pd.isna(value):
        return f"JN: {int(value)}"
    elif key == "role":
        return {
            "referee": "R: RE",
            "player": "R: P",
            "goalkeeper": "R: GK",
            "other": "OT",
            "ball": "R: B"
        }.get(value, "")
    elif key == "team":
        return {
            "left": "T: L",
            "right": "T: R"
        }.get(value, "")
    else:
        return ""
