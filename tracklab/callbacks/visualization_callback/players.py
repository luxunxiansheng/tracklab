"""Player visualization classes for TrackLab with team-based coloring."""

from typing import Any, Dict, List, Optional
import cv2
import pandas as pd

from distinctipy import get_rgb256
from regex import F

from .visualizer import Visualizer, get_fixed_colors
from .detection import DefaultDetection, EllipseDetection
from tracklab.utils.cv2 import draw_text

import logging

log = logging.getLogger(__name__)


class TeamVisualizer(Visualizer):
    def post_init(self, colors: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize team-based coloring.

        Args:
            colors: Color configuration dictionary.
            **kwargs: Additional attributes.
        """
        self.colors = colors
        cmap = get_fixed_colors(colors["cmap"])
        self.cmap = [get_rgb256(i) for i in cmap]

    def color(
        self, detection: Any, is_prediction: bool, color_type: str = "default"
    ) -> Optional[Any]:
        """Get color for detection based on team or track ID.

        Args:
            detection: Detection data.
            is_prediction: Whether this is a prediction or ground truth.
            color_type: Type of coloring to use.

        Returns:
            Color value or None.

        Raises:
            ValueError: If color_type is not defined in colors config.
        """
        assert self.colors is not None
        if color_type not in self.colors:
            if "default" in self.colors:
                color_type = "default"
            else:
                raise ValueError(
                    f"{color_type} not declared in the colors dict for visualization"
                )

        # Check if track_id column exists (it won't exist if tracking is disabled)
        has_track_id = hasattr(detection, "track_id") and "track_id" in detection.index

        if not has_track_id or pd.isna(detection.track_id):
            color = self.colors[color_type].get("no_id")
        else:
            cmap_key = "prediction" if is_prediction else "ground_truth"
            if self.colors[color_type][cmap_key] == "track_id":
                color = self.cmap[(int(detection.track_id) - 1) % len(self.cmap)]
            elif self.colors[color_type][cmap_key] == "team":
                if "team" not in self.colors:
                    color = self.colors[color_type].get("no_id", [255, 0, 0])
                else:
                    try:
                        if hasattr(detection, "role") and detection.role == "referee":
                            color = self.colors["team"][cmap_key]["referee"]
                        elif hasattr(detection, "team") and detection.team in [
                            "left",
                            "right",
                        ]:
                            color = self.colors["team"][cmap_key][detection.team]
                        else:
                            color = self.colors["team"]["no_team"]
                    except Exception as e:
                        log.warning(f"Error accessing team info for detection: {e}")
                        color = self.colors[color_type].get("no_id", [255, 0, 0])
            else:
                color = self.colors[color_type][cmap_key]
        return color


class Player(TeamVisualizer, DefaultDetection):
    pass


class PlayerEllipse(TeamVisualizer, EllipseDetection):
    pass


class CompletePlayerEllipse(TeamVisualizer, EllipseDetection):
    """Advanced player visualizer with detailed information display."""

    def __init__(
        self,
        display_track_id: bool = True,
        display_jersey: bool = False,
        display_role: bool = False,
        display_team: bool = False,
    ) -> None:
        """Initialize the complete player ellipse visualizer.

        Args:
            display_track_id: Whether to display track ID.
            display_jersey: Whether to display jersey number.
            display_role: Whether to display player role.
            display_team: Whether to display team information.
        """
        self.display_list: List[str] = [
            "track_id" if display_track_id else "",
            "jersey_number" if display_jersey else "",
            "role" if display_role else "",
            "team" if display_team else "",
        ]
        self.display_list = [item for item in self.display_list if item]
        super().__init__()

    def draw_detection(
        self,
        image: Any,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw player detection with detailed information.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection data.
            detection_gt: Ground truth detection data.
            metric: Matching metric (unused).
        """
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
                    txt = [
                        pprint(v, getattr(detection, v, None))
                        for v in self.display_list
                    ]
                    txt = "\n".join([v for v in txt if v != ""])
                    draw_text(
                        image,
                        txt,
                        (center[0], center[1]),
                        fontFace=1,
                        fontScale=1,  # Changed from 0.75 to 1
                        thickness=1,
                        alignH="c",
                        alignV="c",
                        color_bg=color,
                        color_txt=None,
                        alpha_bg=0.6,
                    )


def pprint(key: str, value: Any) -> str:
    """Pretty print key-value pair for display.

    Args:
        key: The key to display.
        value: The value to display.

    Returns:
        Formatted string for display.
    """
    if value is None:
        return ""
    if key == "track_id":
        return f"ID:{value}"
    elif key == "jersey_number":
        return f"#{value}"
    elif key == "role":
        return f"{value}"
    elif key == "team":
        return f"{value}"
    else:
        return f"{key}:{value}"
