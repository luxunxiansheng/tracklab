"""Ball visualization classes for TrackLab."""

from typing import Any, Optional
import cv2
import numpy as np

from .visualizer import DetectionVisualizer


class BallCircle(DetectionVisualizer):
    """Ball visualizer that draws circles around detected balls."""

    def __init__(
        self,
        print_id: bool = True,
        print_confidence: bool = False,
        circle_color: tuple = (0, 255, 255),  # Yellow in BGR
        circle_thickness: int = 2,
    ) -> None:
        """Initialize the ball circle visualizer.

        Args:
            print_id: Whether to print track IDs on balls.
            print_confidence: Whether to print confidence scores on balls.
            circle_color: Color of the circle in BGR format.
            circle_thickness: Thickness of the circle border.
        """
        super().__init__()
        self.print_id = print_id
        self.print_confidence = print_confidence
        self.circle_color = circle_color
        self.circle_thickness = circle_thickness

    def draw_detection(
        self,
        image: np.ndarray,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw ball detection as a circle on the image.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection data.
            detection_gt: Ground truth detection data.
            metric: Matching metric between pred and gt (unused).
        """
        for detection, is_pred in zip([detection_pred, detection_gt], [True, False]):
            if detection is not None:
                # Only draw if this is a ball (category_id == 2 or role == "ball")
                is_ball = False
                if hasattr(detection, "category_id") and detection.category_id == 2:
                    is_ball = True
                elif hasattr(detection, "role") and detection.role == "ball":
                    is_ball = True

                if is_ball:
                    x1, y1, x2, y2 = detection.bbox.ltrb()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    # Use bbox width as approximate ball diameter
                    radius = max(int((x2 - x1) / 2), int((y2 - y1) / 2))

                    # Draw circle
                    cv2.circle(
                        image,
                        (center_x, center_y),
                        radius,
                        self.circle_color,
                        self.circle_thickness,
                    )

                    # Optionally draw track ID
                    if self.print_id:
                        has_track_id = (
                            hasattr(detection, "track_id")
                            and "track_id" in detection.index
                            and not np.isnan(detection.track_id)
                        )
                        if has_track_id:
                            text = f"B{int(detection.track_id)}"
                            # Position text above the circle
                            text_y = center_y - radius - 5
                            cv2.putText(
                                image,
                                text,
                                (center_x - 15, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                self.circle_color,
                                1,
                                cv2.LINE_AA,
                            )

                    # Optionally draw confidence
                    if self.print_confidence and hasattr(detection, "bbox_conf"):
                        conf_text = f"{detection.bbox_conf:.2f}"
                        text_y = center_y + radius + 15
                        cv2.putText(
                            image,
                            conf_text,
                            (center_x - 15, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            self.circle_color,
                            1,
                            cv2.LINE_AA,
                        )


class BallBBox(DetectionVisualizer):
    """Ball visualizer that draws bounding boxes around detected balls."""

    def __init__(
        self,
        print_id: bool = True,
        print_confidence: bool = False,
        bbox_color: tuple = (0, 255, 255),  # Yellow in BGR
        bbox_thickness: int = 2,
    ) -> None:
        """Initialize the ball bbox visualizer.

        Args:
            print_id: Whether to print track IDs on balls.
            print_confidence: Whether to print confidence scores on balls.
            bbox_color: Color of the bounding box in BGR format.
            bbox_thickness: Thickness of the bbox border.
        """
        super().__init__()
        self.print_id = print_id
        self.print_confidence = print_confidence
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness

    def draw_detection(
        self,
        image: np.ndarray,
        detection_pred: Optional[Any],
        detection_gt: Optional[Any],
        metric: Optional[float] = None,
    ) -> None:
        """Draw ball detection as a bounding box on the image.

        Args:
            image: Image to draw on.
            detection_pred: Predicted detection data.
            detection_gt: Ground truth detection data.
            metric: Matching metric between pred and gt (unused).
        """
        for detection, is_pred in zip([detection_pred, detection_gt], [True, False]):
            if detection is not None:
                # Only draw if this is a ball (category_id == 2 or role == "ball")
                is_ball = False
                if hasattr(detection, "category_id") and detection.category_id == 2:
                    is_ball = True
                elif hasattr(detection, "role") and detection.role == "ball":
                    is_ball = True

                if is_ball:
                    x1, y1, x2, y2 = detection.bbox.ltrb()

                    # Draw bounding box
                    cv2.rectangle(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        self.bbox_color,
                        self.bbox_thickness,
                    )

                    # Optionally draw track ID
                    if self.print_id:
                        has_track_id = (
                            hasattr(detection, "track_id")
                            and "track_id" in detection.index
                            and not np.isnan(detection.track_id)
                        )
                        if has_track_id:
                            text = f"Ball-{int(detection.track_id)}"
                            cv2.putText(
                                image,
                                text,
                                (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                self.bbox_color,
                                1,
                                cv2.LINE_AA,
                            )

                    # Optionally draw confidence
                    if self.print_confidence and hasattr(detection, "bbox_conf"):
                        conf_text = f"{detection.bbox_conf:.2f}"
                        cv2.putText(
                            image,
                            conf_text,
                            (int(x1), int(y2) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            self.bbox_color,
                            1,
                            cv2.LINE_AA,
                        )
