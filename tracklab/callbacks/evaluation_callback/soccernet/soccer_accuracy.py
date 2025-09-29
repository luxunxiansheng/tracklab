"""Soccer accuracy evaluator for TrackLab soccer tracking evaluation."""

from typing import Any, TYPE_CHECKING

import pycocotools  # type: ignore

from tracklab.pipeline import Evaluator

if TYPE_CHECKING:
    from tracklab.datastruct import TrackerState


class SoccerAccuracy(Evaluator):
    """Evaluator for soccer tracking accuracy using COCO evaluation metrics.

    This evaluator computes accuracy metrics for soccer tracking tasks
    using the pycocotools library for COCO-style evaluation.
    """

    def __init__(self, eval_set: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the soccer accuracy evaluator.

        Args:
            eval_set: The evaluation set to use (e.g., 'val', 'test').
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.eval_set = eval_set

    def run(self, tracker_state: "TrackerState") -> None:
        """Run soccer accuracy evaluation on the tracker state.

        Args:
            tracker_state: The tracker state containing predictions and ground truth.
        """
        pycocotools
