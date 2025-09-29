"""Base evaluator classes for TrackLab evaluation functionality."""

from abc import ABC, abstractmethod
from typing import Any

from tracklab.datastruct.tracker_state import TrackerState


class Evaluator(ABC):
    """Abstract base class for dataset evaluators in TrackLab.

    This class defines the interface for implementing custom evaluators
    that can assess tracking performance on different datasets. Evaluators
    are responsible for computing metrics and generating evaluation reports.
    """

    @abstractmethod
    def __init__(self, cfg: Any) -> None:
        """Initialize the evaluator.

        Args:
            cfg: Configuration object from Hydra containing evaluator settings.
        """
        self.cfg = cfg

    @abstractmethod
    def run(self, tracker_state: TrackerState) -> None:
        """Run the evaluation on the provided tracker state.

        Args:
            tracker_state: The tracker state containing predictions and ground truth
                data to evaluate.
        """
        pass
