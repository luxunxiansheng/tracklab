import logging
from typing import TYPE_CHECKING

from hydra.utils import instantiate
from tracklab.callbacks.callback import Callback

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class EvaluationCallback(Callback):
    """
    Callback that runs evaluation on the tracking results at the end of dataset tracking.
    """

    def __init__(self, eval_cfg, export_cfg, eval_tracking: bool = True):
        """
        Initialize the evaluation callback.

        Args:
            eval_cfg: Configuration for the evaluator
            export_cfg: Configuration for the exporter (needed by some evaluators)
            eval_tracking: Whether to perform evaluation
        """
        self.eval_cfg = eval_cfg
        self.export_cfg = export_cfg
        self.eval_tracking = eval_tracking

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        """Run evaluation on the entire dataset."""
        if self.eval_tracking:
            try:
                log.info("Starting evaluation...")
                exporter = self.export_cfg
                evaluator = instantiate(
                    self.eval_cfg,
                    tracking_dataset=engine.tracker_state.tracking_set,
                    exporter=exporter,
                )
                evaluator.run(engine.tracker_state)
                log.info("Evaluation completed.")
            except Exception as e:
                log.error(f"Evaluation failed: {e}")
        else:
            log.info("Evaluation skipped (eval_tracking=False)")
