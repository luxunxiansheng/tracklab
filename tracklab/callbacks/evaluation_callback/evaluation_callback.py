import logging
from typing import TYPE_CHECKING, Any, Optional

from hydra.utils import instantiate

from tracklab.callbacks.callback import Callback

if TYPE_CHECKING:
    from tracklab.engine import TrackingEngine

log = logging.getLogger(__name__)


class EvaluationCallback(Callback):
    """Callback that runs evaluation on tracking results at the end of dataset tracking.

    This callback instantiates and runs an evaluator on the complete tracking results
    after all videos have been processed. It can optionally skip evaluation based on
    configuration.
    """

    after_saved_state = True

    def __init__(
        self,
        eval_cfg: Any,
        export_path: Optional[str] = None,
        eval_tracking: bool = True,
    ):
        """Initialize the evaluation callback.

        Args:
            eval_cfg: Configuration for the evaluator, typically instantiated by Hydra.
            export_path: Path where tracking results were exported, if applicable.
            eval_tracking: Whether to perform evaluation after tracking.
        """
        self.eval_cfg = eval_cfg
        self.export_path = export_path
        self.eval_tracking = eval_tracking

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        """Run evaluation on the entire dataset after tracking completes.

        Args:
            engine: The tracking engine instance containing the tracker state.
        """
        if self.eval_tracking:
            try:
                log.info("Starting evaluation...")

                # eval_cfg should already be an instantiated evaluator from Hydra
                evaluator = self.eval_cfg
                # Update the export_path and tracking_dataset
                evaluator.export_path = self.export_path
                evaluator.tracking_dataset = engine.tracker_state.tracking_set

                evaluator.run(engine.tracker_state)
                log.info("Evaluation completed.")
            except Exception as e:
                log.error(f"Evaluation failed: {e}")
        else:
            log.info("Evaluation skipped (eval_tracking=False)")
