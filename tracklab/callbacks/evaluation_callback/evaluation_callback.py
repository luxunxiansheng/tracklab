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
    ) -> None:
        """
        Initialize the evaluation callback.

        Args:
            eval_cfg: Configuration for the evaluator, typically instantiated by Hydra.
            export_path: Path where tracking results were exported, if applicable.
        """
        self.eval_cfg: Any = eval_cfg
        self.export_path: Optional[str] = export_path

    def on_dataset_track_end(self, engine: "TrackingEngine") -> None:
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
