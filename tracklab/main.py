import os
from pathlib import Path

from cv2 import exp
import rich.logging
import torch
import hydra
import warnings
import logging
from typing import TYPE_CHECKING

from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import progress, wandb

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


@hydra.main(
    version_base=None, config_path="pkg://tracklab.configs", config_name="config"
)
def main(cfg: DictConfig) -> int:
    """Main entry point for the TrackLab tracking pipeline.

    Initializes the environment, instantiates dataset and modules based on the
    configuration, trains modules if enabled, and runs tracking inference.

    Args:
        cfg: Hydra configuration object containing all settings.

    Returns:
        Exit code (0 for success).
    """
    device = init_environment(cfg)

    # Instantiate all modules
    tracking_dataset = instantiate(cfg.dataset)

    modules = []
    if cfg.module_order is not None:
        for name in cfg.module_order:
            module = cfg.pipeline[name]
            inst_module = instantiate(
                module, device=device, tracking_dataset=tracking_dataset
            )
            modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    # Train tracking modules
    training_modules = [module for module in modules if module.training_enabled]
    if training_modules:
        for i, module in enumerate(training_modules, 1):
            module.train(
                tracking_dataset,
                pipeline,
                OmegaConf.to_container(cfg.dataset, resolve=True),
            )
            log.info(f"✅ Finished training module {i}/{len(training_modules)}")

    # Infer tracking
    if cfg.infer_tracking:
        # Init tracker state and tracking engine
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
        tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )

        # Run tracking and visualization
        tracking_engine.track_dataset()

        # Save tracker state
        if tracker_state.save_file is not None:
            log.info(f"💾 Saved state at : {tracker_state.save_file.resolve()}")

    close_environment()

    return 0


def set_sharing_strategy() -> None:
    """Set PyTorch multiprocessing sharing strategy to file_system for compatibility."""
    torch.multiprocessing.set_sharing_strategy("file_system")


def init_environment(cfg: DictConfig) -> str:
    """Initialize the tracking environment and return the device to use.

    Sets up progress reporting, multiprocessing strategy, device detection,
    logging configuration, and Weights & Biases initialization.

    Args:
        cfg: Hydra configuration object.

    Returns:
        The device string ('cuda', 'mps', or 'cpu').
    """
    # For Hydra and Slurm compatibility
    progress.use_rich = cfg.use_rich
    set_sharing_strategy()  # Do not touch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if cfg.use_rich:
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.INFO)
    wandb.init(cfg)
    log.info(f"Run directory: {Path().absolute()}")
    log.info(f"Using device: '{device}'.")

    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    return device


def close_environment() -> None:
    """Close the tracking environment, finishing Weights & Biases logging."""
    wandb.finish()


if __name__ == "__main__":
    main()
