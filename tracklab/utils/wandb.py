import logging
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import pandas as pd
from omegaconf import OmegaConf

try:
    import wandb
except ImportError:
    pass

logger = logging.getLogger(__name__)

# FIXME not sure it is the right to do that. It is annoying to update this every time we add a new config
keep_dict = {
    "dataset": ["dataset_path", "nframes", "nvid", "vids_dict"],
    "detect_multiple": [
        "min_confidence",
        "path_to_config",
        "path_to_checkpoint",
        "instance_min_confidence",
        "keypoint_min_confidence",
        "bbox",
        "predict",
        "train",
    ],
    "detect_single": [
        "min_keypoints_score",
        "min_keypoints_confidence",
        "path_to_config",
        "path_to_checkpoint",
        "bbox",
        "predict",
        "train",
    ],
    "eval": ["mot"],
    "reid": ["data", "loss", "model", "sampler", "test", "train", "dataset"],
    "track": True,
}


def normalize_subdict(subdict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize subdictionary for wandb logging.

    Args:
        subdict: Dictionary to normalize.

    Returns:
        Normalized dictionary.
    """
    if "_target_" in subdict:
        subdict["target"] = subdict.pop("_target_")
    if "cfg" in subdict:
        for k, v in subdict["cfg"].items():
            subdict[k] = v
        del subdict["cfg"]
    return subdict


def init(cfg: Any) -> None:
    """Initialize wandb logging.

    Args:
        cfg: Configuration object.
    """
    global use_wandb
    use_wandb = cfg.use_wandb
    if use_wandb:
        kwargs = {}
        if "wandb" in cfg:
            kwargs = cfg.wandb
        cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg["experiment_name"], config=cfg, **kwargs)


def log_metric(
    res_dict: Dict[str, Any], name: str, video_dict: Optional[Dict[str, Any]] = None
) -> None:
    """Log metrics to wandb.

    Args:
        res_dict: Dictionary of metrics to log.
        name: Name prefix for the metrics.
        video_dict: Optional video metrics dictionary.
    """
    if use_wandb:
        try:
            wandb.log(
                {f"{name}/{k}": v for k, v in res_dict.items()},
                step=0,
            )
            if video_dict is not None:
                video_df = pd.DataFrame.from_dict(video_dict, orient="index")
                video_df.insert(0, "video", video_df.index)
                wandb.log({f"{name}/videos": video_df}, step=0)
        except wandb.Error:
            logger.warning("Wandb error, skipping logging")
            pass


def log(res_dict: Dict[str, Any]) -> None:
    """Log dictionary to wandb.

    Args:
        res_dict: Dictionary to log.
    """
    if use_wandb:
        try:
            wandb.log(res_dict)
        except wandb.Error:
            logger.warning("Wandb error, skipping logging")
            pass


def apply_recursively(
    d: Any,
    f: Callable[[Any], Any] = lambda v: v,
    filter: Callable[[str, Any], bool] = lambda k, v: True,
    always_filter: bool = False,
) -> Any:
    """Apply a function to leaf values of a dict recursively and/or filter dict.

    Args:
        d: Dictionary to process.
        f: Function taking the value of a leaf as argument and returning
           a transformation of that value.
        filter: Condition to apply f to only (sub-)branches of the tree.
        always_filter: If true filter sub-branches, if false only filter leaves.

    Returns:
        Transformed and filtered dict.
    """
    for k, v in d.items():
        if isinstance(v, Mapping):
            if always_filter:
                d[k] = apply_recursively(v, f, filter) if filter(k, v) else v
            else:
                d[k] = apply_recursively(v, f, filter, always_filter)
        elif isinstance(v, str):
            d[k] = f(v) if filter(k, v) else v
        elif isinstance(v, Sequence):
            if len(v) == 0:
                d[k] = v
            elif isinstance(v[0], Mapping):
                d[k] = [apply_recursively(val, f, filter) for val in v]
            else:
                d[k] = [f(val) if filter(k, val) else val for val in v]
        else:
            d[k] = f(v) if filter(k, v) else v
    return d


def finish() -> None:
    """Finish wandb logging."""
    if use_wandb:
        wandb.finish()
