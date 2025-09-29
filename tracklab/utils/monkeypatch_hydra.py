import functools
from typing import Any, Callable, Dict, Tuple, Union

import torch
from hydra._internal.instantiate import _instantiate2
from hydra._internal.instantiate._instantiate2 import (
    _convert_target_to_string,
    _extract_pos_args,
)
from hydra._internal.utils import _locate
from hydra.errors import InstantiationException
from omegaconf import OmegaConf


def new_call_target(
    _target_: Callable[..., Any],
    _partial_: bool,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    try:
        args, kwargs = _extract_pos_args(args, kwargs)
        # detaching configs from parent.
        # At this time, everything is resolved and the parent link can cause
        # issues when serializing objects in some scenarios.
        for arg in args:
            if OmegaConf.is_config(arg):
                arg._set_parent(None)
        for v in kwargs.values():
            if OmegaConf.is_config(v):
                v._set_parent(None)
    except Exception as e:
        msg = (
            f"Error in collecting args and kwargs for '{_convert_target_to_string(_target_)}':"
            + f"\n{repr(e)}"
        )
        if full_key:
            msg += f"\nfull_key: {full_key}"

        raise InstantiationException(msg) from e

    if _partial_:
        try:
            return functools.partial(_target_, *args, **kwargs)
        except Exception as e:
            msg = (
                f"Error in creating partial({_convert_target_to_string(_target_)}, ...) object:"
                + f"\n{repr(e)}"
            )
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    else:
        return _target_(*args, **kwargs)


def new_resolve_target(
    target: Union[str, type, Callable[..., Any]], full_key: str
) -> Union[type, Callable[..., Any]]:
    """Resolve target string, type or callable into type or callable."""
    if isinstance(target, str):
        target = _locate(target)
    if not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise InstantiationException(msg)
    return target


_instantiate2._call_target = new_call_target
_instantiate2._resolve_target = new_resolve_target

# Monkeypatch torch.load to handle weights_only issue
try:
    original_torch_load = torch.load

    def patched_torch_load(
        fpath: Any,
        map_location: Any = None,
        pickle_module: Any = None,
        *,
        weights_only: Any = None,
        **kwargs: Any,
    ) -> Any:
        try:
            return original_torch_load(
                fpath,
                map_location=map_location,
                pickle_module=pickle_module,
                weights_only=weights_only,
                **kwargs,
            )
        except Exception as e:
            if "Weights only load failed" in str(e) or "WeightsUnpickler error" in str(
                e
            ):
                # Retry with weights_only=False for trusted checkpoints
                if str(fpath).endswith(".pth.tar") or "prtreid" in str(fpath):
                    return original_torch_load(
                        fpath,
                        map_location=map_location,
                        pickle_module=pickle_module,
                        weights_only=False,
                        **kwargs,
                    )
            raise

    torch.load = patched_torch_load
except ImportError:
    pass
