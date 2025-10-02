import logging
import re
from abc import ABC, ABCMeta
from typing import List, Dict, Set, Optional, Union, Any

log = logging.getLogger(__name__)


class MetaModule(ABCMeta):
    @property
    def name(cls) -> str:
        name = cls.__name__
        return name  # re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def level(cls) -> str:
        name = cls.__bases__[0].__name__
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.split("_")[0]


class Module(metaclass=ABCMeta):
    """Base class for all pipeline modules.

    Modules process data at different levels (detection, image, video) and
    define their input/output column requirements.
    """

    input_columns: Optional[Union[List[str], Dict[str, List[str]]]] = None
    output_columns: Optional[Union[List[str], Dict[str, List[str]]]] = None
    training_enabled: bool = False
    forget_columns: List[str] = []

    @property
    def name(self) -> str:
        """Get the module name."""
        name = self.__class__.__name__
        return name  # re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def level(self) -> str:
        """Get the processing level of the module."""
        name = self.__class__.__bases__[0].__name__
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.split("_")[0]

    def validate_input(self, dataframe) -> None:
        """Validate that the input dataframe contains required columns.

        Args:
            dataframe: Input dataframe to validate.
        """
        assert self.input_columns is not None, "Every model should define its inputs"
        for col in self.input_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The input detection should contain {col}.")

    def validate_output(self, dataframe) -> None:
        """Validate that the output dataframe contains expected columns.

        Args:
            dataframe: Output dataframe to validate.
        """
        assert self.output_columns is not None, "Every model should define its outputs"
        for col in self.output_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The output detection should contain {col}.")

    def get_input_columns(self, level: str) -> List[str]:
        """Get input columns for the specified level.

        Args:
            level: Processing level ('detection', 'image', etc.).

        Returns:
            List of required input column names.
        """
        if isinstance(self.input_columns, list):
            return self.input_columns if level == "detection" else []
        elif isinstance(self.input_columns, dict):
            return self.input_columns.get(level, [])
        else:
            return []

    def get_output_columns(self, level: str) -> List[str]:
        """Get output columns for the specified level.

        Args:
            level: Processing level ('detection', 'image', etc.).

        Returns:
            List of output column names.
        """
        if isinstance(self.output_columns, list):
            return self.output_columns if level == "detection" else []
        elif isinstance(self.output_columns, dict):
            return self.output_columns.get(level, [])
        else:
            return []


class Pipeline:
    """Manages a sequence of processing modules."""

    def __init__(self, modules: List[Module]) -> None:
        """Initialize the pipeline with a list of modules.

        Args:
            modules: List of modules to include in the pipeline.
        """
        self.modules = [module for module in modules if module.name != "skip"]
        log.info("Pipeline: " + " -> ".join(module.name for module in self.modules))

    def validate(self, load_columns: Dict[str, Set[str]]) -> None:
        """Validate that the pipeline can process the loaded columns.

        Args:
            load_columns: Dictionary mapping levels to sets of available columns.
        """
        columns = {k: set(v) for k, v in load_columns.items()}
        for level in ["image", "detection"]:
            for module in self.modules:
                if module.input_columns is None or module.output_columns is None:
                    raise AttributeError(
                        f"{type(module)} should contain input_ and output_columns"
                    )
                if not set(module.get_input_columns(level)).issubset(columns[level]):
                    raise AttributeError(
                        f"The {module.name} model doesn't have "
                        "all the input needed, "
                        f"needed {module.get_input_columns(level)}, provided {columns[level]}"
                    )
                columns[level].update(module.get_output_columns(level))
        log.info(f"Pipeline has been validated")

    def __str__(self) -> str:
        return " -> ".join(module.name for module in self.modules)

    def __getitem__(self, item: int) -> Module:
        return self.modules[item]

    def is_empty(self) -> bool:
        """Check if the pipeline has no modules."""
        return len(self.modules) == 0


class Skip(Module):
    """A module that does nothing, used for skipping processing."""

    def __init__(self, **kwargs) -> None:
        """Initialize the Skip module."""
        pass

    @property
    def name(self) -> str:
        return "skip"
