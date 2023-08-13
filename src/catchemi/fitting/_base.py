import abc

from typing import Sequence

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass

from catchemi.input import BaseInput, CombinedInput, FixedDimCombinedInput


@dataclass
class BaseFitting(CombinedInput, abc.ABC):
    """Abstract base class for fitting.

    This class is used to fit the model to the data. This class is
    not meant to be used directly, but rather to be subclassed.
    """

    inputs: Sequence[BaseInput]

    @abc.abstractmethod
    def get_predicted_chemisorption_energies(self, *args, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def get_mean_absolute_error(self, *args, **kwargs) -> float:
        pass
