import abc

from typing import Sequence

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass

from catchemi.input import BaseInput, CombinedInput, FixedDimCombinedInput


@dataclass
class BaseFitting(CombinedInput, abc.ABC):
    inputs: Sequence[BaseInput]

    @abc.abstractmethod
    def get_predicted_chemisorption_energies(self, *args, **kwargs) -> float:
        pass
