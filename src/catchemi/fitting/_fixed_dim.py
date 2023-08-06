from typing import Sequence

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from catchemi.input import BaseInput, FixedDimCombinedInput
from catchemi.core import FixedDimensionCalculation


@dataclass
class FixedDimFitting(FixedDimCombinedInput):
    inputs: Sequence[BaseInput]

    def error(self, *args, **kwargs) -> float:
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        Delta0 = kwargs["Delta0"]
        gamma = kwargs["gamma"]
        eps_a = kwargs["eps_a"]
        kappa = kwargs.get("kappa", None)

        eps = self.get_combined_eps()
        pdos = self.get_combined_pdos()
        coupling_sd = self.get_combined_coupling_sd()
