from typing import Sequence

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from catchemi.input import BaseInput, FixedDimCombinedInput
from catchemi.core import FixedDimensionCalculation


@dataclass
class FixedDimFitting(FixedDimCombinedInput):
    inputs: Sequence[BaseInput]

    def get_predicted_chemisorption_energies(
        self,
        alpha: npt.ArrayLike,
        beta: npt.ArrayLike,
        gamma: float,
        Delta0: float,
        eps_a: npt.ArrayLike,
    ) -> float:
        eps = self.get_combined_eps()
        pdos = self.get_combined_pdos()
        coupling_sd = self.get_combined_coupling_sd()
        chemisorption_energies = np.zeros((eps.shape[0], 1))

        if eps.ndim == 2:
            eps = eps[:, None, :]
            pdos = pdos[:, None, :]
            coupling_sd = coupling_sd[:, None, :]

        for idx_a, _eps_a in enumerate(eps_a):
            for idx_s, _eps_as in enumerate(_eps_a):
                calculation = FixedDimensionCalculation(
                    eps=eps[:, idx_s, ...],
                    pdos=pdos[:, idx_s, ...],
                    coupling_sd=coupling_sd[:, idx_s],
                    eps_a=_eps_as,
                    alpha=alpha[idx_a, idx_s],
                    beta=beta[idx_a, idx_s],
                    Delta0=Delta0,
                )
                _chemisorption_energies = calculation.get_chemisorption_energy()
                chemisorption_energies += _chemisorption_energies
        chemisorption_energies += gamma
        return chemisorption_energies
