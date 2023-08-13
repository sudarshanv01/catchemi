from typing import Sequence

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from catchemi.input import BaseInput, FixedDimCombinedInput
from catchemi.core import FixedDimensionCalculation


@dataclass
class FixedDimFitting(FixedDimCombinedInput):
    """Fit the model to the data.

    This class is used to fit the model to the data. Starting from
    the combined input, the predicted chemisorption energy is computed
    for each input. The predicted chemisorption energy is can be
    compared to the DFT chemisorption energies to determine the best
    parameters for the fit.

    Parameters
    ----------
    inputs : Sequence[BaseInput]
        The inputs to fit the model to.
    """

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
                    spin_polarized=self.is_spin_polarized,
                )
                _chemisorption_energies = calculation.get_chemisorption_energy()
                chemisorption_energies += _chemisorption_energies
        chemisorption_energies += gamma
        return chemisorption_energies

    def get_mean_absolute_error(
        self,
        alpha: npt.ArrayLike,
        beta: npt.ArrayLike,
        gamma: float,
        Delta0: float,
        eps_a: npt.ArrayLike,
    ) -> float:
        predicted_chemisorption_energies = self.get_predicted_chemisorption_energies(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            Delta0=Delta0,
            eps_a=eps_a,
        )
        dft_chemisorption_energies = self.get_combined_dft_energy()
        return np.mean(
            np.abs(predicted_chemisorption_energies - dft_chemisorption_energies)
        )
