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
        """Get the predicted chemisorption energies.

        Based on the paramaters alpha, beta, gamma, Delta0, and eps_a,
        compute the chemisorption energies using the `FixedDimensionCalculation`
        model.

        Parameters
        ----------
        alpha : npt.ArrayLike
            The alpha parameter, expected to be of shape (num_parameters, spin_dimension).
        beta : npt.ArrayLike
            The beta parameter, expected to be of shape (num_parameters, spin_dimension).
        gamma : float
            The gamma parameter, a single constant which is added to each chemisorption energy.
        Delta0 : float
            The Delta0 parameter, a single constant which is used in the calculation of the
            chemisorption energy (added to Delta).
        eps_a : npt.ArrayLike
            The eps_a parameter, expected to be of shape (num_parameters, spin_dimension).

        Returns
        -------
        chemisorption_energies : float
            The predicted chemisorption energies, expected to be of shape (num_inputs, 1).

        See Also
        --------
        FixedDimensionCalculation

        Examples
        --------
        >>> import numpy as np
        >>> from catchemi.input import BaseInput
        >>> from catchemi.fitting import FixedDimFitting
        >>> from catchemi.core import FixedDimensionCalculation
        >>> # Create some dummy inputs for two different metals
        >>> inputs = [
        ...     BaseInput(
        ...         eps=np.random.rand(10),
        ...         pdos=np.random.rand(10),
        ...         coupling_sd=np.random.rand(10),
        ...         dft_energy=-1.0,
        ...         spin_polarized=False,
        ...     ),
        ...     BaseInput(
        ...         eps=np.random.rand(10),
        ...         pdos=np.random.rand(10),
        ...         coupling_sd=np.random.rand(10),
        ...         dft_energy=-2.0,
        ...         spin_polarized=False,
        ...     ),
        ... ]
        >>> # Create the fitting object with one parameter
        >>> # i.e. only one alpha and beta value
        >>> fitting = FixedDimFitting(inputs=inputs)
        >>> alpha = np.random.rand(1, 1)
        >>> beta = np.random.rand(1, 1)
        >>> gamma = 0.0
        >>> Delta0 = 1.0
        >>> eps_a = np.random.rand(1, 1)
        >>> chemisorption_energies = fitting.get_predicted_chemisorption_energies(
        ...     alpha=alpha,
        ...     beta=beta,
        ...     gamma=gamma,
        ...     Delta0=Delta0,
        ...     eps_a=eps_a,
        ... )
        >>> # The expected output is a 2x1 array, chemisorption energies for each input metal
        >>> assert chemisorption_energies.shape == (2, 1)
        """

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
        """Get the mean absolute error of the predicted chemisorption energies.

        Based on the paramaters alpha, beta, gamma, Delta0, and eps_a, compute
        the chemisorption energies using the `FixedDimensionCalculation` model and
        compare them to the DFT chemisorption energies.

        Parameters
        ----------
        alpha : npt.ArrayLike
            The alpha parameter, expected to be of shape (num_parameters, spin_dimension).
        beta : npt.ArrayLike
            The beta parameter, expected to be of shape (num_parameters, spin_dimension).
        gamma : float
            The gamma parameter, a single constant which is added to each chemisorption energy.
        Delta0 : float
            The Delta0 parameter, a single constant which is used in the calculation of the
            chemisorption energy (added to Delta).
        eps_a : npt.ArrayLike
            The eps_a parameter, expected to be of shape (num_parameters, spin_dimension).

        Returns
        -------
        mean_absolute_error : float
            The mean absolute error of the predicted chemisorption energies compared to the
            DFT chemisorption energies.

        See Also
        --------
        FixedDimensionCalculation
        """

        predicted_chemisorption_energies = self.get_predicted_chemisorption_energies(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            Delta0=Delta0,
            eps_a=eps_a,
        )
        dft_chemisorption_energies = self.get_combined_dft_energy()
        predicted_chemisorption_energies = predicted_chemisorption_energies.reshape(-1)
        dft_chemisorption_energies = dft_chemisorption_energies.reshape(-1)
        return np.mean(
            np.abs(predicted_chemisorption_energies - dft_chemisorption_energies)
        )

    def get_least_squares_error(
        self,
        alpha: npt.ArrayLike,
        beta: npt.ArrayLike,
        gamma: float,
        Delta0: float,
        eps_a: npt.ArrayLike,
    ):
        """Get the least squares error of the predicted chemisorption energies.

        Based on the paramaters alpha, beta, gamma, Delta0, and eps_a, compute
        the chemisorption energies using the `FixedDimensionCalculation` model and
        compare them to the DFT chemisorption energies.

        Parameters
        ----------
        alpha : npt.ArrayLike
            The alpha parameter, expected to be of shape (num_parameters, spin_dimension).
        beta : npt.ArrayLike
            The beta parameter, expected to be of shape (num_parameters, spin_dimension).
        gamma : float
            The gamma parameter, a single constant which is added to each chemisorption energy.
        Delta0 : float
            The Delta0 parameter, a single constant which is used in the calculation of the
            chemisorption energy (added to Delta).
        eps_a : npt.ArrayLike
            The eps_a parameter, expected to be of shape (num_parameters, spin_dimension).

        Returns
        -------
        least_squares_error : float
            The least squares error of the predicted chemisorption energies compared to the
            DFT chemisorption energies.

        See Also
        --------
        FixedDimensionCalculation
        """

        predicted_chemisorption_energies = self.get_predicted_chemisorption_energies(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            Delta0=Delta0,
            eps_a=eps_a,
        )
        dft_chemisorption_energies = self.get_combined_dft_energy()
        predicted_chemisorption_energies = predicted_chemisorption_energies.reshape(-1)
        dft_chemisorption_energies = dft_chemisorption_energies.reshape(-1)
        return np.sum(
            (predicted_chemisorption_energies - dft_chemisorption_energies) ** 2
        )
