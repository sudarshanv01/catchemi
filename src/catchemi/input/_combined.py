import abc

from dataclasses import dataclass

from typing import Sequence

import numpy as np
import numpy.typing as npt

from catchemi.input import BaseInput


@dataclass
class CombinedInput(abc.ABC):
    """Abstract base class for combined input.

    This class is used to combine multiple inputs into one. This
    class is not meant to be used directly, but rather to be
    subclassed.
    """

    inputs: Sequence[BaseInput]

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]

    def __len__(self):
        return len(self.inputs)

    def __next__(self):
        return next(self.inputs)

    @abc.abstractmethod
    def get_combined_eps(self):
        pass

    @abc.abstractmethod
    def get_combined_pdos(self):
        pass

    @abc.abstractmethod
    def get_combined_coupling_sd(self):
        pass


@dataclass
class FixedDimCombinedInput(CombinedInput):
    """Combine outputs that all have a fixed dimension.

    This class is used to combine multiple inputs into one. This
    class is useful for further processing of the density of states
    and to run different models. The shapes of the spin polarised
    dimension is preserved and an extra dimension is added for each
    input.

    Parameters
    ----------
    inputs : Sequence[BaseInput]
        A sequence of inputs to combine. Each input must have the
        same shape.

    Raises
    ------
    AssertionError
        If the number of points is not the same for all inputs.

    Examples
    --------
    >>> from catchemi.input import BaseInput, FixedDimCombinedInput
    >>> eps = np.linspace(-10, 10, 100) # Dummy energies
    >>> pdos = np.abs(np.random.rand(100)) # Dummy PDOS
    >>> coupling_sd = np.random.rand(1) # Dummy coupling element
    >>> input1 = BaseInput(eps, pdos, coupling_sd)
    >>> input2 = BaseInput(eps, pdos, coupling_sd)
    >>> sequence_of_inputs = [input1, input2]
    >>> combined_input = FixedDimCombinedInput(sequence_of_inputs)
    >>> combined_input.get_combined_eps().shape # (num_of_inputs, spin_dimension, number_of_points)
    (2, 1, 100)
    >>> combined_input.get_combined_pdos().shape # (num_of_inputs, spin_dimension, number_of_points)
    (2, 1, 100)
    """

    inputs: Sequence[BaseInput]

    def _check_shapes(self):
        num_of_points = [input.number_of_points for input in self.inputs]
        assert (
            len(set(num_of_points)) == 1
        ), "Number of points must be the same for all inputs."

    def _check_same_spin_dimension(self):
        spin_dimension = [input.spin_polarized for input in self.inputs]
        assert (
            len(set(spin_dimension)) == 1
        ), "Spin dimension must be the same for all inputs."

    def __post_init__(self):
        self._check_shapes()
        self._check_same_spin_dimension()

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def is_spin_polarized(self):
        return self.inputs[0].spin_polarized

    def get_combined_eps(self):
        return np.asarray([input.eps for input in self.inputs])

    def get_combined_pdos(self):
        return np.asarray([input.pdos for input in self.inputs])

    def get_combined_coupling_sd(self):
        return np.asarray([input.coupling_sd for input in self.inputs])

    def get_combined_dft_energy(self):
        return np.asarray([input.dft_energy for input in self.inputs])
