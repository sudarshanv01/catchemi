import abc

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BaseInput:
    """Base class that defines characteristics of the surface.

    This class defines quantities that are related to the surface that
    come from DFT (or other) calculations.

    Parameters
    ----------
    eps : npt.ArrayLike
        Grid of energies (in eV) at which the PDOS is defined. The expected
        shape of the matrix is (spin_dimension, number_of_points). If a
        calculation is not spin polarized, the spin_dimension is automatically
        set to 1.
    pdos : npt.ArrayLike
        Partial density of states (PDOS) of the surface. The expected shape
        of the matrix is (spin_dimension, number_of_points). If a calculation
        is not spin polarized, the spin_dimension is automatically set to 1.
    coupling_sd : npt.ArrayLike
        Coupling element between s-orbitals of the adsorbate and the d-bands
        of the surface. The expected shape of the matrix is (spin_dimension, 1)
    dft_energy: float
        DFT computed energy of chemisorption of the adsorbate on the surface.
        Typically, this is the energy of the adsorbate on the surface minus
        the energy of the adsorbate and the surface separately.
    spin_polarized : bool
        Whether the calculation is spin polarized or not. If True, the
        spin_dimension of the eps, pdos and coupling_sd matrices must be 2. Else,
        the spin_dimension must be 1 (or is automatically set to 1).

    Examples
    --------
    For non-spin polarized calculations, the spin_dimension is automatically
    set to 1, as shown in the following example:

    >>> eps = np.linspace(-10, 10, 100)
    >>> pdos = np.abs(np.random.rand(100))
    >>> coupling_sd = np.random.rand(1)
    >>> input = BaseInput(eps, pdos, coupling_sd)
    >>> input.eps.shape
    (1, 100)
    >>> input.pdos.shape
    (1, 100)
    >>> input.coupling_sd.shape
    (1, 1)

    For spin polarized calculations, the spin_dimension must be 2, as shown
    in the following example:

    >>> _eps = np.linspace(-10, 10, 100)
    >>> eps = np.vstack((_eps, _eps))
    >>> _pdos = np.abs(np.random.rand(100))
    >>> pdos = np.vstack((_pdos, _pdos))
    >>> coupling_sd = np.random.rand(2).reshape(-1, 1)
    >>> input = BaseInput(eps, pdos, coupling_sd, spin_polarized=True)
    >>> input.eps.shape
    (2, 100)
    >>> input.pdos.shape
    (2, 100)
    >>> input.coupling_sd.shape
    (2, 1)
    """

    eps: npt.ArrayLike
    pdos: npt.ArrayLike
    coupling_sd: npt.ArrayLike
    dft_energy: float
    spin_polarized: bool = False

    def __post_init__(self):
        self.eps = np.asarray(self.eps)
        self.pdos = np.asarray(self.pdos)
        self.coupling_sd = np.asarray(self.coupling_sd)

        if not self.spin_polarized:
            if self.eps.ndim == 1:
                self.eps = self.eps.reshape(1, -1)
            if self.pdos.ndim == 1:
                self.pdos = self.pdos.reshape(1, -1)
            if self.coupling_sd.ndim <= 1:
                self.coupling_sd = self.coupling_sd.reshape(-1, 1)

    def __repr__(self):
        return f"""{self.__class__.__name__}:
                    eps: {self.eps}
                    pdos: {self.pdos}
                    coupling_sd: {self.coupling_sd}
                    """

    @property
    def number_of_points(self):
        return self.eps.shape[-1]
