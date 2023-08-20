import abc

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import scipy
from scipy import signal

from catchemi.core import BaseCalculation


@dataclass
class FixedDimensionCalculation(BaseCalculation):
    """Perform calculations for the fixed dimension model.

    This class is used to perform calculations for when the inputs
    are performed at a fixed dimension. Starting from the inputs of
    energies, projected density of states, and coupling strengths,
    the hybridization energy, orthogonalization energy, and
    chemisorption energy are computed.

    Parameters
    ----------
    eps : npt.ArrayLike
        Grid of energies.
    pdos : npt.ArrayLike
        Projected density of states on the grid of energies.
    coupling_sd : npt.ArrayLike
        Coupling strengths of the s-orbital of the adsorbate and
        the d-orbital of the surface.
    eps_a : float
        Energy of the adsorbate state.
    alpha : float
        Parameter responsible for controlling orthogonalization energy.
    beta : float
        Parameter responsible for controlling hybridization energy and
        chemisorption energy. Scaling factor for Vsd and Vak.
    Delta0 : float
        Constant shift of the normalized projected density of states.
    spin_polarized : bool, optional
        Whether or not the system is spin polarized. If True, the
        spin factor is 1. If False, the spin factor is 2.
    eps_f : float, optional
        Fermi energy. Default is 0.

    Examples
    --------
    >>> import numpy as np
    >>> from catchemi.core import FixedDimensionCalculation
    >>> eps = np.linspace(-10, 10, 100) # Energy grid from DFT
    >>> eps = eps.reshape(1, -1) # Reshape to make first dimension input dim
    >>> pdos = np.abs(np.random.rand(100)) # Projected density of states from DFT
    >>> pdos = pdos.reshape(1, -1) # Reshape to match the energy grid
    >>> coupling_sd = np.random.rand(1) # Coupling strength parameter
    >>> coupling_sd = coupling_sd.reshape(1, -1) # Reshape to match the energy grid
    >>> eps_a = 0.0 # Adsorbate energy
    >>> alpha = 1.0 # Orthogonalization parameter
    >>> beta = 0.2 # Hybridization parameter
    >>> Delta0 = 0.5 # Constant shift of the normalized projected density of states
    >>> spin_polarized = False # Whether or not the system is spin polarized
    >>> eps_f = 0.0 # Fermi energy
    >>> fixed_dim_calc = FixedDimensionCalculation(
    ...     eps=eps,
    ...     pdos=pdos,
    ...     coupling_sd=coupling_sd,
    ...     eps_a=eps_a,
    ...     alpha=alpha,
    ...     beta=beta,
    ...     Delta0=Delta0,
    ...     spin_polarized=spin_polarized,
    ...     eps_f=eps_f,
    ... )
    >>> hybridization_energy = fixed_dim_calc.get_hybridization_energy()
    >>> orthogonalization_energy = fixed_dim_calc.get_orthogonalization_energy()
    >>> chemisorption_energy = fixed_dim_calc.get_chemisorption_energy()
    >>> assert hybridization_energy.shape == (1, 1)
    >>> assert orthogonalization_energy.shape == (1, 1)
    >>> assert chemisorption_energy.shape == (1, 1)
    >>> assert hybridization_energy <= 0
    >>> assert orthogonalization_energy >= 0
    >>> assert chemisorption_energy <= 0
    """

    eps: npt.ArrayLike
    pdos: npt.ArrayLike
    coupling_sd: npt.ArrayLike
    eps_a: float
    alpha: float
    beta: float
    Delta0: float
    spin_polarized: bool = False
    eps_f: float = 0

    def __post_init__(self):
        if self.spin_polarized:
            self.spin_factor = 1
        else:
            self.spin_factor = 2

        mask = np.zeros_like(self.eps)
        mask[self.eps < self.eps_f] = 1
        self.mask = mask

    def normalize_pdos(self, pdos: npt.ArrayLike, eps: npt.ArrayLike) -> npt.ArrayLike:
        """Normalize the projected density of states.

        Make sure taht the projected density of states is normalized.
        Denoted as rho_aa in most literature.

        Parameters
        ----------
        pdos : npt.ArrayLike
            Projected density of states.

        Returns
        -------
        npt.ArrayLike
            Normalized projected density of states.
        """
        integral = np.trapz(pdos, eps, axis=-1)
        integral = integral.reshape(-1, 1)
        return pdos / integral

    def get_Delta(self, Vaksq: npt.ArrayLike, rho_d: npt.ArrayLike) -> npt.ArrayLike:
        """Get the Delta parameter.

        The Delta parameter is computed for the d-bands and the sp-bands. The d-bands
        contribution is given by,
        .. math::
            \\Delta_{d} = \\pi V_{ak}^{2} \\rho_{d} + \\Delta_{0}
        and the total contribution is given by,
        .. math::
            \\Delta = \\Delta_{d} + \\Delta_{0}
        where :math:`\\Delta_{0}` is the contribution from the sp-bands and is
        assumed constants throughout the energy grid.

        Parameters
        ----------
        Vaksq : npt.ArrayLike
            Coupling strength of the orbital of the adsorbate and the d-orbital
            of the surface.
        rho_d : npt.ArrayLike
            Normalizd projected density of states of the d-orbital of the surface.

        Returns
        -------
        npt.ArrayLike
            Delta parameter.
        """
        Delta = np.pi * Vaksq * rho_d
        Delta += self.Delta0
        return Delta

    def get_Lambda(self, Delta: npt.ArrayLike) -> npt.ArrayLike:
        """Get the Lambda parameter.

        The Lambda parameter is simply the Hilbert transform of the Delta parameter.

        Parameters
        ----------
        Delta : npt.ArrayLike
            Delta parameter.

        Returns
        -------
        npt.ArrayLike
            Lambda parameter.
        """
        Lambda = np.imag(scipy.signal.hilbert(Delta, axis=-1))
        return Lambda

    def _get_arctan_integrand(
        self,
        Delta: npt.ArrayLike,
        Lambda: npt.ArrayLike,
        eps: npt.ArrayLike,
        eps_a: float,
    ) -> npt.ArrayLike:
        arctan_integrand_numerator = Delta
        arctan_integrand_denominator = eps - eps_a - Lambda
        arctan_integrand = np.arctan2(
            arctan_integrand_numerator, arctan_integrand_denominator
        )
        arctan_integrand -= np.pi
        return arctan_integrand

    def get_hybridization_energy(self) -> npt.ArrayLike:
        """Get the hybridization energy.

        The hybridization energy is given by,
        .. math::
            E_{hyb} = \\frac{1}{\\pi} \\int_{-\\infty}^{\\infty} d\\epsilon \\frac{\\Delta(\\epsilon)}{\\epsilon - \\epsilon_{a} - \\Lambda(\\epsilon)}
        where :math:`\\Delta(\\epsilon)` is the Delta parameter, :math:`\\epsilon_{a}` is the
        adsorbate energy level and :math:`\\Lambda(\\epsilon)` is the Lambda parameter all on
        the energy grid :math:`\\epsilon` for each spin channel.
        The integral is evaluated using the trapezoidal rule.

        Returns
        -------
        npt.ArrayLike
            Hybridization energy (in eV).
        """
        Vaksq = self.beta * self.coupling_sd**2
        rho_d = self.normalize_pdos(self.pdos, self.eps)
        Delta = self.get_Delta(Vaksq, rho_d)
        Lambda = self.get_Lambda(Delta)
        arctan_integrand = self._get_arctan_integrand(
            Delta, Lambda, self.eps, self.eps_a
        )
        arctan_integrand *= self.mask
        hybridisation_energy = np.trapz(arctan_integrand, x=self.eps, axis=-1)
        hybridisation_energy = hybridisation_energy.reshape(-1, 1)
        hybridisation_energy /= np.pi
        hybridisation_energy *= self.spin_factor
        hybridisation_energy -= self.spin_factor * self.eps_a
        return hybridisation_energy

    def get_adsorbate_density_of_states(
        self,
        Delta: npt.ArrayLike,
        Lambda: npt.ArrayLike,
        eps: npt.ArrayLike,
        eps_a: float,
    ) -> npt.ArrayLike:
        """Get the adsorbate density of states.

        The adsorbate density of states is given by,
        .. math::
            \\rho_{aa} = \\frac{1}{\\pi} \\frac{\\Delta}{(\\epsilon - \\epsilon_{a} - \\Lambda)^{2} + \\Delta^{2}}
        where :math:`\\Delta` is the Delta parameter, :math:`\\epsilon_{a}` is the
        adsorbate energy level and :math:`\\Lambda` is the Lambda parameter all on
        the energy grid :math:`\\epsilon` for each spin channel.
        """
        rho_aa = Delta
        rho_aa /= (eps - eps_a - Lambda) ** 2 + Delta**2
        rho_aa /= np.pi
        return rho_aa

    def _get_occupancy(
        self, rho_aa: npt.ArrayLike, eps: npt.ArrayLike
    ) -> npt.ArrayLike:
        na_integrand = rho_aa * self.mask
        na = np.trapz(na_integrand, x=eps, axis=-1)
        na = na.reshape(-1, 1)
        return na

    def _get_filling(self, Delta: npt.ArrayLike, eps: npt.ArrayLike) -> npt.ArrayLike:
        f_integrand_numerator = Delta * self.mask
        f_integrand_denominator = Delta
        filling = np.trapz(f_integrand_numerator, x=eps, axis=-1) / np.trapz(
            f_integrand_denominator, x=eps, axis=-1
        )
        filling = filling.reshape(-1, 1)
        return filling

    def _get_overlap(self, Vaksq: npt.ArrayLike, alpha: float) -> npt.ArrayLike:
        overlap = -1 * np.sqrt(Vaksq) * alpha
        return overlap

    def get_orthogonalization_energy(self) -> npt.ArrayLike:
        """Get the orthogonalization energy.

        The orthogonalization energy is given by,
        .. math::
            E_{ortho} = - (n_{a} + f) \\sqrt{V_{ak}^{2}} \\alpha
        for the adsorbate density of states :math:`\\rho_{aa}`, the adsorbate energy level
        :math:`\\epsilon_{a}`, the filling :math:`f`, the occupancy :math:`n_{a}`, the
        overlap :math:`\\alpha` and the coupling :math:`V_{ak}` in the two-state
        approximation.

        Returns
        -------
        npt.ArrayLike
            Orthogonalization energy (in eV).
        """
        Vaksq = self.beta * self.coupling_sd**2
        rho_d = self.normalize_pdos(self.pdos, self.eps)
        Delta = self.get_Delta(Vaksq, rho_d)
        Lambda = self.get_Lambda(Delta)
        rho_aa = self.get_adsorbate_density_of_states(
            Delta, Lambda, self.eps, self.eps_a
        )
        occupancy = self._get_occupancy(rho_aa, self.eps)
        filling = self._get_filling(Delta, self.eps)
        overlap = self._get_overlap(Vaksq, self.alpha)
        orthogonalization_energy = -self.spin_factor * (occupancy + filling)
        orthogonalization_energy *= np.sqrt(Vaksq)
        orthogonalization_energy *= overlap
        return orthogonalization_energy

    def get_chemisorption_energy(self) -> npt.ArrayLike:
        """Get the chemisorption energy.

        The chemisorption energy is given by,
        .. math::
            E_{chem} = E_{hyb} + E_{ortho}
        where :math:`E_{hyb}` is the hybridization energy and :math:`E_{ortho}` is the
        orthogonalization energy.

        Returns
        -------
        npt.ArrayLike
            Chemisorption energy (in eV).
        """
        hybridization_energy = self.get_hybridization_energy()
        orthogonalization_energy = self.get_orthogonalization_energy()
        chemisorption_energy = hybridization_energy + orthogonalization_energy
        return chemisorption_energy
