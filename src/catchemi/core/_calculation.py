import abc

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import scipy
from scipy import signal


@dataclass
class BaseCalculation(abc.ABC):
    eps: npt.ArrayLike
    pdos: npt.ArrayLike
    coupling_sd: npt.ArrayLike
    eps_a: float
    alpha: float
    beta: float
    Delta0: float
    spin_polarized: bool = False
    eps_f: float = 0

    @abc.abstractmethod
    def get_hybridization_energy(self) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def get_orthogonalization_energy(self) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def get_chemisorption_energy(self) -> npt.ArrayLike:
        pass


@dataclass
class FixedDimensionCalculation(BaseCalculation):
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

    def normalize_pdos(self, pdos: npt.ArrayLike) -> npt.ArrayLike:
        return pdos / np.sum(pdos, axis=1)[:, None]

    def get_Delta(self, Vaksq: npt.ArrayLike, rho_aa: npt.ArrayLike) -> npt.ArrayLike:
        Delta = self.beta * Vaksq * rho_aa
        Delta += self.Delta0
        return Delta

    def get_Lambda(self, Delta: npt.ArrayLike) -> npt.ArrayLike:
        Lambda = np.imag(scipy.signal.hilbert(Delta, axis=-1))
        return Lambda

    def get_arctan_integrand(
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
        Vaksq = self.beta * self.coupling_sd**2
        rho_d = self.normalize_pdos(self.pdos)
        Delta = self.get_Delta(Vaksq, rho_d)
        Lambda = self.get_Lambda(Delta)
        arctan_integrand = self.get_arctan_integrand(
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
        rho_aa = Delta
        rho_aa /= (eps - eps_a - Lambda) ** 2 + Delta**2
        rho_aa /= np.pi
        return rho_aa

    def get_occupancy(self, rho_aa: npt.ArrayLike, eps: npt.ArrayLike):
        na_integrand = rho_aa * self.mask
        na = np.trapz(na_integrand, x=eps, axis=-1)
        na = na.reshape(-1, 1)
        return na

    def get_filling(self, Delta: npt.ArrayLike, eps: npt.ArrayLike):
        f_integrand_numerator = Delta * self.mask
        f_integrand_denominator = Delta
        filling = np.trapz(f_integrand_numerator, x=eps, axis=-1) / np.trapz(
            f_integrand_denominator, x=eps, axis=-1
        )
        filling = filling.reshape(-1, 1)
        return filling

    def get_overlap(self, Vaksq: npt.ArrayLike, alpha: float) -> npt.ArrayLike:
        overlap = -1 * np.sqrt(Vaksq) * alpha
        return overlap

    def get_orthogonalization_energy(self) -> npt.ArrayLike:
        Vaksq = self.beta * self.coupling_sd**2
        rho_d = self.normalize_pdos(self.pdos)
        Delta = self.get_Delta(Vaksq, rho_d)
        Lambda = self.get_Lambda(Delta)
        rho_aa = self.get_adsorbate_density_of_states(
            Delta, Lambda, self.eps, self.eps_a
        )
        occupancy = self.get_occupancy(rho_aa, self.eps)
        filling = self.get_filling(Delta, self.eps)
        overlap = self.get_overlap(Vaksq, self.alpha)
        orthogonalization_energy = -self.spin_factor * (occupancy + filling)
        orthogonalization_energy *= np.sqrt(Vaksq)
        orthogonalization_energy *= overlap
        return orthogonalization_energy

    def get_chemisorption_energy(self) -> npt.ArrayLike:
        hybridization_energy = self.get_hybridization_energy()
        orthogonalization_energy = self.get_orthogonalization_energy()
        chemisorption_energy = hybridization_energy + orthogonalization_energy
        return chemisorption_energy
