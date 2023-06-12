import numpy as np

from typing import List, Dict, Any, Tuple, Union

from pathlib import Path

import numpy.typing as npt

import scipy
from scipy import optimize
from scipy import signal


class FastMultipleMetalChemisorption:
    def __init__(
        self,
        Delta0: float,
        eps_a: float,
        rho_d: npt.ArrayLike,
        energy: npt.ArrayLike,
        Vaksq: npt.ArrayLike,
        S: npt.ArrayLike,
        constant: float = 0,
        eps_f: float = 0,
    ):
        """Calculate the chemisorption energy for a single metal with a constant
        energy grid on which the density of states is computed. This vectorized
        version is much faster than the fixed grid version.

        Args:
            Delta0 (float): The bare adsorbate-metal hybridisation energy.
            eps_a (float): The adsorbate energy.
            rho_d (npt.ArrayLike): The adsorbate density of states.
            energy (npt.ArrayLike): The energy grid.
            Vaksq (npt.ArrayLike): The adsorbate-metal hybridisation matrix
                elements.
            S (npt.ArrayLike): The overlap matrix elements.
            constant (float, optional): A constant energy offset. Defaults to 0.
            eps_f (float, optional): The Fermi energy. Defaults to 0.
        """
        self.rho_d = rho_d
        self.energy = energy
        self.Delta0 = Delta0
        self.eps_a = eps_a
        self.Vaksq = Vaksq
        self.S = S
        self.constant = constant

        self.mask = np.zeros_like(self.energy)
        self.mask[self.energy < eps_f] = 1

    def __call__(self, *args: Any, **kwgs: Any) -> Any:
        self.generate_Delta()
        self.generate_Lambda()
        self.generate_hybridisation_energy()
        self.generate_rho_aa()
        self.generate_na()
        self.generate_f()
        self.generate_orthogonalization_energy()

        total_energy = (
            self.hybridisation_energy + self.orthogonalization_energy + self.constant
        )
        return total_energy

    def generate_Delta(self):
        """Delta = pi V_{ak}^2 rho_d + Delta_0"""
        self.Delta = np.pi * self.Vaksq * self.rho_d
        self.Delta += self.Delta0

    def generate_Lambda(self):
        """Calculate the Hilbert transform for each row of Delta."""
        self.Lambda = np.imag(scipy.signal.hilbert(self.Delta, axis=-1))

    def get_arctan_integrand(self):
        """Generate the integrand for the arctan integral."""
        arctan_integrand_numerator = self.Delta
        arctan_integrand_denominator = self.energy - self.eps_a - self.Lambda
        arctan_integrand = np.arctan2(
            arctan_integrand_numerator, arctan_integrand_denominator
        )
        arctan_integrand -= np.pi
        return arctan_integrand

    def generate_hybridisation_energy(self):
        """Generate the hybridisation energy."""

        arctan_integrand = self.get_arctan_integrand()
        arctan_integrand *= self.mask

        self.hybridisation_energy = np.trapz(arctan_integrand, x=self.energy, axis=-1)
        self.hybridisation_energy = self.hybridisation_energy.reshape(-1, 1)
        self.hybridisation_energy /= np.pi
        self.hybridisation_energy *= 2
        self.hybridisation_energy -= 2 * self.eps_a

    def generate_rho_aa(self):
        """Calculate the adsorbate density of states."""
        rho_aa = self.Delta
        rho_aa /= (self.energy - self.eps_a - self.Lambda) ** 2 + self.Delta**2
        rho_aa /= np.pi
        self.rho_aa = rho_aa

    def generate_na(self):
        """Calculate the occupancy of the adsorbate."""
        na_integrand = self.rho_aa * self.mask
        na = np.trapz(na_integrand, x=self.energy, axis=-1)
        na = na.reshape(-1, 1)
        self.na = na

    def generate_f(self):
        """Calculate the filling of the metal."""
        f_integrand_numerator = self.Delta * self.mask
        f_integrand_denominator = self.Delta

        f = np.trapz(f_integrand_numerator, x=self.energy, axis=-1) / np.trapz(
            f_integrand_denominator, x=self.energy, axis=-1
        )
        f = f.reshape(-1, 1)
        self.f = f

    def generate_orthogonalization_energy(self):
        """Calculate the orthogonalization energy."""
        self.orthogonalization_energy = (
            -2 * (self.na + self.f) * np.sqrt(self.Vaksq) * self.S
        )
