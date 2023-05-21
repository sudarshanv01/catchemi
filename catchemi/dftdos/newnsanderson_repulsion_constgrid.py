import numpy as np

from typing import List, Dict, Any, Tuple, Union

from pathlib import Path

import numpy.typing as npt

import scipy
from scipy import optimize
from scipy import signal


class ConstantGridChemisorption:
    def __init__(
        self,
        Delta0: float,
        eps_a: float,
        rho_d: npt.ArrayLike,
        energy: npt.ArrayLike,
        Vaksq: npt.ArrayLike,
        S: npt.ArrayLike,
        constant: float = 0,
    ):
        self.rho_d = rho_d
        self.energy = energy
        self.Delta0 = Delta0
        self.eps_a = eps_a
        self.Vaksq = Vaksq
        self.S = S
        self.constant = constant

        self.mask = np.zeros_like(self.energy)
        self.mask[self.energy < self.eps_a] = 1

    def __call__(self, *args: Any, **kwgs: Any) -> Any:
        self.generate_Delta()
        self.generate_Lambda()
        self.generate_hybridisation_energy()
        self.generate_rho_aa()
        self.generate_na()
        self.generate_f()
        self.generate_orthogonalization_energy()

        return self.hybridisation_energy + self.orthogonalization_energy + self.constant

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

        arctan_integrand = self.generate_arctan_integrand()
        arctan_integrand *= self.mask

        self.hybridisation_energy = np.trapz(arctan_integrand, x=self.energy, axis=-1)

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
        self.na = na

    def generate_f(self):
        """Calculate the filling of the metal."""
        f_integrand_numerator = self.Delta * self.mask
        f_integrand_denominator = self.Delta

        f = np.trapz(f_integrand_numerator, x=self.energy, axis=-1) / np.trapz(
            f_integrand_denominator, x=self.energy, axis=-1
        )
        self.f = f

    def generate_orthogonalization_energy(self):
        """Calculate the orthogonalization energy."""
        self.orthogonalization_energy = (
            -2 * (self.na + self.f) * np.sqrt(self.Vaksq) * self.S
        )


class ConstantGridFittingParameters(ConstantGridChemisorption):
    def __init__(
        self,
        adsorption_energy_filename: Union[str, Path],
        Vsdsq_filename: Union[str, Path],
        pdos_filename: Union[str, Path],
        Delta0: float,
        eps_a: float,
    ):
        self.adsorption_energies = np.loadtxt(adsorption_energy_filename, skiprows=1)
        self.Vsdsq = np.loadtxt(Vsdsq_filename, skiprows=1)
        self.pdos = np.loadtxt(pdos_filename, skiprows=1)
        self.remove_nan_entries()
        self.generate_rho_d()

    def remove_nan_entries(self):
        """Remove the nan entries in self.adsorption_energies and corresponding
        entries in self.Vsdsq and self.pdos."""
        mask = np.isnan(self.adsorption_energies)
        self.adsorption_energies = self.adsorption_energies[~mask]
        self.Vsdsq = self.Vsdsq[~mask]
        self.pdos = self.pdos[~mask]

    def generate_rho_d(self):
        """Generate the d-density of states by normalising the pdos."""
        self.rho_d = self.pdos / np.trapz(self.pdos, x=self.energy, axis=-1)

    def objective_function(self, alpha, beta, gamma):
        """Generate the objective function."""
        Vaksq = beta * self.Vsdsq
        S = -alpha * np.sqrt(self.Vsdsq)

        super().__init__(
            Delta0=self.Delta0,
            eps_a=self.eps_a,
            rho_d=self.rho_d,
            energy=self.energy,
            Vaksq=Vaksq,
            S=S,
            constant=gamma,
        )

        return self()
