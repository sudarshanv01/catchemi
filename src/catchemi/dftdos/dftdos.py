"""Perform a simple calculation to get the chemisorption energy."""
import os
import logging
import json
from typing import Dict, List, Tuple

from collections import defaultdict

import numpy as np
import numpy.typing as nptyp

import scipy
from scipy import optimize
from scipy import signal

import matplotlib.pyplot as plt


class BaseChemisorption:
    # Occassionally, during the optimisation process
    # we will have a situation where the E_hyb is positive
    # This positive number is numerically caused. Allow
    # a little bit of noise (but E_hyb) will be set to 0
    # if the value is greater than 0 and below INTEGRAL_NOISE.
    INTEGRAL_NOISE = 0.3

    def __init__(
        self,
        dft_dos: List[float],
        dft_energy_range: List[float],
        Vak: float,
        Sak: float,
        Delta0: float,
        eps_a: float,
        no_of_bonds: int,
    ):
        """Base class to perform a simple calculation
        to get the chemisorption energy where the energy grid
        is different.

        Args:
            dft_dos (List[float]): The d-projected density of states.
            dft_energy_range (List[float]): The energy range of the d-projected density of states.
            Vak (float): The value of Vak.
            Sak (float): The value of Sak.
            Delta0 (float): The value of Delta0.
            eps_a (float): The value of epsilon_a.
            no_of_bonds (int): The number of bonds.
        """
        self.dft_dos = dft_dos
        self.eps = dft_energy_range
        self.Vak = Vak
        self.Sak = Sak
        self.Delta0 = Delta0
        self.eps_a = eps_a
        self.no_of_bonds = no_of_bonds

    def __call__(self) -> float:
        self._validate_inputs()
        self.get_Delta()
        self.get_Lambda()
        self.get_hybridisation_energy()
        self.get_orthogonalisation_energy()
        self.get_chemisorption_energy()

        return self.E_chem

    def _validate_inputs(self):
        """Throw up assertion errors if the inputs
        are not in accordance with what is expected."""
        # Ensure that the integral of the density of states is 1.
        # That is, the d-projected density of states is normalised
        # to 1.
        integral_d_dos = np.trapz(self.dft_dos, self.eps)
        assert np.allclose(integral_d_dos, 1.0)
        assert self.Sak <= 0.0, "Sak must be negative or zero."
        assert self.Vak >= 0.0, "Vak must be positive"

        # Make everything a numpy array
        self.dft_dos = np.array(self.dft_dos)
        self.eps = np.array(self.eps)

        # Delta0 must be a float and greater than 0
        assert isinstance(self.Delta0, float), "Delta0 must be a float"
        assert self.Delta0 >= 0.0, "Delta0 must be positive"

    def get_Delta(self) -> nptyp.ArrayLike:
        """Get Vak by multiplying the d-density of
        states by the following relations:
        Delta = pi * Vak**2 * rho_d
        where rho_d is the DFT (normalised) density
        of states.
        """
        self.Delta = np.pi * self.Vak**2 * self.dft_dos
        return self.Delta

    def get_Lambda(self) -> nptyp.ArrayLike:
        """Get the Hilbert transform of the Delta array."""
        self.Lambda = np.imag(scipy.signal.hilbert(self.Delta + self.Delta0))
        return self.Lambda

    def get_chemisorption_energy(self) -> float:
        """The chemisorption energy is the sum of the
        hybridisation energy and the orthogonalisation energy."""

        self.E_chemi = self.E_hyb + self.E_ortho
        logging.debug(
            f"Chemisorption energy: {self.E_chemi:0.2f} eV, Hybridisation energy: {self.E_hyb:0.2f} eV, Orthogonalisation energy: {self.E_ortho:0.2f} eV; n_a = {self.n_a:0.2f} e"
        )
        return self.E_chemi

    def get_hybridisation_energy(self) -> float:
        """Get the hybridisation energy on the basis
        of the DFT density of states and the Newns-Anderson
        model energy."""

        # Refresh Delta and Lambda
        self.get_Delta()
        self.get_Lambda()

        ind_below_fermi = np.where(self.eps <= 0)[0]

        # Create the integral and just numerically integrate it.
        integrand_numer = self.Delta[ind_below_fermi] + self.Delta0
        integrand_denom = (
            self.eps[ind_below_fermi] - self.eps_a - self.Lambda[ind_below_fermi]
        )

        arctan_integrand = np.arctan2(integrand_numer, integrand_denom)
        arctan_integrand -= np.pi

        assert np.all(arctan_integrand <= 0), "Arctan integrand must be negative"
        assert np.all(
            arctan_integrand >= -np.pi
        ), "Arctan integrand must be greater than -pi"

        E_hyb = np.trapz(arctan_integrand, self.eps[ind_below_fermi])

        E_hyb *= 2
        E_hyb /= np.pi

        E_hyb -= 2 * self.eps_a

        # Multiply by the number of bonds
        E_hyb *= self.no_of_bonds

        self.E_hyb = E_hyb

        logging.debug(
            f"Hybridisation energy: {self.E_hyb:0.2f} eV for eps_a: {self.eps_a:0.2f} eV"
        )
        try:
            assert self.E_hyb <= 0.0, "Hybridisation energy must be negative"
        except AssertionError as e:
            if self.E_hyb < self.INTEGRAL_NOISE:
                logging.warning(
                    f"Numerically, E_hyb is very slightly greater than 0 ({self.E_hyb:0.2f} eV). Setting it to 0."
                )
                self.E_hyb = 0.0
            else:
                raise e

        return self.E_hyb

    def get_adsorbate_dos(self) -> nptyp.ArrayLike:
        """Get the adsorbate projected density of states."""
        numerator = self.Delta + self.Delta0
        denominator = (self.eps - self.eps_a - self.Lambda) ** 2
        denominator += (self.Delta + self.Delta0) ** 2
        self.rho_a = numerator / denominator / np.pi
        return self.rho_a

    def get_occupancy(self) -> float:
        """Integrate up rho_a upto the Fermi level."""
        # Integrate up rho_a upto the Fermi level.
        index_ = np.where(self.eps <= 0)[0]
        self.n_a = np.trapz(self.rho_a[index_], self.eps[index_])
        return self.n_a

    def get_filling(self) -> float:
        """Integrate up Delta to get the filling of
        the d-density of states of the metal."""
        denominator = np.trapz(self.Delta, self.eps)
        # Get the index of self.eps that are lower than 0
        index_ = np.where(self.eps < 0)[0]
        numerator = np.trapz(self.Delta[index_], self.eps[index_])
        self.filling = numerator / denominator
        assert self.filling >= 0.0, "Filling must be positive"
        return self.filling

    def get_orthogonalisation_energy(self) -> float:
        """Get the orthogonalisation energy on the basis of the
        smeared out two-state equation."""
        self.get_Delta()
        self.get_Lambda()
        self.get_adsorbate_dos()
        self.get_occupancy()
        self.get_filling()
        E_ortho = -2 * (self.n_a + self.filling) * self.Vak * self.Sak
        # Mutliply by the number of bonds
        E_ortho *= self.no_of_bonds
        self.E_ortho = E_ortho
        assert self.E_ortho >= 0.0, "Orthogonalisation energy must be positive"
        return self.E_ortho


class MultipleAdsorbateAndMetalChemisorption(BaseChemisorption):
    def __init__(
        self,
        dos_data: Dict,
        Vak_data: Dict,
        Sak_data: Dict,
        Delta0: float,
        eps_a_data: List,
        no_of_bonds_list: List[int],
    ):

        """Perform the adsorbate chemisorption analysis
        based on an arbitrary set of parameters (both metal and
        adsorbate).

        Args:
            dos_data (Dict): A dictionary of dictionaries
                containing the DFT density of states for each
                surface.
            Vak_data (Dict): A dictionary of lists containing
                the Vak values for each surface.
            Sak_data (Dict): A dictionary of lists containing
                the Sak values for each surface.
            Delta0 (float): The value of the hybridisation
                energy at the Fermi level.
            eps_a_data (List): A list of the adsorbate energies
                for each surface.
            no_of_bonds_list (List[int]): A list of the number
                of bonds for each surface.
        """

        self.Delta0 = Delta0

        model_outputs = defaultdict(dict)

        for identifier, dos_dict in dos_data.items():
            self.dft_dos = dos_dict["dft_dos"]
            self.eps = dos_dict["eps"]
            Vak_list = Vak_data[identifier]
            Sak_list = Sak_data[identifier]

            for i, eps_a in enumerate(eps_a_data):
                self.eps_a = eps_a

                # Store the float values of Vak and Sak
                self.Vak = Vak_list[i]
                self.Sak = Sak_list[i]
                self.no_of_bonds = no_of_bonds_list[i]

                self._validate_inputs()

                e_hyb = self.get_hybridisation_energy()
                e_ortho = self.get_orthogonalisation_energy()
                e_chemi = self.get_chemisorption_energy()

                model_outputs[identifier][eps_a] = {
                    "hyb": e_hyb,
                    "ortho": e_ortho,
                    "chemi": e_chemi,
                }
        self.model_outputs = model_outputs
