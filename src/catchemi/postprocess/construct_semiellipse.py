import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as nptyp

from catchemi.dftdos.dftdos import BaseChemisorption


class SemiEllipseHypothetical(BaseChemisorption):
    def __init__(
        self,
        json_params: str,
        eps_d_list: nptyp.ArrayLike,
        w_d_list: nptyp.ArrayLike,
        eps: nptyp.ArrayLike,
        Vsd: float,
    ):
        """Given an arbitrary set of eps_d and w_d values, create
        a meshgrid of the chemisorption, hybridisation and orthogonalisation energies.
        Multiple eps_a is supported and added together.

        Args:
            json_params (str): The path to the json file containing the parameters.
            eps_d_list (nptyp.ArrayLike): The list of eps_d values.
            w_d_list (nptyp.ArrayLike): The list of w_d values.
            eps (nptyp.ArrayLike): The energy range.
            Vsd (float): The value of Vsd.
        """
        self.json_params = json_params
        self.eps_d_list = eps_d_list
        self.w_d_list = w_d_list
        self.eps = eps
        self.Vsd = Vsd

        self.GRID_SIZE = len(self.eps_d_list)
        assert self.GRID_SIZE == len(
            self.w_d_list
        ), "The number of eps_d and w_d must be equal"

        self._read_parameters()

    def _read_parameters(self) -> None:
        """Read the input parameters from the input file."""
        with open(self.json_params) as handle:
            data = json.load(handle)
        self.eps_a_list = data["eps_a"]
        self.alpha = data["alpha"]
        self.beta = data["beta"]
        self.gamma = data["gamma"]
        self.factor_mult_list = [1] + data["epsilon"]
        self.Delta0 = data["delta0"]
        self.no_of_bonds_list = data["no_of_bonds_list"]

    def _create_semi_ellipse(self, eps_d: float, w_d: float) -> None:
        """Create a semi-ellipse for the d-density of states."""
        energy_ref = (self.eps - eps_d) / w_d
        delta = np.zeros(len(self.eps))
        for i, eps_ in enumerate(energy_ref):
            if np.abs(eps_) < 1:
                delta[i] = (1 - eps_**2) ** 0.5
        # We will need a very tight grid in order to make
        # the normalisation perfect. So in this case, we will
        # proceed by setting the integral of the semi-ellipse to 1.
        delta /= np.trapz(delta, self.eps)
        # This parameter is out `dft_dos`, even though it is constructed
        self.dft_dos = delta

    def generate_meshgrid(
        self,
    ) -> Tuple[nptyp.ArrayLike, nptyp.ArrayLike, nptyp.ArrayLike]:
        """Generate meshgrid of chemisorption, hybridisation
        and orthogonalisation energies."""

        self.hyb_energy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.chem_energy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.ortho_energy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.occupancy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

        for index_epsd, eps_d in enumerate(self.eps_d_list):
            for index_wd, w_d in enumerate(self.w_d_list):

                # Define the d-density of states.
                self._create_semi_ellipse(eps_d, w_d)

                # If there are more than one eps_a values, add them up.
                for index_epsa, eps_a in enumerate(self.eps_a_list):

                    # Generate the adsorbate specific parameters
                    self.eps_a = eps_a
                    self.no_of_bonds = self.no_of_bonds_list[index_epsa]
                    self.Vak = (
                        self.factor_mult_list[index_epsa]
                        * np.sqrt(self.beta)
                        * self.Vsd
                    )
                    self.Sak = (
                        self.factor_mult_list[index_epsa] * -1 * self.alpha * self.Vak
                    )

                    self._validate_inputs()

                    # Get the chemisorption and hybridisation energies.
                    self.hyb_energy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_hybridisation_energy()
                    self.ortho_energy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_orthogonalisation_energy()
                    self.chem_energy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_chemisorption_energy()

                    # Store also the occupancies
                    self.occupancy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_occupancy()

        # Add gamma to the chemisorption energy.
        self.chem_energy_meshgrid += self.gamma
        self.hyb_energy_meshgrid += self.gamma

        return (
            self.hyb_energy_meshgrid,
            self.ortho_energy_meshgrid,
            self.chem_energy_meshgrid,
            self.occupancy_meshgrid,
        )
