import json
import logging
from collections import defaultdict
from typing import List, Tuple, Union

from pathlib import Path

import numpy as np

from catchemi.dftdos.dftdos import MultipleAdsorbateAndMetalChemisorption
from catchemi.dftdos.fast_dftdos import FastMultipleMetalChemisorption


class BaseFitParameters:
    """Given a json file with the dos and energy stored, and
    some information about epsilon_a values, perform the fitting
    procedure to determine alpha, beta and a _single_ constant.

    Args:
        json_filename (str): The name of the json file containing the dos and energy.
        eps_a_data (list): The list of epsilon_a values to use.
        DEBUG (bool): Whether to print debug information.
        Delta0 (float): The value of Delta0 to use, fixed constant.
        return_extended_output (bool): Whether to return the extended output.
    """

    def __init__(
        self,
        json_filename: str,
        eps_a_data: List,
        Delta0: float,
        DEBUG: bool = False,
        return_extended_output: bool = False,
        no_of_bonds_list: List[int] = None,
    ):
        self.json_filename = json_filename
        self.eps_a_data = eps_a_data
        self.Delta0 = Delta0
        self.DEBUG = DEBUG
        self.return_extended_output = return_extended_output
        self.no_of_bonds_list = no_of_bonds_list

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate inputs to make sure everything has
        the right dimensions and type."""
        assert isinstance(self.eps_a_data, list), "eps_a_data must be a list"
        assert isinstance(self.Delta0, float), "Delta0 must be a float"

        # Convert the eps_a_data to a numpy array.
        self.eps_a_data = np.array(self.eps_a_data)

    def load_data(self):
        """Load the data from the json file."""
        with open(self.json_filename, "r") as f:
            data = json.load(f)
        self.data = data

    def objective_function(
        self,
        x: Tuple,
    ) -> float:
        """Objective function to be minimised.

        Args:
            x (Tuple): The parameters to be used in the minimisation.
                       Include alpha, beta and gamma (fixed constant)
                       and a list of epsilon values, epsilon is used
                       to determine the scaling between each of the adsorbates.
        """

        alpha, beta, gamma, *epsilon = x

        alpha = np.abs(alpha)
        beta = np.abs(beta)
        epsilon = np.abs(epsilon)

        factor_mult_list = [1] + [epsilon[i] for i in range(len(epsilon))]
        factor_mult_list = np.array(factor_mult_list)

        inputs = defaultdict(lambda: defaultdict(dict))

        inputs["Delta0"] = self.Delta0
        inputs["eps_a_data"] = self.eps_a_data
        inputs["DEBUG"] = self.DEBUG
        inputs["no_of_bonds_list"] = self.no_of_bonds_list

        for _id in self.data:
            pdos = self.data[_id]["pdos"]
            energy_grid = self.data[_id]["energy_grid"]
            Vsd = self.data[_id]["Vsd"]
            Vsd = np.array(Vsd)

            inputs["dos_data"][_id]["dft_dos"] = pdos
            inputs["dos_data"][_id]["eps"] = energy_grid
            inputs["Vak_data"][_id] = factor_mult_list * np.sqrt(beta) * Vsd
            inputs["Sak_data"][_id] = -factor_mult_list * alpha * Vsd

        model_energies_class = MultipleAdsorbateAndMetalChemisorption(**inputs)
        model_outputs = model_energies_class.model_outputs

        mean_absolute_error = 0.0
        for _id in model_outputs:
            ads_energy_DFT = self.data[_id]["ads_energy"]
            model_energy = 0.0
            for eps_a in model_outputs[_id]:
                model_energy += model_outputs[_id][eps_a]["chemi"]

            model_energy += gamma

            sq_error = np.abs(model_energy - ads_energy_DFT)
            mean_absolute_error += sq_error

        mean_absolute_error = mean_absolute_error / len(model_outputs)

        logging.info(
            f"Parameters: {x} leads to mean absolute error: {mean_absolute_error}eV"
        )

        if self.return_extended_output:

            predicted_energy = []
            actual_energy = []
            id_order = []

            species_string = []

            for _id in model_outputs:
                ads_energy_DFT = self.data[_id]["ads_energy"]
                actual_energy.append(ads_energy_DFT)

                model_energy = 0.0

                for eps_a in model_outputs[_id]:
                    model_energy += model_outputs[_id][eps_a]["chemi"]

                model_energy += gamma
                predicted_energy.append(model_energy)

                id_order.append(_id)

                spec_string = "".join(self.data[_id]["species"])
                species_string.append(spec_string)

            logging.info("Predicted and actual energies parsed.")

            output_data = {}
            output_data["predicted_energy"] = predicted_energy
            output_data["actual_energy"] = actual_energy
            output_data["id_order"] = id_order
            output_data["species_string"] = species_string

            return mean_absolute_error, output_data

        else:
            return mean_absolute_error


class FastFitParameters(FastMultipleMetalChemisorption):
    def __init__(
        self,
        adsorption_energy_filename: Union[str, Path],
        Vsd_filename: Union[str, Path],
        pdos_filename: Union[str, Path],
        energy_filename: Union[str, Path],
        Delta0: float,
        eps_a: List[float],
        indices_to_keep: List[int],
    ):
        self.adsorption_energies = np.loadtxt(adsorption_energy_filename, skiprows=1)
        self.Vsd = np.loadtxt(Vsd_filename, skiprows=1)
        self.pdos = np.loadtxt(pdos_filename, skiprows=1)
        self.energy = np.loadtxt(energy_filename, skiprows=1)
        self.indices_to_keep = indices_to_keep

        self.adsorption_energies = self.adsorption_energies[self.indices_to_keep]
        self.Vsd = self.Vsd[self.indices_to_keep]
        self.pdos = self.pdos[self.indices_to_keep]
        self.energy = self.energy[self.indices_to_keep]

        self.cleanup_data()
        self.generate_rho_d()
        self.Delta0 = Delta0
        if isinstance(eps_a, float):
            self._eps_a = np.array([eps_a])
        self._eps_a = eps_a

    def cleanup_data(self):
        """Remove the nan entries in self.adsorption_energies and corresponding
        entries in self.Vsdsq and self.pdos."""
        self.adsorption_energies = self.adsorption_energies.reshape(-1, 1)
        mask = np.isnan(self.adsorption_energies)
        self.adsorption_energies = self.adsorption_energies[~mask]
        self.adsorption_energies = self.adsorption_energies.reshape(-1, 1)

        self.Vsd = self.Vsd.reshape(-1, 1)
        self.Vsd = self.Vsd[~mask]
        self.Vsdsq = self.Vsd**2
        self.Vsdsq = self.Vsdsq.reshape(-1, 1)

        self.pdos = self.pdos[~mask.reshape(-1), :]
        self.energy = self.energy[~mask.reshape(-1), :]

    def generate_rho_d(self):
        """Generate the d-density of states by normalising the pdos."""
        normalization = np.trapz(self.pdos, x=self.energy, axis=-1)
        self.rho_d = self.pdos / normalization.reshape(-1, 1)

    def objective_function(
        self, x, loss_function: str = "mean_absolute_error"
    ) -> float:
        """Generate the objective function.
        The order of the parameters is:
        alpha1, alpha2... beta1, beta2... gamma
        """
        alpha = x[: len(self._eps_a)]
        beta = x[len(self._eps_a) : -1]
        gamma = x[-1]

        model_energies = np.zeros_like(self.adsorption_energies)

        for idx, eps_a in enumerate(self._eps_a):

            Vaksq = beta[idx] * self.Vsdsq
            S = -alpha[idx] * np.sqrt(Vaksq)

            super().__init__(
                Delta0=self.Delta0,
                eps_a=eps_a,
                rho_d=self.rho_d,
                energy=self.energy,
                Vaksq=Vaksq,
                S=S,
                constant=gamma,
            )

            model_energies += self()

        model_energies = np.array(model_energies).reshape(-1, 1)
        self.model_energies = model_energies

        if loss_function == "mean_absolute_error":
            mean_absolute_error = np.mean(
                np.abs(self.adsorption_energies - model_energies)
            )
            return mean_absolute_error
        elif loss_function == "root_mean_squared_error":
            mean_squared_error = np.mean(
                (self.adsorption_energies - model_energies) ** 2
            )
            root_mean_squared_error = np.sqrt(mean_squared_error)
            return root_mean_squared_error
